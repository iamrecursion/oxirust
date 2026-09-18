//! A per-case wall-clock budget, originally built for the QT3 drivers and now shared by
//! [`crate::xslt30`] too — a catalogue this size needs the same guarantee regardless of which
//! engine is under test, and *why* is the same reason in both places: see below.
//!
//! The catalogue contains queries this workspace's engines do not terminate on. Before the
//! catalogue-root (`shared`) environments were merged in, the drivers never reached them and the
//! full run finished in minutes; with those environments available the XQuery driver stopped
//! finishing at all. A driver that hangs measures nothing, so every case now runs on a worker
//! thread under [`CASE_BUDGET`] and a case that overruns is recorded as a **failure** whose
//! message starts with [`TIMEOUT_MARKER`] — never as a skip, and never silently: a case this
//! engine cannot finish is a conformance failure, and it has to be visible in the failure dump so
//! that it can be root-caused. XSLT30's own catalogue has no case known to hang the same way, but
//! it inherits the same bound for a different reason: run through `oxixml_testsuite::all_suites`/
//! `oxixml-conformance-report`, it runs *after* QT3, and a QT3 case that timed out leaves its
//! worker thread abandoned and running (see below) for the rest of the process — competing for
//! CPU with everything that follows, XSLT30 included. A per-case budget is what turns "the
//! combined run's total wall time depends on how much of that contention there happens to be"
//! into "bounded, regardless".
//!
//! When a case overruns, its worker is *abandoned*: there is no safe way to cancel a running
//! computation in Rust, so the thread is left where it is, the runner drops its handles to it and
//! a fresh worker takes over. An abandoned worker keeps one case's working set alive for as long
//! as it goes on computing, which is why the budget is generous but finite and why the runner is
//! shared across a whole driver run rather than re-created per case.
//!
//! A driver *can* shorten that: an engine given the same instant as a cooperative deadline stops
//! its own computation at the next checkpoint it reaches, and the abandoned thread then falls out
//! of its loop and ends. [`CaseRunner::still_running`] counts the ones that have not, so whether a
//! given engine's deadline reaches a given case is measured rather than assumed — a stop that is
//! checked between instructions cannot interrupt one long call, and a case whose cost lies inside
//! a single retrieval or a single match will still be running when the next case starts.

use std::sync::{Arc, Weak, mpsc};
use std::time::Duration;

use crate::assertions::Verdict;

/// The environment variable that overrides [`CASE_BUDGET`], in whole seconds.
///
/// Triage only: shortening the budget turns "slow" into "timed out" and changes what the run
/// measures, so a reported conformance number must always come from a run that leaves it unset.
pub(crate) const CASE_BUDGET_VAR: &str = "OXIXML_QT3_CASE_BUDGET_SECS";

/// How long one test case may run before its driver gives up on it.
///
/// Long enough that no case which merely *is* slow (the suite has several that build large
/// intermediate sequences) is mistaken for one that does not terminate, short enough that a
/// handful of non-terminating cases cannot dominate a full-catalogue run.
pub(crate) const CASE_BUDGET: Duration = Duration::from_secs(30);

/// The prefix every timed-out case's failure message carries, so that a failure dump can be
/// filtered down to exactly the cases that ran out of time.
pub(crate) const TIMEOUT_MARKER: &str = "TIMEOUT";

/// The environment variable that turns on per-test-set progress lines on standard error, shared
/// by every driver that calls [`progress`] — the QT3 pair and `xslt30` alike, so one variable
/// gives visibility into a whole `all_suites()`/`oxixml-conformance-report` run.
///
/// A full-catalogue run takes minutes; without progress there is no way to tell a long run from a
/// stuck one, which is exactly the distinction this crate's own history has needed.
pub(crate) const PROGRESS_VAR: &str = "OXIXML_QT3_PROGRESS";

/// Writes one progress line to standard error, when [`PROGRESS_VAR`] is set. `driver` is the
/// full bracketed label a caller wants (e.g. `"qt3-xpath"`, `"xslt30"`), not just an engine name,
/// so each driver controls its own line's shape.
pub(crate) fn progress(driver: &str, test_set: &str) {
    if std::env::var_os(PROGRESS_VAR).is_some() {
        eprintln!("[{driver}] {test_set}");
    }
}

/// The stack a worker thread gets: the catalogue is full of deeply nested/recursive expressions
/// built specifically to stress a recursive-descent implementation, and a stack overflow aborts
/// the *process*, so this has to be sized for the worst case the catalogue contains rather than
/// for the average one. Matched to what the per-crate QT3 harness in `oxixml-xquery` uses, which
/// is known to survive the whole catalogue.
pub(crate) const WORKER_STACK: usize = 512 * 1024 * 1024;

/// The budget in force for this process.
pub(crate) fn case_budget() -> Duration {
    match std::env::var(CASE_BUDGET_VAR) {
        Ok(text) => match text.trim().parse::<u64>() {
            Ok(seconds) if seconds > 0 => Duration::from_secs(seconds),
            _ => CASE_BUDGET,
        },
        Err(_) => CASE_BUDGET,
    }
}

/// What running one QT3 case produced: `None` when the driver could not reproduce the case's
/// environment (a skip), otherwise the assertion checker's verdict.
pub(crate) type CaseOutcome = Option<Verdict>;

/// A unit of work handed to the worker thread.
type Job<T> = Box<dyn FnOnce() -> T + Send + 'static>;

/// Runs cases on a worker thread, one at a time, under [`case_budget`].
///
/// `T` is whatever a driver's own case function answers — [`CaseOutcome`] for the QT3 pair, a
/// richer outcome carrying a skip reason for [`crate::xslt30`]. The runner never inspects it: its
/// job is the wall-clock bound and the worker's stack, not the verdict.
pub(crate) struct CaseRunner<T> {
    jobs: mpsc::Sender<Job<T>>,
    results: mpsc::Receiver<T>,
    /// A handle on the *current* worker's life token, moved to [`Self::left_behind`] when the
    /// worker is abandoned. See [`Self::still_running`].
    alive: Weak<()>,
    budget: Duration,
    abandoned: usize,
    /// One life token per abandoned worker, still upgradable while that worker's thread lives.
    left_behind: Vec<Weak<()>>,
}

impl<T: Send + 'static> CaseRunner<T> {
    /// Starts a runner with a fresh worker, enforcing [`case_budget`].
    pub(crate) fn new() -> Self {
        Self::with_budget(case_budget())
    }

    /// Starts a runner with an explicit budget (the tests use this; drivers use [`Self::new`]).
    pub(crate) fn with_budget(budget: Duration) -> Self {
        let (jobs, results, alive) = Self::spawn_worker();
        Self {
            jobs,
            results,
            alive,
            budget,
            abandoned: 0,
            left_behind: Vec::new(),
        }
    }

    fn spawn_worker() -> (mpsc::Sender<Job<T>>, mpsc::Receiver<T>, Weak<()>) {
        let (jobs, incoming) = mpsc::channel::<Job<T>>();
        let (outgoing, results) = mpsc::channel::<T>();
        // The token the worker holds for exactly as long as its thread runs: it is moved into the
        // closure and dropped when the thread ends, however it ends, so a `Weak` to it answers
        // "is that worker still out there" without any cooperation from the worker itself.
        let alive = Arc::new(());
        let token = Weak::clone(&Arc::downgrade(&alive));
        // A failure to spawn leaves `incoming`/`outgoing` dropped, so the very first `run` sees a
        // closed channel and reports the case as timed out rather than panicking the driver.
        drop(
            std::thread::Builder::new()
                .stack_size(WORKER_STACK)
                .spawn(move || {
                    let _alive = alive;
                    while let Ok(job) = incoming.recv() {
                        if outgoing.send(job()).is_err() {
                            break;
                        }
                    }
                }),
        );
        (jobs, results, token)
    }

    /// Replaces the worker with a fresh one, keeping a handle on the old one's life token when it
    /// was left mid-computation.
    fn restart(&mut self, keep: bool) {
        if keep {
            self.left_behind.push(Weak::clone(&self.alive));
        }
        let (jobs, results, alive) = Self::spawn_worker();
        self.jobs = jobs;
        self.results = results;
        self.alive = alive;
    }

    /// The budget this runner enforces, for the message a caller writes on a timeout.
    pub(crate) const fn budget(&self) -> Duration {
        self.budget
    }

    /// How many cases this runner has abandoned mid-computation so far — over the whole run,
    /// whether or not the computation has since stopped.
    pub(crate) const fn abandoned(&self) -> usize {
        self.abandoned
    }

    /// How many abandoned workers are still running *now*.
    ///
    /// An abandoned thread is left where it was, and it keeps its [`WORKER_STACK`]-sized stack and
    /// its whole working set for as long as it goes on computing — which, for a driver that gives
    /// its engine no way to stop, is the rest of the process. That is what makes a later timeout a
    /// statement about the run rather than about the case, so it is the number [`timeout_message`]
    /// reports.
    ///
    /// It is also the measurement that says whether a *cooperative* stop works: a driver that hands
    /// the engine a deadline expects an abandoned worker to notice it and exit, and this counts the
    /// ones that did. The two counts diverging — many abandoned, few still running — is the whole
    /// of the claim.
    ///
    /// Read it **before** the [`Self::run`] whose message it will describe: afterwards it may
    /// already count the case being reported, which was not competing with itself.
    pub(crate) fn still_running(&self) -> usize {
        self.left_behind
            .iter()
            .filter(|worker| worker.strong_count() > 0)
            .count()
    }

    /// Runs `job` to completion, or answers `None` when it did not finish within the budget.
    ///
    /// On a timeout the worker is abandoned — it is still inside the case — and a fresh one takes
    /// over, so the next case is *adjudicated* rather than blocked behind a computation that may
    /// never end. Whether it is *unaffected* by it depends on the driver: an abandoned worker goes
    /// on computing and allocating until something stops it, and a driver that hands its engine no
    /// deadline has nothing that will. Measured on this workspace's own catalogue: `catalog-003`
    /// and `catalog-007` both finish inside the default budget when their test set is driven on its
    /// own, and both overrun it when the set is driven immediately after one that abandons a
    /// worker. A timeout is therefore only a statement about the *case* while
    /// [`Self::still_running`] is zero; afterwards it is a statement about the case and the run
    /// together, which is why [`timeout_message`] records the count beside it.
    pub(crate) fn run(&mut self, job: Job<T>) -> Option<T> {
        if self.jobs.send(job).is_err() {
            // The worker is gone (its thread failed to spawn, or it broke out of its loop); no
            // case was abandoned mid-computation, so the count must not move.
            self.restart(false);
            return None;
        }
        match self.results.recv_timeout(self.budget) {
            Ok(outcome) => Some(outcome),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.abandoned += 1;
                self.restart(true);
                None
            }
            // The worker *died* rather than overran: a job that panics unwinds out of the loop and
            // drops its end of the channel, which `recv_timeout` reports immediately. Nothing is
            // still computing, so — as on the send-failure path above — the count must not move.
            // Only `crate::xslt30` wraps its case in `catch_unwind`; a panicking QT3 case arrives
            // here, and counting it would make every later timeout claim a contention that is not
            // there.
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.restart(false);
                None
            }
        }
    }
}

/// The failure message recorded for a case that ran out of time.
///
/// The two counts are read from [`CaseRunner::still_running`] and [`CaseRunner::abandoned`] as
/// they stood when the case *started*, and they say different things. While nothing is still
/// running the case had the machine to itself and the timeout is a statement about the case; once
/// something is, the run is competing with its own leftovers and a case that merely comes close to
/// the budget can be pushed past it. The total is kept beside it because their difference is the
/// other measurement a dump should carry: how many earlier timeouts stopped computing once the
/// engine noticed the deadline it was given.
///
/// `detail` is the case's query/expression text, truncated so that one pathological case cannot
/// bloat a whole failure dump.
pub(crate) fn timeout_message(
    budget: Duration,
    still_running: usize,
    abandoned: usize,
    detail: &str,
) -> String {
    const MAX_DETAIL: usize = 4000;
    let flattened: String = detail.split_whitespace().collect::<Vec<_>>().join(" ");
    let shown = match flattened.char_indices().nth(MAX_DETAIL) {
        Some((end, _)) => format!("{}…", &flattened[..end]),
        None => flattened,
    };
    let contention = match (still_running, abandoned) {
        (_, 0) => String::new(),
        (0, 1) => ", after 1 earlier timeout, which is no longer running".to_owned(),
        (0, total) => format!(", after {total} earlier timeouts, none of them still running"),
        (1, total) => format!(", with 1 of {total} earlier timeouts still running"),
        (running, total) => format!(", with {running} of {total} earlier timeouts still running"),
    };
    format!(
        "{TIMEOUT_MARKER}: did not finish within {}s{contention} — {shown}",
        budget.as_secs()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_job_that_finishes_returns_its_outcome() {
        let mut runner = CaseRunner::<CaseOutcome>::new();
        let outcome = runner
            .run(Box::new(|| Some(Verdict::Pass)))
            .expect("the job finished");
        assert_eq!(outcome, Some(Verdict::Pass));
    }

    /// The runner carries whatever a driver's case function answers, not only [`CaseOutcome`]:
    /// `crate::xslt30` runs its own richer outcome type through the very same worker.
    #[test]
    fn a_job_of_any_outcome_type_travels_back_unchanged() {
        let mut runner = CaseRunner::<String>::new();
        let outcome = runner
            .run(Box::new(|| "carried".to_owned()))
            .expect("the job finished");
        assert_eq!(outcome, "carried");
    }

    #[test]
    fn a_job_that_overruns_is_abandoned_and_the_next_one_still_runs() {
        let mut runner = CaseRunner::<CaseOutcome>::with_budget(Duration::from_millis(200));
        let timed_out = runner.run(Box::new(|| {
            std::thread::sleep(Duration::from_secs(30));
            Some(Verdict::Pass)
        }));
        assert!(timed_out.is_none(), "the job must overrun its budget");
        let next = runner
            .run(Box::new(|| Some(Verdict::Fail("next".to_owned()))))
            .expect("a fresh worker took over");
        assert_eq!(next, Some(Verdict::Fail("next".to_owned())));
    }

    /// The count exists so that a dump says whether a timeout was measured against a quiet machine
    /// or against one the run is itself loading; a case that *finishes* must never move it, or
    /// every later timeout would read as contended when nothing was left behind.
    #[test]
    fn an_abandoned_worker_is_counted_and_a_finished_one_is_not() {
        let mut runner = CaseRunner::<CaseOutcome>::with_budget(Duration::from_millis(200));
        assert_eq!(runner.abandoned(), 0, "nothing has run yet");
        // The first case to overrun was competing with nothing, so the count a caller reads
        // *before* handing it over — which is the one that reaches `timeout_message` — is zero.
        let before_the_first_overrun = runner.abandoned();
        assert_eq!(before_the_first_overrun, 0);

        assert!(runner.run(Box::new(|| Some(Verdict::Pass))).is_some());
        assert_eq!(runner.abandoned(), 0, "a case that finished left nothing");

        for expected in 1..=2 {
            let timed_out = runner.run(Box::new(|| {
                std::thread::sleep(Duration::from_secs(30));
                Some(Verdict::Pass)
            }));
            assert!(timed_out.is_none(), "the job must overrun its budget");
            assert_eq!(runner.abandoned(), expected);
        }

        assert!(runner.run(Box::new(|| Some(Verdict::Pass))).is_some());
        assert_eq!(
            runner.abandoned(),
            2,
            "the two workers left behind are still out there; finishing a case does not reclaim one"
        );
    }

    /// The two counts answer different questions, and the difference between them is the whole
    /// value of handing an engine a deadline: `abandoned` never falls, while `still_running` falls
    /// as soon as a left-behind worker's own computation ends. A job that stops by itself is the
    /// stand-in here for an engine that honours a deadline.
    #[test]
    fn a_worker_that_stops_by_itself_is_no_longer_counted_as_running() {
        let mut runner = CaseRunner::<CaseOutcome>::with_budget(Duration::from_millis(200));
        assert_eq!(runner.still_running(), 0, "nothing has been left behind");

        // Overruns the budget, then ends of its own accord shortly afterwards — which is what an
        // engine reaching its deadline does.
        let timed_out = runner.run(Box::new(|| {
            std::thread::sleep(Duration::from_millis(600));
            Some(Verdict::Pass)
        }));
        assert!(timed_out.is_none(), "the job must overrun its budget");
        assert_eq!(runner.abandoned(), 1);

        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while runner.still_running() > 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(
            runner.still_running(),
            0,
            "the abandoned worker finished, so it is no longer competing for the machine"
        );
        assert_eq!(
            runner.abandoned(),
            1,
            "what it cost the run is still on the record"
        );

        // And one that does *not* stop is counted for as long as it lasts.
        let timed_out = runner.run(Box::new(|| {
            std::thread::sleep(Duration::from_secs(30));
            Some(Verdict::Pass)
        }));
        assert!(timed_out.is_none(), "the job must overrun its budget");
        assert_eq!(runner.abandoned(), 2);
        assert_eq!(runner.still_running(), 1);
    }

    /// A worker that dies leaves nothing running, so it is not an abandonment however much it
    /// looks like one from the receiving end — both are a `recv_timeout` error and only one of
    /// them means "a computation is still out there". The `thread … panicked` line this test
    /// prints is the deliberate cause, not a failure.
    #[test]
    fn a_worker_that_dies_is_replaced_without_being_counted() {
        let mut runner = CaseRunner::<CaseOutcome>::with_budget(Duration::from_secs(30));
        let died = runner.run(Box::new(|| panic!("the case under test blew up")));
        assert!(died.is_none(), "a worker that died answers no outcome");
        assert_eq!(
            runner.abandoned(),
            0,
            "nothing is still computing, so nothing was abandoned"
        );
        let next = runner
            .run(Box::new(|| Some(Verdict::Pass)))
            .expect("a fresh worker took over");
        assert_eq!(next, Some(Verdict::Pass));
    }

    #[test]
    fn a_timeout_message_is_marked_and_bounded() {
        let message = timeout_message(
            Duration::from_secs(30),
            0,
            0,
            "for $x in 1 to 10\n  return $x",
        );
        assert!(message.starts_with(TIMEOUT_MARKER));
        assert!(message.contains("30s"));
        assert!(
            message.contains("for $x in 1 to 10 return $x"),
            "the query text is flattened onto one line: {message}"
        );

        let long = "x".repeat(10_000);
        let bounded = timeout_message(Duration::from_secs(30), 0, 0, &long);
        assert!(bounded.len() < 4_200, "one case cannot bloat a dump");
        assert!(bounded.ends_with('…'));
    }

    /// A timeout measured while an earlier case's worker is still computing is not a statement
    /// about the case alone, and the dump has to say so — `catalog-003`/`-007` pass inside the
    /// default budget when their test set runs on its own and overrun it in a full catalogue run,
    /// which is a fact about the run, not about them.
    #[test]
    fn a_timeout_says_when_the_run_was_competing_with_itself() {
        let quiet = timeout_message(Duration::from_secs(30), 0, 0, "doc('x')");
        assert!(
            !quiet.contains("abandoned"),
            "an uncontended timeout claims nothing about contention: {quiet}"
        );

        let one = timeout_message(Duration::from_secs(30), 1, 3, "doc('x')");
        assert!(
            one.contains("did not finish within 30s, with 1 of 3 earlier timeouts still running")
        );
        assert!(
            one.ends_with("— doc('x')"),
            "the detail still closes it: {one}"
        );

        let many = timeout_message(Duration::from_secs(30), 4, 4, "doc('x')");
        assert!(many.contains("with 4 of 4 earlier timeouts still running"));

        // The other half of the same fact, and the one that says a cooperative deadline reached
        // the cases it was given: earlier timeouts that are no longer computing.
        let reclaimed = timeout_message(Duration::from_secs(30), 0, 1, "doc('x')");
        assert!(reclaimed.contains("after 1 earlier timeout, which is no longer running"));
        let several = timeout_message(Duration::from_secs(30), 0, 7, "doc('x')");
        assert!(several.contains("after 7 earlier timeouts, none of them still running"));

        for message in [&quiet, &one, &many, &reclaimed, &several] {
            assert!(
                message.starts_with(TIMEOUT_MARKER),
                "the marker a dump filters on is unmoved: {message}"
            );
        }
    }

    #[test]
    fn the_default_budget_is_used_when_the_override_is_absent_or_junk() {
        if std::env::var_os(CASE_BUDGET_VAR).is_some() {
            return; // A triage run set it; the default is not what is in force.
        }
        assert_eq!(case_budget(), CASE_BUDGET);
    }
}
