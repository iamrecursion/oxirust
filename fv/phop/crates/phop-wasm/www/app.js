// phop in-browser demo — discover → verify, entirely client-side. No network calls are made for
// any computation; the wasm module carries the discovery engine, the CAS, and the SMT solver.
import init, {
  capabilities,
  discover_and_verify,
  set_panic_hook,
} from "../pkg/phop_wasm.js";

const $ = (id) => document.getElementById(id);

// ---- Example datasets (generated locally) -----------------------------------------------------
function rows(n, f) {
  return Array.from({ length: n }, (_, i) => f(i));
}
const EXP_TARGET = '{"root":{"Eml":{"left":{"Var":0},"right":"One"}},"num_vars":1}';
const EXAMPLES = {
  "Exponential growth  y = eˣ": {
    method: "enumerate",
    maxDepth: 1,
    target: EXP_TARGET,
    csv:
      "x0,y\n" +
      rows(21, (i) => { const x = i * 0.15; return `${x.toFixed(4)},${Math.exp(x).toFixed(6)}`; }).join("\n"),
  },
  "Product law  y = x₀·x₁": {
    method: "rich",
    maxDepth: 2,
    target: null,
    csv:
      "x0,x1,y\n" +
      rows(30, (i) => { const a = 1 + 0.1 * i, b = 0.6 + 0.05 * i; return `${a.toFixed(4)},${b.toFixed(4)},${(a * b).toFixed(6)}`; }).join("\n"),
  },
  "Power law (Kepler)  T = a³ᐟ²": {
    method: "rich",
    maxDepth: 2,
    target: null,
    csv:
      "a,T\n" +
      rows(30, (i) => { const a = 1 + 0.08 * i; return `${a.toFixed(4)},${Math.pow(a, 1.5).toFixed(6)}`; }).join("\n"),
  },
};
let currentTarget = EXP_TARGET;

// ---- CSV → { x, y } ---------------------------------------------------------------------------
function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter((l) => l.trim().length);
  if (lines.length < 2) throw new Error("need a header row and at least one data row");
  const x = [], y = [];
  for (let r = 1; r < lines.length; r++) {
    const nums = lines[r].split(",").map((s) => Number(s.trim()));
    if (nums.some((v) => Number.isNaN(v))) throw new Error(`row ${r + 1} has a non-numeric value`);
    x.push(nums.slice(0, -1));
    y.push(nums[nums.length - 1]);
  }
  return { x, y };
}

// ---- Offline LaTeX → HTML prettifier (no KaTeX / no network) ----------------------------------
function prettyLatex(s) {
  if (!s) return "";
  let t = s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  for (let i = 0; i < 6; i++) t = t.replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "($1)/($2)");
  t = t.replace(/\\left/g, "").replace(/\\right/g, "");
  t = t.replace(/\\cdot/g, "·").replace(/\\times/g, "×");
  t = t.replace(/\\ln/g, "ln").replace(/\\exp/g, "exp").replace(/\\pi/g, "π").replace(/\\sqrt/g, "√");
  for (let i = 0; i < 6; i++) {
    t = t.replace(/\^\{([^{}]*)\}/g, "<sup>$1</sup>");
    t = t.replace(/_\{([^{}]*)\}/g, "<sub>$1</sub>");
  }
  t = t.replace(/\^(\w)/g, "<sup>$1</sup>").replace(/_(\w)/g, "<sub>$1</sub>");
  t = t.replace(/\\[,;:!]/g, " ").replace(/\\([a-zA-Z]+)/g, "$1");
  return t;
}

// ---- Verdict classification (color) -----------------------------------------------------------
function verdictClass(v) {
  const s = String(v).toLowerCase();
  if (/disproven|refuted|counterexample|hasroot/.test(s)) return "verdict-refuted";
  if (/proven|noroot|uniqueexists/.test(s)) return "verdict-proven";
  return "verdict-unknown";
}

function vrow(k, v, klass) {
  if (v === undefined || v === null) return "";
  const val = klass ? `<span class="${klass}">${v}</span>` : v;
  return `<div class="vrow"><div class="k">${k}</div><div class="v">${val}</div></div>`;
}

function card(s) {
  const sym = s.symbolic ? `<span class="badge sym">symbolic</span>` : "";
  const r2 = Number.isFinite(s.r2) ? s.r2.toFixed(6) : "n/a";
  let html = `<div class="card">
    <div class="top">
      <span class="badge ${s.source}">${s.source}</span>${sym}
    </div>
    <div class="law">${prettyLatex(s.latex)}</div>
    <div class="metrics">R² = <b>${r2}</b> &nbsp;·&nbsp; MSE = <b>${s.mse.toExponential(3)}</b> &nbsp;·&nbsp; complexity = <b>${s.complexity}</b></div>
    <div class="src">${s.pretty}</div>`;
  if (s.derivative) html += vrow("d/dx₀ (CAS)", prettyLatex(s.derivative));
  if (s.antiderivative) html += vrow("∫ · dx₀ (CAS)", prettyLatex(s.antiderivative));
  if (s.canonical_latex) html += vrow("canonical (e-graph)", prettyLatex(s.canonical_latex));
  if (s.certified_range) html += vrow("certified range", `f(x) ∈ [${s.certified_range[0].toPrecision(6)}, ${s.certified_range[1].toPrecision(6)}]`, "verdict-proven");
  if (s.certified_root) html += vrow("certified root (x₀)", s.certified_root, verdictClass(s.certified_root));
  if (s.proven_no_root) html += vrow("SMT: no root in box", s.proven_no_root, verdictClass(s.proven_no_root));
  if (s.equivalent_to_target) html += vrow("SMT: ≡ true law", s.equivalent_to_target, verdictClass(s.equivalent_to_target));
  html += `</div>`;
  return html;
}

async function run() {
  const btn = $("run");
  btn.disabled = true;
  $("results").innerHTML = "";
  $("status").textContent = "⏳ discovering & proving — in this tab, no server…";
  // Yield so the status paints before the (synchronous) wasm call blocks the thread.
  await new Promise((r) => setTimeout(r, 30));
  try {
    const { x, y } = parseCsv($("csv").value);
    const cfg = {
      method: $("method").value,
      max_epochs: 200,
      max_depth: 3,
      seed: 0,
      top_k: parseInt($("topk").value, 10),
      analyze: $("analyze").checked,
      canonical: $("canonical").checked,
      certify: $("certify").checked,
      prove_no_root: $("prove_no_root").checked,
    };
    if ($("equiv").checked && currentTarget) cfg.target_model = currentTarget;
    const t0 = performance.now();
    const out = JSON.parse(discover_and_verify(JSON.stringify({ x, y }), JSON.stringify(cfg)));
    const ms = Math.round(performance.now() - t0);
    if (out.error) throw new Error(out.error);
    $("status").textContent = `✓ done in ${ms} ms — entirely in your browser. ${out.solutions.length} solution(s).`;
    let body = out.solutions.map(card).join("");
    if (out.pi_groups) body = `<div class="metrics">Buckingham-π groups: ${JSON.stringify(out.pi_groups)}</div>` + body;
    $("results").innerHTML = body;
  } catch (e) {
    $("results").innerHTML = `<div class="err">${String(e.message || e)}</div>`;
    $("status").textContent = "";
  } finally {
    btn.disabled = false;
  }
}

function loadExample(name) {
  const ex = EXAMPLES[name];
  $("csv").value = ex.csv;
  $("method").value = ex.method;
  currentTarget = ex.target;
}

async function main() {
  await init();
  set_panic_hook();
  // Capability chips reflect which verification tiers were compiled into this wasm build.
  const caps = JSON.parse(capabilities());
  const labels = { analyze: "CAS analysis", certify: "certified range/root", canonical: "e-graph canonical", smt: "SMT proofs (OxiZ)" };
  $("caps").innerHTML = Object.entries(labels)
    .map(([k, label]) => `<span class="cap ${caps[k] ? "on" : "off"}">${label}</span>`)
    .join("");
  // Example buttons.
  $("examples").innerHTML = Object.keys(EXAMPLES).map((n) => `<button data-ex="${n}">${n}</button>`).join("");
  $("examples").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => loadExample(b.dataset.ex)));
  $("run").addEventListener("click", run);
  // Start on the exponential example so the page is one-click compelling.
  loadExample(Object.keys(EXAMPLES)[0]);
}

main();
