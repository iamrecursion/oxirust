//! `calc` / `calc_ut`: the `swe_calc` / `swe_calc_ut` analogues.
//!
//! Output convention (mirroring the SE documentation): a `[f64; 6]` of
//! `[lon, lat, dist, dlon/day, dlat/day, ddist/day]` — degrees and AU
//! (radians with [`crate::flags::SEFLG_RADIANS`]) — or Cartesian
//! `[x, y, z, vx, vy, vz]` in AU and AU/day with
//! [`crate::flags::SEFLG_XYZ`]. The three speed slots are `0.0` unless
//! [`crate::flags::SEFLG_SPEED`] (or `SEFLG_SPEED3`) is set.
//!
//! # Speeds
//!
//! Wherever possible the speeds are the apparent pipeline's own
//! central-difference rates
//! ([`oxiephemeris_bodies::apparent::SphericalRates`], step ±0.01 d
//! geocentric / ±0.001 d topocentric). For the output shapes the
//! pipeline does not provide rates for — Cartesian
//! ([`crate::flags::SEFLG_XYZ`]), the fixed-J2000 frame (post-rotated
//! here, see below), and the node/apogee bodies — this module applies
//! the **same central-difference scheme and step sizes** at its own
//! level, so every speed this crate returns shares one documented
//! method.
//!
//! # The `SEFLG_J2000` frame
//!
//! [`oxiephemeris_bodies::apparent::Frame`] has no "mean equator and
//! equinox of J2000.0" variant (its `MeanOfDate` is of-date). This crate
//! reproduces the J2000 frame exactly as the frame-bias rotation at
//! `t = 0`: the position is computed in the ICRS frame and rotated by
//! [`precession_bias_matrix`]`(0.0)` (equatorial) or
//! [`mean_of_date_to_ecliptic`]`(0.0) ·` [`precession_bias_matrix`]`(0.0)`
//! (ecliptic) — at `t = 0` the precession part is the identity, so these
//! are precisely the GCRS → J2000-mean rotations.
//!
//! # The `SEFLG_ICRS` modifier
//!
//! Per [`crate::flags::SEFLG_ICRS`] (SE's behavior, verified against SE
//! output): with `SEFLG_J2000` it selects the raw ICRS frame (no
//! rotation at all; the ecliptic variant applies only
//! [`mean_of_date_to_ecliptic`]`(0.0)`); alone it keeps the of-date
//! frame but strips the frame-bias factor from the rotation chain —
//! implemented as the epoch-dependent post rotation `F(t) · B₀ᵀ` (see
//! `PostRot::OfDateNoBias`).
//!
//! # Node/apogee bodies (`SE_MEAN_NODE` … `SE_OSCU_APOG`)
//!
//! Longitudes come from [`oxiephemeris_astro::nodes`] (mean-element
//! polynomials / osculating elements, mean ecliptic and equinox of
//! date); the default (nutation-on) output adds the IAU 2000A nutation
//! in longitude `Δψ` so the longitude refers to the **true** equinox of
//! date, exactly like the planetary default. With
//! [`crate::flags::SEFLG_NONUT`] the raw mean-equinox value is returned.
//! Latitude and distance (and their speeds) are **always `0.0`** for
//! these four bodies — see the crate docs' "no radius/latitude
//! convention" difference note. The aberration/deflection/`TRUEPOS`
//! bits are meaningless for them (mean elements and osculating elements
//! have no light path) and are accepted and ignored; the frame bits
//! `SEFLG_EQUATORIAL`/`SEFLG_XYZ`/`SEFLG_J2000`/`SEFLG_ICRS` are
//! rejected as conflicts (these are inherently ecliptic-of-date
//! quantities here).

use oxiephemeris_astro::ayanamsha::{ayanamsha_rad, Ayanamsha};
use oxiephemeris_astro::nodes::{mean_apogee, mean_node, true_node_of_date};
use oxiephemeris_bodies::apparent::{
    apparent, apparent_topocentric, BodyPosition, Frame, Options, Target,
};
use oxiephemeris_bodies::frames::{
    gcrs_to_true_of_date_with, mean_of_date_to_ecliptic, nutation_iau2000a, precession_bias_matrix,
    true_of_date_to_ecliptic_with, NutationModel,
};
use oxiephemeris_bodies::math::{Mat3, Vec3};
use oxiephemeris_core::angle::{normalize_0_two_pi, normalize_pm_pi, RAD2DEG};
use oxiephemeris_core::time::{JulianDate, J2000_JD};
use oxiephemeris_core::CoreError;

use super::{jd_tt_from_ut, Context, DAYS_PER_CENTURY};
use crate::bodies::{require_geocentric, resolve, NodeKind, ResolvedBody};
use crate::flags::{decode, DecodedFlags, FrameChoice};
use crate::CompatError;

/// Central-difference step, days — the same ±0.01 d the apparent
/// pipeline documents for [`Options::with_speed`].
const SPEED_STEP_DAYS: f64 = 0.01;

/// Central-difference step for **topocentric** speeds, days — matching
/// the pipeline's own topocentric step: the diurnal-parallax oscillation
/// (`ω = 2π/day`) makes even the ±0.01 d truncation error ≈ 4×10⁻³ °/day
/// for the Moon; ±0.001 d brings it to ≈ 4×10⁻⁵ °/day.
const TOPO_SPEED_STEP_DAYS: f64 = 0.001;

/// The extra rotation some flag combinations apply after the pipeline
/// (see the module docs on `SEFLG_J2000` and `SEFLG_ICRS`).
#[derive(Clone, Copy)]
enum PostRot {
    /// A fixed rotation: the `SEFLG_J2000` reproduction and the
    /// ecliptic-J2000 / ecliptic-ICRS ties (all evaluated at `t = 0`).
    Const(Mat3),
    /// Of-date output with the frame bias omitted (`SEFLG_ICRS` without
    /// `SEFLG_J2000` — see [`crate::flags::SEFLG_ICRS`]): `F(t) · B₀ᵀ`,
    /// where `F(t)` is the requested of-date rotation chain and `B₀`
    /// the constant frame bias. Epoch-dependent, so it is re-evaluated
    /// at every epoch a speed difference touches.
    OfDateNoBias { true_equinox: bool, ecliptic: bool },
}

impl PostRot {
    /// The rotation matrix at the TT epoch `jd_tt`.
    fn matrix(self, jd_tt: JulianDate) -> Mat3 {
        match self {
            Self::Const(m) => m,
            Self::OfDateNoBias {
                true_equinox,
                ecliptic,
            } => {
                let t = centuries_tt(jd_tt);
                let model = NutationModel::default();
                let f = match (true_equinox, ecliptic) {
                    (true, false) => gcrs_to_true_of_date_with(t, model),
                    (true, true) => true_of_date_to_ecliptic_with(t, model)
                        .mul(&gcrs_to_true_of_date_with(t, model)),
                    (false, false) => precession_bias_matrix(t),
                    (false, true) => mean_of_date_to_ecliptic(t).mul(&precession_bias_matrix(t)),
                };
                f.mul(&precession_bias_matrix(0.0).transpose())
            }
        }
    }
}

/// Julian centuries TT since J2000.0 of a two-part TT epoch.
fn centuries_tt(jd_tt: JulianDate) -> f64 {
    ((jd_tt.hi - J2000_JD) + jd_tt.lo) / DAYS_PER_CENTURY
}

/// Frame routing for [`Context::calc`]-family direct bodies: the
/// pipeline [`Frame`] plus the extra [`PostRot`] rotation applied after
/// it (see the module docs on `SEFLG_J2000` and `SEFLG_ICRS`).
fn route_frame(flags: &DecodedFlags) -> (Frame, Option<PostRot>) {
    match (flags.frame, flags.equatorial) {
        (FrameChoice::OfDate { true_equinox }, equatorial) if flags.icrs_no_bias => (
            Frame::Icrs,
            Some(PostRot::OfDateNoBias {
                true_equinox,
                ecliptic: !equatorial,
            }),
        ),
        (FrameChoice::OfDate { true_equinox: true }, true) => (Frame::TrueOfDate, None),
        (FrameChoice::OfDate { true_equinox: true }, false) => (Frame::EclipticTrueOfDate, None),
        (
            FrameChoice::OfDate {
                true_equinox: false,
            },
            true,
        ) => (Frame::MeanOfDate, None),
        (
            FrameChoice::OfDate {
                true_equinox: false,
            },
            false,
        ) => (Frame::EclipticMeanOfDate, None),
        (FrameChoice::Icrs, true) => (Frame::Icrs, None),
        (FrameChoice::Icrs, false) => (
            Frame::Icrs,
            Some(PostRot::Const(mean_of_date_to_ecliptic(0.0))),
        ),
        (FrameChoice::J2000, true) => (
            Frame::Icrs,
            Some(PostRot::Const(precession_bias_matrix(0.0))),
        ),
        (FrameChoice::J2000, false) => (
            Frame::Icrs,
            Some(PostRot::Const(
                mean_of_date_to_ecliptic(0.0).mul(&precession_bias_matrix(0.0)),
            )),
        ),
    }
}

/// Spherical `(lon, lat, r)` of a Cartesian vector, `lon` in `[0, 2π)`.
fn spherical_of(v: Vec3) -> (f64, f64, f64) {
    let rho_xy = libm::sqrt(v[0] * v[0] + v[1] * v[1]);
    let r = libm::sqrt(rho_xy * rho_xy + v[2] * v[2]);
    (
        normalize_0_two_pi(libm::atan2(v[1], v[0])),
        libm::atan2(v[2], rho_xy),
        r,
    )
}

impl Context<'_> {
    /// Body position at TT (SE: "ET") epoch `jd_et`, mirroring
    /// `swe_calc(tjd_et, ipl, iflag, xx, serr)`. See the module docs for
    /// the output convention and [`crate::flags`] for the flag set.
    ///
    /// # Errors
    ///
    /// [`CompatError::Core`] for a non-finite `jd_et`;
    /// [`CompatError::UnsupportedBody`] /
    /// [`CompatError::UnsupportedFlags`] /
    /// [`CompatError::UnsupportedFlagName`] /
    /// [`CompatError::ConflictingFlags`] / [`CompatError::TopoNotSet`]
    /// from body/flag validation; [`CompatError::NoEphemeris`] when the
    /// body needs a DE file and none is attached (only `SE_MEAN_NODE` /
    /// `SE_MEAN_APOG` work without one); [`CompatError::Bodies`] /
    /// [`CompatError::Node`] from the underlying computation (epoch
    /// outside the DE span, pre-1972 topocentric epochs, …).
    pub fn calc(&self, jd_et: f64, ipl: i32, iflag: u32) -> Result<[f64; 6], CompatError> {
        if !jd_et.is_finite() {
            return Err(CompatError::Core(CoreError::InvalidInput));
        }
        self.calc_at(JulianDate::from_f64(jd_et), ipl, iflag)
    }

    /// Body position at UT epoch `jd_ut`, mirroring
    /// `swe_calc_ut(tjd_ut, ipl, iflag, xx, serr)`: identical to
    /// [`Context::calc`] after the ΔT-based UT → TT conversion described
    /// in the crate docs ("Different time-scale models").
    ///
    /// # Errors
    ///
    /// As [`Context::calc`], plus [`CompatError::Core`] if `jd_ut`
    /// cannot be converted (non-finite / out of calendar range).
    #[allow(clippy::similar_names)] // jd_ut / jd_tt are the domain-standard names
    pub fn calc_ut(&self, jd_ut: f64, ipl: i32, iflag: u32) -> Result<[f64; 6], CompatError> {
        let jd_tt = jd_tt_from_ut(jd_ut)?;
        self.calc_at(jd_tt, ipl, iflag)
    }

    /// Shared implementation of [`Context::calc`] / [`Context::calc_ut`].
    fn calc_at(&self, jd_tt: JulianDate, ipl: i32, iflag: u32) -> Result<[f64; 6], CompatError> {
        let flags = decode(iflag, self.topo.is_some())?;
        match resolve(ipl)? {
            ResolvedBody::Direct(target) => self.calc_direct(jd_tt, target, &flags),
            ResolvedBody::Node(kind) => self.calc_node(jd_tt, kind, &flags),
        }
    }

    /// A direct ephemeris body (Sun..Pluto, Earth) through the apparent
    /// pipeline.
    fn calc_direct(
        &self,
        jd_tt: JulianDate,
        target: Target,
        flags: &DecodedFlags,
    ) -> Result<[f64; 6], CompatError> {
        let (frame, post) = route_frame(flags);

        // The pipeline's own rates serve only the plain spherical path;
        // XYZ and post-rotated output difference at this level instead.
        let pipeline_speed = flags.with_speed && post.is_none() && !flags.xyz;
        let opts = Options::new(
            flags.center,
            frame,
            flags.aberration,
            flags.deflection,
            !flags.truepos,
            pipeline_speed,
            NutationModel::default(),
        );
        let de = self.ephemeris()?;
        let place = |jd: JulianDate| -> Result<BodyPosition, CompatError> {
            if flags.topocentric {
                let topo = self.topo.as_ref().ok_or(CompatError::TopoNotSet)?;
                Ok(apparent_topocentric(de, target, jd, opts, topo)?)
            } else {
                Ok(apparent(de, target, jd, opts)?)
            }
        };
        // This level's own central-difference step (XYZ and post-rotated
        // shapes): matches the pipeline's, including the finer
        // topocentric step (diurnal-parallax truncation, see the const).
        let step = if flags.topocentric {
            TOPO_SPEED_STEP_DAYS
        } else {
            SPEED_STEP_DAYS
        };
        let pos = place(jd_tt)?;

        if flags.xyz {
            let rotate = |p: &BodyPosition| -> Vec3 {
                post.map_or(p.position_au, |pr| pr.matrix(p.jd_tt).apply(p.position_au))
            };
            let v = rotate(&pos);
            let mut out = [v[0], v[1], v[2], 0.0, 0.0, 0.0];
            if flags.with_speed {
                let plus = place(jd_tt.add_days(step))?;
                let minus = place(jd_tt.add_days(-step))?;
                let (vp, vm) = (rotate(&plus), rotate(&minus));
                for i in 0..3 {
                    out[i + 3] = (vp[i] - vm[i]) / (2.0 * step);
                }
            }
            return Ok(out);
        }

        let (mut lon, lat, r) = match post {
            Some(pr) => spherical_of(pr.matrix(pos.jd_tt).apply(pos.position_au)),
            None => (pos.lon_rad, pos.lat_rad, pos.r_au),
        };
        let (mut dlon, mut dlat, mut dr) = (0.0, 0.0, 0.0);
        if flags.with_speed {
            if let Some(rates) = pos.rates {
                (dlon, dlat, dr) = (
                    rates.lon_rad_per_day,
                    rates.lat_rad_per_day,
                    rates.r_au_per_day,
                );
            } else {
                // Post-rotated frame: same central differences, applied
                // to the rotated spherical coordinates (the post
                // rotation is re-evaluated at each epoch, so the
                // epoch-dependent SEFLG_ICRS chain differentiates
                // correctly too).
                let plus = place(jd_tt.add_days(step))?;
                let minus = place(jd_tt.add_days(-step))?;
                let rot = |p: &BodyPosition| -> (f64, f64, f64) {
                    match post {
                        Some(pr) => spherical_of(pr.matrix(p.jd_tt).apply(p.position_au)),
                        None => (p.lon_rad, p.lat_rad, p.r_au),
                    }
                };
                let (lp, bp, rp) = rot(&plus);
                let (lm, bm, rm) = rot(&minus);
                dlon = normalize_pm_pi(lp - lm) / (2.0 * step);
                dlat = (bp - bm) / (2.0 * step);
                dr = (rp - rm) / (2.0 * step);
            }
        }

        if flags.sidereal {
            let t = centuries_tt(jd_tt);
            let kind = self.sidereal_mode().to_ayanamsha();
            let true_equinox = matches!(flags.frame, FrameChoice::OfDate { true_equinox: true });
            lon = normalize_0_two_pi(lon - sidereal_offset_rad(kind, t, true_equinox));
            if flags.with_speed {
                dlon -= sidereal_offset_rate_rad_per_day(kind, t, true_equinox);
            }
        }

        let scale = if flags.radians { 1.0 } else { RAD2DEG };
        Ok([lon * scale, lat * scale, r, dlon * scale, dlat * scale, dr])
    }

    /// The four lunar node/apogee bodies (see the module docs).
    fn calc_node(
        &self,
        jd_tt: JulianDate,
        kind: NodeKind,
        flags: &DecodedFlags,
    ) -> Result<[f64; 6], CompatError> {
        require_geocentric(flags.center)?;
        if flags.topocentric {
            return Err(CompatError::ConflictingFlags(
                "the node/apogee bodies are geocentric quantities: SEFLG_TOPOCTR does not apply",
            ));
        }
        if flags.equatorial || flags.xyz {
            return Err(CompatError::ConflictingFlags(
                "the node/apogee bodies are ecliptic-of-date longitudes here: \
                 SEFLG_EQUATORIAL / SEFLG_XYZ are not defined for them",
            ));
        }
        // `icrs_no_bias` (SEFLG_ICRS alone) keeps FrameChoice::OfDate,
        // so it needs its own rejection to honor the module docs'
        // promise that SEFLG_ICRS is a conflict for these bodies.
        if flags.icrs_no_bias {
            return Err(CompatError::ConflictingFlags(
                "the node/apogee bodies are of-date quantities here: \
                 SEFLG_J2000 / SEFLG_ICRS are not defined for them",
            ));
        }
        let true_equinox = match flags.frame {
            FrameChoice::OfDate { true_equinox } => true_equinox,
            FrameChoice::J2000 | FrameChoice::Icrs => {
                return Err(CompatError::ConflictingFlags(
                    "the node/apogee bodies are of-date quantities here: \
                     SEFLG_J2000 / SEFLG_ICRS are not defined for them",
                ));
            }
        };

        // Ecliptic-of-date longitude of the requested point; the mean
        // bodies need no ephemeris, the osculating ones evaluate the DE
        // Moon state.
        let lon_at = |jd: JulianDate| -> Result<f64, CompatError> {
            let t = centuries_tt(jd);
            let mean_equinox_lon = match kind {
                NodeKind::MeanNode => mean_node(t),
                NodeKind::MeanApogee => mean_apogee(t),
                NodeKind::TrueNode => true_node_of_date(self.ephemeris()?, jd)?.node_lon_rad,
                NodeKind::OscuApogee => true_node_of_date(self.ephemeris()?, jd)?.apogee_lon_rad,
            };
            if true_equinox {
                Ok(normalize_0_two_pi(
                    mean_equinox_lon + nutation_iau2000a(t).dpsi_rad,
                ))
            } else {
                Ok(mean_equinox_lon)
            }
        };

        let mut lon = lon_at(jd_tt)?;
        let mut dlon = 0.0;
        if flags.with_speed {
            let lp = lon_at(jd_tt.add_days(SPEED_STEP_DAYS))?;
            let lm = lon_at(jd_tt.add_days(-SPEED_STEP_DAYS))?;
            dlon = normalize_pm_pi(lp - lm) / (2.0 * SPEED_STEP_DAYS);
        }

        if flags.sidereal {
            let t = centuries_tt(jd_tt);
            let ay_kind = self.sidereal_mode().to_ayanamsha();
            lon = normalize_0_two_pi(lon - sidereal_offset_rad(ay_kind, t, true_equinox));
            if flags.with_speed {
                dlon -= sidereal_offset_rate_rad_per_day(ay_kind, t, true_equinox);
            }
        }

        let scale = if flags.radians { 1.0 } else { RAD2DEG };
        Ok([lon * scale, 0.0, 0.0, dlon * scale, 0.0, 0.0])
    }
}

/// The sidereal offset subtracted from tropical longitudes, in radians:
/// the ayanamsha referred to the **true equinox of date**, i.e. the
/// mean-equinox [`ayanamsha_rad`] plus the nutation in longitude `Δψ`
/// (when the longitudes themselves are true-equinox ones). Subtracting
/// `Δψ` along with the ayanamsha makes the sidereal longitudes free of
/// the equinox's nutation wobble — a star-anchored zodiac should not
/// nutate — and matches Swiss Ephemeris' output convention (verified
/// against SE 2.10.03 *output*: SE's sidereal longitudes differ from
/// `tropical − swe_get_ayanamsa` by exactly `−Δψ`, while
/// `swe_get_ayanamsa` itself returns the mean-equinox value, as
/// [`crate::Context::get_ayanamsa`] does). With `SEFLG_NONUT` the
/// longitudes are mean-equinox already, so only the mean ayanamsha is
/// subtracted.
fn sidereal_offset_rad(kind: Ayanamsha, t_centuries: f64, true_equinox: bool) -> f64 {
    let dpsi = if true_equinox {
        nutation_iau2000a(t_centuries).dpsi_rad
    } else {
        0.0
    };
    ayanamsha_rad(kind, t_centuries) + dpsi
}

/// Drift rate of [`sidereal_offset_rad`] in rad/day: the same
/// [`SPEED_STEP_DAYS`] central difference as every other speed in this
/// module.
fn sidereal_offset_rate_rad_per_day(kind: Ayanamsha, t_centuries: f64, true_equinox: bool) -> f64 {
    let dt_c = SPEED_STEP_DAYS / DAYS_PER_CENTURY;
    let plus = sidereal_offset_rad(kind, t_centuries + dt_c, true_equinox);
    let minus = sidereal_offset_rad(kind, t_centuries - dt_c, true_equinox);
    normalize_pm_pi(plus - minus) / (2.0 * SPEED_STEP_DAYS)
}

#[cfg(test)]
mod tests {
    use super::{sidereal_offset_rad, sidereal_offset_rate_rad_per_day, spherical_of};
    use oxiephemeris_astro::ayanamsha::Ayanamsha;
    use oxiephemeris_core::angle::RAD2DEG;

    #[test]
    fn spherical_of_recovers_axes() {
        let (lon, lat, r) = spherical_of([1.0, 0.0, 0.0]);
        assert!(lon.abs() < 1e-15 && lat.abs() < 1e-15 && (r - 1.0).abs() < 1e-15);
        let (lon, lat, r) = spherical_of([0.0, 0.0, 2.0]);
        assert!((lat - core::f64::consts::FRAC_PI_2).abs() < 1e-15);
        assert!((r - 2.0).abs() < 1e-15);
        let _ = lon; // longitude is arbitrary on the pole axis
    }

    #[test]
    fn sidereal_offset_rate_is_about_50_arcsec_per_year() {
        // General precession ~ 50.29 arcsec/yr = 0.1377 arcsec/day; the
        // true-equinox offset adds the nutation drift (≤ ~0.016 arcsec/day,
        // 2π·17.2″/6798 d at the 18.6-yr term's steepest).
        let expected_deg_day = 50.29 / 365.25 / 3600.0;
        let mean_rate =
            sidereal_offset_rate_rad_per_day(Ayanamsha::J2000Zero, 0.0, false) * RAD2DEG;
        assert!(
            (mean_rate - expected_deg_day).abs() < 1e-3 * expected_deg_day,
            "mean rate = {mean_rate} deg/day"
        );
        let true_rate = sidereal_offset_rate_rad_per_day(Ayanamsha::J2000Zero, 0.0, true) * RAD2DEG;
        assert!(
            (true_rate - expected_deg_day).abs() < 0.02 / 3600.0,
            "true-equinox rate = {true_rate} deg/day"
        );
    }

    #[test]
    fn sidereal_offset_true_equinox_differs_by_dpsi() {
        // At J2000 the nutation in longitude is ≈ −13.9″; the true-equinox
        // offset must differ from the mean one by exactly that.
        let mean = sidereal_offset_rad(Ayanamsha::J2000Zero, 0.0, false);
        let true_eq = sidereal_offset_rad(Ayanamsha::J2000Zero, 0.0, true);
        let dpsi_arcsec = (true_eq - mean) * RAD2DEG * 3600.0;
        assert!(
            (dpsi_arcsec + 13.9).abs() < 0.2,
            "dpsi at J2000 = {dpsi_arcsec} arcsec"
        );
    }
}
