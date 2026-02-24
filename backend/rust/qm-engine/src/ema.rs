//! EMA derivative chain math — pure functions, no I/O.
//!
//! Ports Python event_engine.py lines 108–199 exactly.

/// Compute EMA blending factor for variable time intervals.
///
/// Uses the continuous-time formula: alpha = 1 - exp(-dt / tau).
/// Returns 0.0 for non-positive dt, 1.0 for dt/tau > 50.
#[inline]
pub fn ema_alpha(dt_s: f64, tau: f64) -> f64 {
    if dt_s <= 0.0 {
        return 0.0;
    }
    let ratio = dt_s / tau;
    if ratio > 50.0 {
        return 1.0;
    }
    1.0 - (-ratio).exp()
}

/// Update a three-level derivative chain from value changes.
///
/// Use for snapshot quantities (e.g. rest_depth) where the rate of change
/// of the signal itself is the correct input.
///
/// Returns (v_new, a_new, j_new). Guaranteed finite — falls back to prev on NaN/Inf.
#[inline]
pub fn update_derivative_chain(
    prev_value: f64,
    new_value: f64,
    dt_s: f64,
    v_prev: f64,
    a_prev: f64,
    j_prev: f64,
    tau_v: f64,
    tau_a: f64,
    tau_j: f64,
) -> (f64, f64, f64) {
    if dt_s <= 0.0 {
        return (v_prev, a_prev, j_prev);
    }

    let rate = (new_value - prev_value) / dt_s;

    let alpha_v = ema_alpha(dt_s, tau_v);
    let v_new = alpha_v * rate + (1.0 - alpha_v) * v_prev;

    let dv_rate = (v_new - v_prev) / dt_s;
    let alpha_a = ema_alpha(dt_s, tau_a);
    let a_new = alpha_a * dv_rate + (1.0 - alpha_a) * a_prev;

    let da_rate = (a_new - a_prev) / dt_s;
    let alpha_j = ema_alpha(dt_s, tau_j);
    let j_new = alpha_j * da_rate + (1.0 - alpha_j) * j_prev;

    if !v_new.is_finite() || !a_new.is_finite() || !j_new.is_finite() {
        return (v_prev, a_prev, j_prev);
    }

    (v_new, a_new, j_new)
}

/// Update a three-level derivative chain from a raw event delta.
///
/// Use for decayed accumulators (add_mass, pull_mass, fill_mass) to separate
/// passive decay from the derivative signal.
///
/// Returns (v_new, a_new, j_new). Guaranteed finite — falls back to prev on NaN/Inf.
#[inline]
pub fn update_derivative_chain_from_delta(
    delta: f64,
    dt_s: f64,
    v_prev: f64,
    a_prev: f64,
    j_prev: f64,
    tau_v: f64,
    tau_a: f64,
    tau_j: f64,
) -> (f64, f64, f64) {
    if dt_s <= 0.0 {
        return (v_prev, a_prev, j_prev);
    }

    let rate = delta / dt_s;

    let alpha_v = ema_alpha(dt_s, tau_v);
    let v_new = alpha_v * rate + (1.0 - alpha_v) * v_prev;

    let dv_rate = (v_new - v_prev) / dt_s;
    let alpha_a = ema_alpha(dt_s, tau_a);
    let a_new = alpha_a * dv_rate + (1.0 - alpha_a) * a_prev;

    let da_rate = (a_new - a_prev) / dt_s;
    let alpha_j = ema_alpha(dt_s, tau_j);
    let j_new = alpha_j * da_rate + (1.0 - alpha_j) * j_prev;

    if !v_new.is_finite() || !a_new.is_finite() || !j_new.is_finite() {
        return (v_prev, a_prev, j_prev);
    }

    (v_new, a_new, j_new)
}

/// Decay a derivative chain with zero-rate input for time advance.
///
/// This is the advance_time equivalent: new_v = (1-alpha)*v_prev, then
/// a/j follow from the resulting dv/da rates.
#[inline]
pub fn decay_derivative_chain(
    dt_s: f64,
    v_prev: f64,
    a_prev: f64,
    j_prev: f64,
    tau_v: f64,
    tau_a: f64,
    tau_j: f64,
) -> (f64, f64, f64) {
    if dt_s <= 0.0 {
        return (v_prev, a_prev, j_prev);
    }

    let alpha_v = ema_alpha(dt_s, tau_v);
    let alpha_a = ema_alpha(dt_s, tau_a);
    let alpha_j = ema_alpha(dt_s, tau_j);
    let keep_v = 1.0 - alpha_v;
    let keep_a = 1.0 - alpha_a;
    let keep_j = 1.0 - alpha_j;

    let v_new = keep_v * v_prev;
    let dv_rate = (v_new - v_prev) / dt_s;
    let a_new = alpha_a * dv_rate + keep_a * a_prev;

    let da_rate = (a_new - a_prev) / dt_s;
    let j_new = alpha_j * da_rate + keep_j * j_prev;

    let v_out = if v_new.is_finite() { v_new } else { 0.0 };
    let a_out = if a_new.is_finite() { a_new } else { 0.0 };
    let j_out = if j_new.is_finite() { j_new } else { 0.0 };

    (v_out, a_out, j_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_alpha_zero_dt() {
        assert_eq!(ema_alpha(0.0, 2.0), 0.0);
        assert_eq!(ema_alpha(-1.0, 2.0), 0.0);
    }

    #[test]
    fn test_ema_alpha_large_ratio() {
        assert_eq!(ema_alpha(200.0, 2.0), 1.0); // ratio=100 > 50
    }

    #[test]
    fn test_ema_alpha_typical() {
        let alpha = ema_alpha(2.0, 2.0); // ratio=1.0, alpha = 1 - 1/e
        let expected = 1.0 - (-1.0_f64).exp();
        assert!((alpha - expected).abs() < 1e-15);
    }

    #[test]
    fn test_derivative_chain_decay_to_zero() {
        // With zero rate-of-change, chain should decay toward zero
        let (v, a, j) =
            update_derivative_chain(5.0, 5.0, 10.0, 1.0, 0.5, 0.2, 2.0, 5.0, 10.0);
        assert!(v.abs() < 1.0);
        assert!(a.abs() < 0.5);
        assert!(j.abs() < 0.2);
    }

    #[test]
    fn test_derivative_chain_zero_dt_passthrough() {
        let (v, a, j) =
            update_derivative_chain(1.0, 2.0, 0.0, 0.5, 0.3, 0.1, 2.0, 5.0, 10.0);
        assert_eq!(v, 0.5);
        assert_eq!(a, 0.3);
        assert_eq!(j, 0.1);
    }

    #[test]
    fn test_from_delta_symmetry() {
        // update_derivative_chain_from_delta with delta=new-prev*dt should match update_derivative_chain
        let (v1, a1, j1) =
            update_derivative_chain(0.0, 3.0, 2.0, 0.0, 0.0, 0.0, 2.0, 5.0, 10.0);
        // rate = (3-0)/2 = 1.5, delta = 1.5 * 2 = 3.0
        let (v2, a2, j2) =
            update_derivative_chain_from_delta(3.0, 2.0, 0.0, 0.0, 0.0, 2.0, 5.0, 10.0);
        assert!((v1 - v2).abs() < 1e-14);
        assert!((a1 - a2).abs() < 1e-14);
        assert!((j1 - j2).abs() < 1e-14);
    }
}
