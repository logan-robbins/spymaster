//! AbsoluteTickEngine — absolute-tick event-driven vacuum/pressure engine.
//!
//! Ports Python event_engine.py: AbsoluteTickEngine class.
//! Exposes to Python via PyO3 as `vp_engine.AbsoluteTickEngine`.

use std::collections::HashMap;

use numpy::PyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};

use crate::book::{
    ACTION_ADD, ACTION_CANCEL, ACTION_CLEAR, ACTION_FILL, ACTION_MODIFY, ACTION_TRADE, SIDE_BID,
    OrderBook,
};
use crate::ema::{
    decay_derivative_chain, update_derivative_chain, update_derivative_chain_from_delta,
};

const PRICE_SCALE: f64 = 1e-9;
const F_SNAPSHOT: u8 = 32;
const F_LAST: u8 = 128;

#[derive(Debug, Serialize, Deserialize)]
struct BookStateV3 {
    _v: u32,
    orders: HashMap<u64, (String, i64, i64)>, // order_id -> (side_char, price_int, qty)
    best_bid: i64,
    best_ask: i64,
    anchor_tick_idx: i64,
    event_counter: i64,
    prev_ts_ns: i64,
    book_valid: bool,
    snapshot_in_progress: bool,
}

#[pyclass(module = "vp_engine")]
pub struct AbsoluteTickEngine {
    // Config
    n_ticks: usize,
    tick_int: i64,
    tau_velocity: f64,
    tau_acceleration: f64,
    tau_jerk: f64,
    tau_rest_decay: f64,
    c1_v_add: f64,
    c2_v_rest_pos: f64,
    c3_a_add: f64,
    c4_v_pull: f64,
    c5_v_fill: f64,
    c6_v_rest_neg: f64,
    c7_a_pull: f64,
    auto_anchor_from_bbo: bool,

    // Physics arrays (25 f64 + 2 i64)
    add_mass: Vec<f64>,
    pull_mass: Vec<f64>,
    fill_mass: Vec<f64>,
    rest_depth: Vec<f64>,
    bid_depth: Vec<f64>,
    ask_depth: Vec<f64>,
    v_add: Vec<f64>,
    v_pull: Vec<f64>,
    v_fill: Vec<f64>,
    v_rest_depth: Vec<f64>,
    v_bid_depth: Vec<f64>,
    v_ask_depth: Vec<f64>,
    a_add: Vec<f64>,
    a_pull: Vec<f64>,
    a_fill: Vec<f64>,
    a_rest_depth: Vec<f64>,
    a_bid_depth: Vec<f64>,
    a_ask_depth: Vec<f64>,
    j_add: Vec<f64>,
    j_pull: Vec<f64>,
    j_fill: Vec<f64>,
    j_rest_depth: Vec<f64>,
    j_bid_depth: Vec<f64>,
    j_ask_depth: Vec<f64>,
    pressure_variant: Vec<f64>,
    vacuum_variant: Vec<f64>,
    last_ts_ns: Vec<i64>,
    last_event_id: Vec<i64>,

    // Order book
    book: OrderBook,

    // Lifecycle
    event_counter: i64,
    prev_ts_ns: i64,
    book_valid: bool,
    snapshot_in_progress: bool,
}

fn parse_action(s: &str) -> Option<u8> {
    s.as_bytes().first().copied()
}

#[pymethods]
impl AbsoluteTickEngine {
    #[new]
    #[pyo3(signature = (
        n_ticks = 500,
        tick_int = 250_000_000,
        bucket_size_dollars = 0.25,
        tau_velocity = 2.0,
        tau_acceleration = 5.0,
        tau_jerk = 10.0,
        tau_rest_decay = 30.0,
        c1_v_add = 1.0,
        c2_v_rest_pos = 0.5,
        c3_a_add = 0.3,
        c4_v_pull = 1.0,
        c5_v_fill = 1.5,
        c6_v_rest_neg = 0.5,
        c7_a_pull = 0.3,
        anchor_tick_idx = None,
        auto_anchor_from_bbo = true,
        fail_on_out_of_range = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_ticks: usize,
        tick_int: i64,
        bucket_size_dollars: f64,
        tau_velocity: f64,
        tau_acceleration: f64,
        tau_jerk: f64,
        tau_rest_decay: f64,
        c1_v_add: f64,
        c2_v_rest_pos: f64,
        c3_a_add: f64,
        c4_v_pull: f64,
        c5_v_fill: f64,
        c6_v_rest_neg: f64,
        c7_a_pull: f64,
        anchor_tick_idx: Option<i64>,
        auto_anchor_from_bbo: bool,
        fail_on_out_of_range: bool,
    ) -> PyResult<Self> {
        let _ = bucket_size_dollars; // accepted but unused; Python API compat only
        if n_ticks < 3 {
            return Err(PyRuntimeError::new_err(format!(
                "n_ticks must be >= 3, got {n_ticks}"
            )));
        }
        if tick_int <= 0 {
            return Err(PyRuntimeError::new_err(format!(
                "tick_int must be > 0, got {tick_int}"
            )));
        }

        let anchor = match anchor_tick_idx {
            None => -1i64,
            Some(a) if a < 0 => {
                return Err(PyRuntimeError::new_err(format!(
                    "anchor_tick_idx must be >= 0 when provided, got {a}"
                )));
            }
            Some(a) => a,
        };

        let mut book = OrderBook::new(n_ticks, tick_int, fail_on_out_of_range);
        book.anchor_tick_idx = anchor;

        let mut engine = AbsoluteTickEngine {
            n_ticks,
            tick_int,
            tau_velocity,
            tau_acceleration,
            tau_jerk,
            tau_rest_decay,
            c1_v_add,
            c2_v_rest_pos,
            c3_a_add,
            c4_v_pull,
            c5_v_fill,
            c6_v_rest_neg,
            c7_a_pull,
            auto_anchor_from_bbo,
            add_mass: vec![0.0; n_ticks],
            pull_mass: vec![0.0; n_ticks],
            fill_mass: vec![0.0; n_ticks],
            rest_depth: vec![0.0; n_ticks],
            bid_depth: vec![0.0; n_ticks],
            ask_depth: vec![0.0; n_ticks],
            v_add: vec![0.0; n_ticks],
            v_pull: vec![0.0; n_ticks],
            v_fill: vec![0.0; n_ticks],
            v_rest_depth: vec![0.0; n_ticks],
            v_bid_depth: vec![0.0; n_ticks],
            v_ask_depth: vec![0.0; n_ticks],
            a_add: vec![0.0; n_ticks],
            a_pull: vec![0.0; n_ticks],
            a_fill: vec![0.0; n_ticks],
            a_rest_depth: vec![0.0; n_ticks],
            a_bid_depth: vec![0.0; n_ticks],
            a_ask_depth: vec![0.0; n_ticks],
            j_add: vec![0.0; n_ticks],
            j_pull: vec![0.0; n_ticks],
            j_fill: vec![0.0; n_ticks],
            j_rest_depth: vec![0.0; n_ticks],
            j_bid_depth: vec![0.0; n_ticks],
            j_ask_depth: vec![0.0; n_ticks],
            pressure_variant: vec![0.0; n_ticks],
            vacuum_variant: vec![0.0; n_ticks],
            last_ts_ns: vec![0i64; n_ticks],
            last_event_id: vec![0i64; n_ticks],
            book,
            event_counter: 0,
            prev_ts_ns: 0,
            book_valid: false,
            snapshot_in_progress: false,
        };

        if anchor >= 0 {
            engine.book.rebuild_depth_from_orders();
        }

        Ok(engine)
    }

    /// Process one MBO event: book update + derivative chain + force.
    /// This is the hot path — called once per event.
    fn update(
        &mut self,
        ts_ns: i64,
        action: &str,
        side: &str,
        price_int: i64,
        size: i64,
        order_id: u64,
        flags: u8,
    ) -> PyResult<()> {
        self.event_counter += 1;
        let event_id = self.event_counter;
        let action_byte = parse_action(action).unwrap_or(0);
        let side_byte = parse_action(side).unwrap_or(0);
        let is_snapshot = (flags & F_SNAPSHOT) != 0;
        let is_last = (flags & F_LAST) != 0;

        // Snapshot lifecycle
        let was_snapshot_in_progress = self.snapshot_in_progress;
        if action_byte == ACTION_CLEAR && is_snapshot {
            self.snapshot_in_progress = true;
        }
        if self.snapshot_in_progress && is_last {
            self.snapshot_in_progress = false;
            self.book_valid = true;
        }
        if !self.book_valid && !self.snapshot_in_progress {
            self.book_valid = true;
        }

        self.prev_ts_ns = ts_ns;

        // Apply event to order book; collect (idx, add_delta, pull_delta, fill_delta)
        let mut touched: Vec<(i64, f64, f64, f64)> = Vec::new();

        match action_byte {
            ACTION_CLEAR => {
                self.book.clear();
                self.book_valid = false;
            }
            ACTION_TRADE => {}
            ACTION_ADD => {
                if let Ok((idx, qty_added)) = self.book.add_order(side_byte, price_int, size, order_id) {
                    if idx >= 0 && qty_added > 0 {
                        touched.push((idx, qty_added as f64, 0.0, 0.0));
                    }
                }
            }
            ACTION_CANCEL => {
                if let Some((idx, qty)) = self.book.cancel_order(order_id) {
                    if idx >= 0 && qty > 0 {
                        touched.push((idx, 0.0, qty as f64, 0.0));
                    }
                }
            }
            ACTION_MODIFY => {
                match self.book.modify_order(order_id, side_byte, price_int, size) {
                    Ok(Some(((old_idx, old_qty), (new_idx, new_qty)))) => {
                        if old_idx != new_idx {
                            if old_idx >= 0 && old_qty > 0 {
                                touched.push((old_idx, 0.0, old_qty as f64, 0.0));
                            }
                            if new_idx >= 0 && new_qty > 0 {
                                touched.push((new_idx, new_qty as f64, 0.0, 0.0));
                            }
                        } else {
                            let size_diff = new_qty - old_qty;
                            if new_idx >= 0 {
                                if size_diff > 0 {
                                    touched.push((new_idx, size_diff as f64, 0.0, 0.0));
                                } else if size_diff < 0 {
                                    touched.push((new_idx, 0.0, (-size_diff) as f64, 0.0));
                                } else {
                                    touched.push((new_idx, 0.0, 0.0, 0.0));
                                }
                            }
                        }
                    }
                    Ok(None) => {
                        // Order didn't exist pre-modify — just added it
                    }
                    Err(_) => {}
                }
            }
            ACTION_FILL => {
                if let Some((idx, eff_fill)) = self.book.fill_order(order_id, size) {
                    if idx >= 0 && eff_fill > 0 {
                        touched.push((idx, 0.0, 0.0, eff_fill as f64));
                    }
                }
            }
            _ => {}
        }

        // Snapshot completion
        let snapshot_completed = was_snapshot_in_progress && !self.snapshot_in_progress;

        // Set anchor from first valid BBO
        let anchor_was_unset = self.book.anchor_tick_idx < 0;
        self.try_set_anchor();
        let anchor_just_set = anchor_was_unset && self.book.anchor_tick_idx >= 0;

        // Sync rest_depth after snapshot or anchor establishment
        if (snapshot_completed || anchor_just_set) && self.book.anchor_tick_idx >= 0 {
            self.sync_rest_depth_from_book_inner();
        }

        // Update mechanics + derivatives at touched ticks
        for (idx, add_delta, pull_delta, fill_delta) in touched {
            if idx < 0 || idx as usize >= self.n_ticks {
                continue;
            }
            let i = idx as usize;

            let last_ts = self.last_ts_ns[i];
            let dt_s = if last_ts > 0 && ts_ns > last_ts {
                (ts_ns - last_ts) as f64 / 1e9
            } else {
                0.0
            };

            let prev_bid = self.bid_depth[i];
            let prev_ask = self.ask_depth[i];

            self.bid_depth[i] = self.book.depth_bid[i] as f64;
            self.ask_depth[i] = self.book.depth_ask[i] as f64;
            self.rest_depth[i] = self.bid_depth[i] + self.ask_depth[i];

            if dt_s > 0.0 {
                let decay = (-dt_s / self.tau_rest_decay).exp();
                self.add_mass[i] = self.add_mass[i] * decay + add_delta;
                self.pull_mass[i] = self.pull_mass[i] * decay + pull_delta;
                self.fill_mass[i] = self.fill_mass[i] * decay + fill_delta;
            } else {
                self.add_mass[i] += add_delta;
                self.pull_mass[i] += pull_delta;
                self.fill_mass[i] += fill_delta;
            }

            if dt_s > 0.0 {
                let (v, a, j) = update_derivative_chain_from_delta(
                    add_delta,
                    dt_s,
                    self.v_add[i],
                    self.a_add[i],
                    self.j_add[i],
                    self.tau_velocity,
                    self.tau_acceleration,
                    self.tau_jerk,
                );
                self.v_add[i] = v;
                self.a_add[i] = a;
                self.j_add[i] = j;

                let (v, a, j) = update_derivative_chain_from_delta(
                    pull_delta,
                    dt_s,
                    self.v_pull[i],
                    self.a_pull[i],
                    self.j_pull[i],
                    self.tau_velocity,
                    self.tau_acceleration,
                    self.tau_jerk,
                );
                self.v_pull[i] = v;
                self.a_pull[i] = a;
                self.j_pull[i] = j;

                let (v, a, j) = update_derivative_chain_from_delta(
                    fill_delta,
                    dt_s,
                    self.v_fill[i],
                    self.a_fill[i],
                    self.j_fill[i],
                    self.tau_velocity,
                    self.tau_acceleration,
                    self.tau_jerk,
                );
                self.v_fill[i] = v;
                self.a_fill[i] = a;
                self.j_fill[i] = j;

                let (v, a, j) = update_derivative_chain(
                    prev_bid,
                    self.bid_depth[i],
                    dt_s,
                    self.v_bid_depth[i],
                    self.a_bid_depth[i],
                    self.j_bid_depth[i],
                    self.tau_velocity,
                    self.tau_acceleration,
                    self.tau_jerk,
                );
                self.v_bid_depth[i] = v;
                self.a_bid_depth[i] = a;
                self.j_bid_depth[i] = j;

                let (v, a, j) = update_derivative_chain(
                    prev_ask,
                    self.ask_depth[i],
                    dt_s,
                    self.v_ask_depth[i],
                    self.a_ask_depth[i],
                    self.j_ask_depth[i],
                    self.tau_velocity,
                    self.tau_acceleration,
                    self.tau_jerk,
                );
                self.v_ask_depth[i] = v;
                self.a_ask_depth[i] = a;
                self.j_ask_depth[i] = j;
            }

            // Invariant: rest_depth derivatives = bid + ask derivatives
            self.v_rest_depth[i] = self.v_bid_depth[i] + self.v_ask_depth[i];
            self.a_rest_depth[i] = self.a_bid_depth[i] + self.a_ask_depth[i];
            self.j_rest_depth[i] = self.j_bid_depth[i] + self.j_ask_depth[i];

            // Pressure / vacuum
            self.pressure_variant[i] = self.c1_v_add * self.v_add[i]
                + self.c2_v_rest_pos * self.v_rest_depth[i].max(0.0)
                + self.c3_a_add * self.a_add[i].max(0.0);

            self.vacuum_variant[i] = self.c4_v_pull * self.v_pull[i]
                + self.c5_v_fill * self.v_fill[i]
                + self.c6_v_rest_neg * (-self.v_rest_depth[i]).max(0.0)
                + self.c7_a_pull * self.a_pull[i].max(0.0);

            self.last_event_id[i] = event_id;
            self.last_ts_ns[i] = ts_ns;
        }

        Ok(())
    }

    /// Vectorized passive time advance for all active ticks.
    ///
    /// Applies zero-delta decay to mass/derivative state for ticks that have
    /// prior state (last_ts_ns > 0) when ts_ns is ahead of their last update.
    fn advance_time(&mut self, ts_ns: i64) -> PyResult<()> {
        if ts_ns <= 0 {
            return Ok(());
        }

        // Collect active tick indices
        let active: Vec<usize> = (0..self.n_ticks)
            .filter(|&i| self.last_ts_ns[i] > 0 && self.last_ts_ns[i] < ts_ns)
            .collect();

        if active.is_empty() {
            return Ok(());
        }

        for &i in &active {
            let last_ts = self.last_ts_ns[i];
            let dt_s = (ts_ns - last_ts) as f64 / 1e9;

            // Decay mechanics mass with no new event delta
            let decay = (-dt_s / self.tau_rest_decay).exp();
            self.add_mass[i] *= decay;
            self.pull_mass[i] *= decay;
            self.fill_mass[i] *= decay;

            // Decay derivative chains with zero event-rate input
            let (v, a, j) = decay_derivative_chain(
                dt_s,
                self.v_add[i],
                self.a_add[i],
                self.j_add[i],
                self.tau_velocity,
                self.tau_acceleration,
                self.tau_jerk,
            );
            self.v_add[i] = v;
            self.a_add[i] = a;
            self.j_add[i] = j;

            let (v, a, j) = decay_derivative_chain(
                dt_s,
                self.v_pull[i],
                self.a_pull[i],
                self.j_pull[i],
                self.tau_velocity,
                self.tau_acceleration,
                self.tau_jerk,
            );
            self.v_pull[i] = v;
            self.a_pull[i] = a;
            self.j_pull[i] = j;

            let (v, a, j) = decay_derivative_chain(
                dt_s,
                self.v_fill[i],
                self.a_fill[i],
                self.j_fill[i],
                self.tau_velocity,
                self.tau_acceleration,
                self.tau_jerk,
            );
            self.v_fill[i] = v;
            self.a_fill[i] = a;
            self.j_fill[i] = j;

            let (v, a, j) = decay_derivative_chain(
                dt_s,
                self.v_bid_depth[i],
                self.a_bid_depth[i],
                self.j_bid_depth[i],
                self.tau_velocity,
                self.tau_acceleration,
                self.tau_jerk,
            );
            self.v_bid_depth[i] = v;
            self.a_bid_depth[i] = a;
            self.j_bid_depth[i] = j;

            let (v, a, j) = decay_derivative_chain(
                dt_s,
                self.v_ask_depth[i],
                self.a_ask_depth[i],
                self.j_ask_depth[i],
                self.tau_velocity,
                self.tau_acceleration,
                self.tau_jerk,
            );
            self.v_ask_depth[i] = v;
            self.a_ask_depth[i] = a;
            self.j_ask_depth[i] = j;

            // Invariant: rest = bid + ask
            self.v_rest_depth[i] = self.v_bid_depth[i] + self.v_ask_depth[i];
            self.a_rest_depth[i] = self.a_bid_depth[i] + self.a_ask_depth[i];
            self.j_rest_depth[i] = self.j_bid_depth[i] + self.j_ask_depth[i];

            // Recompute force fields
            self.pressure_variant[i] = self.c1_v_add * self.v_add[i]
                + self.c2_v_rest_pos * self.v_rest_depth[i].max(0.0)
                + self.c3_a_add * self.a_add[i].max(0.0);

            self.vacuum_variant[i] = self.c4_v_pull * self.v_pull[i]
                + self.c5_v_fill * self.v_fill[i]
                + self.c6_v_rest_neg * (-self.v_rest_depth[i]).max(0.0)
                + self.c7_a_pull * self.a_pull[i].max(0.0);

            // last_event_id intentionally unchanged — no event touched these ticks
            self.last_ts_ns[i] = ts_ns;
        }

        Ok(())
    }

    /// Lightweight book-only event processing for pre-warmup fast-forward.
    ///
    /// Updates order book and BBO without computing mechanics, derivatives, or force.
    /// 10–50x faster than update().
    fn apply_book_event(
        &mut self,
        ts_ns: i64,
        action: &str,
        side: &str,
        price_int: i64,
        size: i64,
        order_id: u64,
        flags: u8,
    ) -> PyResult<()> {
        self.event_counter += 1;
        let action_byte = parse_action(action).unwrap_or(0);
        let side_byte = parse_action(side).unwrap_or(0);
        let is_snapshot = (flags & F_SNAPSHOT) != 0;
        let is_last = (flags & F_LAST) != 0;

        if action_byte == ACTION_CLEAR && is_snapshot {
            self.snapshot_in_progress = true;
        }
        if self.snapshot_in_progress && is_last {
            self.snapshot_in_progress = false;
            self.book_valid = true;
        }
        if !self.book_valid && !self.snapshot_in_progress {
            self.book_valid = true;
        }

        self.prev_ts_ns = ts_ns;

        match action_byte {
            ACTION_CLEAR => {
                self.book.clear();
                self.book_valid = false;
            }
            ACTION_ADD => {
                let _ = self.book.add_order(side_byte, price_int, size, order_id);
            }
            ACTION_CANCEL => {
                self.book.cancel_order(order_id);
            }
            ACTION_MODIFY => {
                let _ = self.book.modify_order(order_id, side_byte, price_int, size);
            }
            ACTION_FILL => {
                self.book.fill_order(order_id, size);
            }
            _ => {}
        }

        self.try_set_anchor();
        Ok(())
    }

    /// Reset anchor to current BBO midpoint.
    ///
    /// Call after importing book state and BEFORE sync_rest_depth_from_book().
    fn reanchor_to_bbo(&mut self) -> bool {
        if self.book.best_bid <= 0 || self.book.best_ask <= 0 {
            return false;
        }
        let new_tick = self.mid_tick_from_bbo();
        self.book.anchor_tick_idx = new_tick;
        self.book.rebuild_depth_from_orders();
        true
    }

    /// Re-anchor to raw order-book BBO and reset per-tick state arrays.
    fn soft_reanchor_to_order_book_bbo(&mut self) -> bool {
        let (best_bid, best_ask) = self.book.raw_bbo_from_orders();
        if best_bid <= 0 || best_ask <= 0 {
            return false;
        }
        let new_tick = ((best_bid + best_ask) as f64 / (2.0 * self.tick_int as f64) + 0.5).floor()
            as i64;
        self.book.best_bid_idx = -1;
        self.book.best_ask_idx = -1;
        self.book.best_bid = best_bid;
        self.book.best_ask = best_ask;
        self.book.anchor_tick_idx = new_tick;
        self.book.rebuild_depth_from_orders();
        self.reset_per_tick_state();
        self.sync_rest_depth_from_book_inner();
        true
    }

    /// Synchronize all rest_depth values from current order book.
    fn sync_rest_depth_from_book(&mut self) -> PyResult<()> {
        if self.book.anchor_tick_idx < 0 {
            return Ok(());
        }
        self.sync_rest_depth_from_book_inner();
        Ok(())
    }

    /// Export book state to msgpack bytes for caching.
    fn export_book_state(&self) -> PyResult<Vec<u8>> {
        let orders: HashMap<u64, (String, i64, i64)> = self
            .book
            .orders
            .iter()
            .map(|(&oid, e)| {
                let side_str = if e.side == SIDE_BID {
                    "B".to_string()
                } else {
                    "A".to_string()
                };
                (oid, (side_str, e.price_int, e.qty))
            })
            .collect();

        let state = BookStateV3 {
            _v: 3,
            orders,
            best_bid: self.book.best_bid,
            best_ask: self.book.best_ask,
            anchor_tick_idx: self.book.anchor_tick_idx,
            event_counter: self.event_counter,
            prev_ts_ns: self.prev_ts_ns,
            book_valid: self.book_valid,
            snapshot_in_progress: self.snapshot_in_progress,
        };

        rmp_serde::to_vec(&state)
            .map_err(|e| PyRuntimeError::new_err(format!("msgpack serialize error: {e}")))
    }

    /// Restore book state from bytes (msgpack or Python pickle).
    fn import_book_state(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<()> {
        // Detect format: pickle starts with \x80\x02-\x05
        let is_pickle =
            data.len() >= 2 && data[0] == 0x80 && data[1] >= 2 && data[1] <= 5;

        if is_pickle {
            self.import_pickle_state(py, data)?;
        } else {
            self.import_msgpack_state(data)?;
        }

        // Rebuild depth under current anchor
        if self.book.anchor_tick_idx >= 0 {
            self.book.rebuild_depth_from_orders();
        } else {
            self.book.recompute_provisional_bbo_from_orders();
        }

        // Reset grid arrays — warmup will populate
        self.reset_per_tick_state();

        Ok(())
    }

    /// Return full grid state as a dict of numpy arrays.
    ///
    /// Copies all 25+ arrays — ~1.8MB per call at 8192 ticks, ~2µs on M4.
    fn grid_snapshot_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);

        macro_rules! put_f64 {
            ($field:ident) => {
                dict.set_item(
                    stringify!($field),
                    PyArray1::from_slice_bound(py, &self.$field),
                )?;
            };
        }
        macro_rules! put_i64 {
            ($field:ident) => {
                dict.set_item(
                    stringify!($field),
                    PyArray1::from_slice_bound(py, &self.$field),
                )?;
            };
        }

        put_f64!(add_mass);
        put_f64!(pull_mass);
        put_f64!(fill_mass);
        put_f64!(rest_depth);
        put_f64!(bid_depth);
        put_f64!(ask_depth);
        put_f64!(v_add);
        put_f64!(v_pull);
        put_f64!(v_fill);
        put_f64!(v_rest_depth);
        put_f64!(v_bid_depth);
        put_f64!(v_ask_depth);
        put_f64!(a_add);
        put_f64!(a_pull);
        put_f64!(a_fill);
        put_f64!(a_rest_depth);
        put_f64!(a_bid_depth);
        put_f64!(a_ask_depth);
        put_f64!(j_add);
        put_f64!(j_pull);
        put_f64!(j_fill);
        put_f64!(j_rest_depth);
        put_f64!(j_bid_depth);
        put_f64!(j_ask_depth);
        put_f64!(pressure_variant);
        put_f64!(vacuum_variant);
        put_i64!(last_event_id);

        Ok(dict)
    }

    /// Return diagnostic book health metrics.
    fn book_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let bid_levels = self.book.depth_bid.iter().filter(|&&x| x > 0).count();
        let ask_levels = self.book.depth_ask.iter().filter(|&&x| x > 0).count();
        let total_bid_qty: i64 = self.book.depth_bid.iter().sum();
        let total_ask_qty: i64 = self.book.depth_ask.iter().sum();

        let dict = PyDict::new_bound(py);
        dict.set_item("order_count", self.book.orders.len())?;
        dict.set_item("bid_levels", bid_levels)?;
        dict.set_item("ask_levels", ask_levels)?;
        dict.set_item("total_bid_qty", total_bid_qty)?;
        dict.set_item("total_ask_qty", total_ask_qty)?;
        dict.set_item("best_bid", self.book.best_bid)?;
        dict.set_item("best_ask", self.book.best_ask)?;
        dict.set_item("anchor_tick_idx", self.book.anchor_tick_idx)?;
        dict.set_item("event_count", self.event_counter)?;
        dict.set_item("book_valid", self.book_valid)?;
        Ok(dict)
    }

    /// Map a spot price_int to an array index (for serve-time windowing).
    fn spot_to_idx(&self, spot_price_int: i64) -> Option<usize> {
        self.book.price_to_idx(spot_price_int)
    }

    // --- Read-only properties ---

    #[getter]
    fn n_ticks(&self) -> usize {
        self.n_ticks
    }

    #[getter]
    fn tick_int(&self) -> i64 {
        self.tick_int
    }

    #[getter]
    fn anchor_tick_idx(&self) -> i64 {
        self.book.anchor_tick_idx
    }

    #[getter]
    fn event_count(&self) -> i64 {
        self.event_counter
    }

    #[getter]
    fn order_count(&self) -> usize {
        self.book.orders.len()
    }

    #[getter]
    fn book_valid(&self) -> bool {
        self.book_valid
    }

    #[getter]
    fn best_bid_price_int(&self) -> i64 {
        self.book.best_bid
    }

    #[getter]
    fn best_ask_price_int(&self) -> i64 {
        self.book.best_ask
    }

    #[getter]
    fn mid_price(&self) -> f64 {
        if self.book.best_bid > 0 && self.book.best_ask > 0 {
            (self.book.best_bid + self.book.best_ask) as f64 * 0.5 * PRICE_SCALE
        } else {
            0.0
        }
    }

    #[getter]
    fn spot_ref_price_int(&self) -> i64 {
        if self.book.best_bid > 0 && self.book.best_ask > 0 {
            let raw_mid = (self.book.best_bid + self.book.best_ask) as f64 / 2.0;
            ((raw_mid / self.tick_int as f64 + 0.5).floor() as i64) * self.tick_int
        } else {
            0
        }
    }
}

// Private helpers (not exposed to Python)
impl AbsoluteTickEngine {
    fn mid_tick_from_bbo(&self) -> i64 {
        ((self.book.best_bid + self.book.best_ask) as f64 / (2.0 * self.tick_int as f64) + 0.5)
            .floor() as i64
    }

    fn try_set_anchor(&mut self) {
        if self.book.anchor_tick_idx >= 0 || !self.auto_anchor_from_bbo {
            return;
        }
        if self.book.best_bid > 0 && self.book.best_ask > 0 {
            let mid_tick = self.mid_tick_from_bbo();
            self.book.anchor_tick_idx = mid_tick;
            self.book.rebuild_depth_from_orders();
        }
    }

    fn reset_per_tick_state(&mut self) {
        self.add_mass.iter_mut().for_each(|x| *x = 0.0);
        self.pull_mass.iter_mut().for_each(|x| *x = 0.0);
        self.fill_mass.iter_mut().for_each(|x| *x = 0.0);
        self.rest_depth.iter_mut().for_each(|x| *x = 0.0);
        self.bid_depth.iter_mut().for_each(|x| *x = 0.0);
        self.ask_depth.iter_mut().for_each(|x| *x = 0.0);
        self.v_add.iter_mut().for_each(|x| *x = 0.0);
        self.v_pull.iter_mut().for_each(|x| *x = 0.0);
        self.v_fill.iter_mut().for_each(|x| *x = 0.0);
        self.v_rest_depth.iter_mut().for_each(|x| *x = 0.0);
        self.v_bid_depth.iter_mut().for_each(|x| *x = 0.0);
        self.v_ask_depth.iter_mut().for_each(|x| *x = 0.0);
        self.a_add.iter_mut().for_each(|x| *x = 0.0);
        self.a_pull.iter_mut().for_each(|x| *x = 0.0);
        self.a_fill.iter_mut().for_each(|x| *x = 0.0);
        self.a_rest_depth.iter_mut().for_each(|x| *x = 0.0);
        self.a_bid_depth.iter_mut().for_each(|x| *x = 0.0);
        self.a_ask_depth.iter_mut().for_each(|x| *x = 0.0);
        self.j_add.iter_mut().for_each(|x| *x = 0.0);
        self.j_pull.iter_mut().for_each(|x| *x = 0.0);
        self.j_fill.iter_mut().for_each(|x| *x = 0.0);
        self.j_rest_depth.iter_mut().for_each(|x| *x = 0.0);
        self.j_bid_depth.iter_mut().for_each(|x| *x = 0.0);
        self.j_ask_depth.iter_mut().for_each(|x| *x = 0.0);
        self.pressure_variant.iter_mut().for_each(|x| *x = 0.0);
        self.vacuum_variant.iter_mut().for_each(|x| *x = 0.0);
        self.last_ts_ns.iter_mut().for_each(|x| *x = 0);
        self.last_event_id.iter_mut().for_each(|x| *x = 0);
    }

    fn sync_rest_depth_from_book_inner(&mut self) {
        for i in 0..self.n_ticks {
            self.bid_depth[i] = self.book.depth_bid[i] as f64;
            self.ask_depth[i] = self.book.depth_ask[i] as f64;
            self.rest_depth[i] = self.bid_depth[i] + self.ask_depth[i];
        }
    }

    fn import_msgpack_state(&mut self, data: &[u8]) -> PyResult<()> {
        let state: BookStateV3 = rmp_serde::from_slice(data).map_err(|e| {
            PyRuntimeError::new_err(format!("msgpack deserialize error: {e}"))
        })?;

        if state._v != 3 {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported book state version {}, expected 3. Delete the cache and retry.",
                state._v
            )));
        }

        self.book.orders.clear();
        for (oid, (side_str, price_int, qty)) in state.orders {
            let side = if side_str == "B" { b'B' } else { b'A' };
            self.book.orders.insert(
                oid,
                crate::book::OrderEntry {
                    side,
                    price_int,
                    qty,
                    idx: -1,
                },
            );
        }

        self.book.anchor_tick_idx = state.anchor_tick_idx;
        self.event_counter = state.event_counter;
        self.prev_ts_ns = state.prev_ts_ns;
        self.book_valid = state.book_valid;
        self.snapshot_in_progress = state.snapshot_in_progress;
        self.book.best_bid = 0;
        self.book.best_ask = 0;
        self.book.best_bid_idx = -1;
        self.book.best_ask_idx = -1;

        Ok(())
    }

    fn import_pickle_state(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<()> {
        let pickle = py.import_bound("pickle")?;
        let state = pickle.call_method1("loads", (data,))?;
        let state_dict = state.downcast::<PyDict>()?;

        let version: u32 = state_dict
            .get_item("_v")?
            .ok_or_else(|| PyRuntimeError::new_err("Missing '_v' in book state"))?
            .extract()?;

        if version != 2 && version != 3 {
            return Err(PyRuntimeError::new_err(format!(
                "Unsupported pickle book state version {version}. Delete the cache and retry."
            )));
        }

        // Parse orders: {order_id: (side_str, price_int, qty)}
        self.book.orders.clear();
        let orders_obj = state_dict
            .get_item("orders")?
            .ok_or_else(|| PyRuntimeError::new_err("Missing 'orders' in book state"))?;
        let orders_dict = orders_obj.downcast::<PyDict>()?;
        for (k, v) in orders_dict.iter() {
            let oid: u64 = k.extract()?;
            let (side_str, price_int, qty): (String, i64, i64) = v.extract()?;
            let side = if side_str == "B" { b'B' } else { b'A' };
            self.book.orders.insert(
                oid,
                crate::book::OrderEntry {
                    side,
                    price_int,
                    qty,
                    idx: -1,
                },
            );
        }

        self.book.anchor_tick_idx = state_dict
            .get_item("anchor_tick_idx")?
            .ok_or_else(|| PyRuntimeError::new_err("Missing 'anchor_tick_idx'"))?
            .extract()?;
        self.event_counter = state_dict
            .get_item("event_counter")?
            .ok_or_else(|| PyRuntimeError::new_err("Missing 'event_counter'"))?
            .extract()?;
        self.prev_ts_ns = state_dict
            .get_item("prev_ts_ns")?
            .ok_or_else(|| PyRuntimeError::new_err("Missing 'prev_ts_ns'"))?
            .extract()?;
        self.book_valid = state_dict
            .get_item("book_valid")?
            .ok_or_else(|| PyRuntimeError::new_err("Missing 'book_valid'"))?
            .extract()?;
        self.snapshot_in_progress = state_dict
            .get_item("snapshot_in_progress")?
            .ok_or_else(|| PyRuntimeError::new_err("Missing 'snapshot_in_progress'"))?
            .extract()?;
        self.book.best_bid = 0;
        self.book.best_ask = 0;
        self.book.best_bid_idx = -1;
        self.book.best_ask_idx = -1;

        Ok(())
    }
}
