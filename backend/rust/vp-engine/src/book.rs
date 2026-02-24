//! Order book reconstruction with incremental BBO tracking.
//!
//! Ports Python event_engine.py book operations exactly.
//! Supports absolute-tick indexing with O(1) BBO maintenance.

use std::collections::HashMap;

pub const SIDE_BID: u8 = b'B';
pub const SIDE_ASK: u8 = b'A';

pub const ACTION_ADD: u8 = b'A';
pub const ACTION_CANCEL: u8 = b'C';
pub const ACTION_MODIFY: u8 = b'M';
pub const ACTION_CLEAR: u8 = b'R';
pub const ACTION_TRADE: u8 = b'T';
pub const ACTION_FILL: u8 = b'F';

#[derive(Clone, Debug)]
pub struct OrderEntry {
    pub side: u8,
    pub price_int: i64,
    pub qty: i64,
    pub idx: i64, // -1 if out-of-range or anchor not yet set
}

pub struct OrderBook {
    pub orders: HashMap<u64, OrderEntry>,
    pub depth_bid: Vec<i64>,
    pub depth_ask: Vec<i64>,
    pub best_bid_idx: i64,
    pub best_ask_idx: i64,
    pub best_bid: i64,
    pub best_ask: i64,
    pub anchor_tick_idx: i64,
    pub n_ticks: usize,
    pub tick_int: i64,
    pub fail_on_out_of_range: bool,
}

impl OrderBook {
    pub fn new(n_ticks: usize, tick_int: i64, fail_on_out_of_range: bool) -> Self {
        OrderBook {
            orders: HashMap::new(),
            depth_bid: vec![0i64; n_ticks],
            depth_ask: vec![0i64; n_ticks],
            best_bid_idx: -1,
            best_ask_idx: -1,
            best_bid: 0,
            best_ask: 0,
            anchor_tick_idx: -1,
            n_ticks,
            tick_int,
            fail_on_out_of_range,
        }
    }

    /// Map an absolute price_int to an array index.
    /// Returns None if anchor is not set or price is out of range.
    pub fn price_to_idx(&self, price_int: i64) -> Option<usize> {
        if self.anchor_tick_idx < 0 {
            return None;
        }
        // tick_abs = round(price_int / tick_int)
        let tick_abs = (price_int as f64 / self.tick_int as f64).round() as i64;
        let half = (self.n_ticks / 2) as i64;
        let idx = tick_abs - self.anchor_tick_idx + half;
        if idx >= 0 && (idx as usize) < self.n_ticks {
            Some(idx as usize)
        } else {
            None
        }
    }

    /// Map a grid index back to absolute price_int.
    pub fn idx_to_price_int(&self, idx: i64) -> i64 {
        if idx < 0 || self.anchor_tick_idx < 0 {
            return 0;
        }
        let half = (self.n_ticks / 2) as i64;
        let tick_abs = self.anchor_tick_idx - half + idx;
        tick_abs * self.tick_int
    }

    /// Compute best bid/ask from raw order prices (ignores grid mapping).
    pub fn raw_bbo_from_orders(&self) -> (i64, i64) {
        let mut best_bid = 0i64;
        let mut best_ask = 0i64;
        for entry in self.orders.values() {
            if entry.qty <= 0 {
                continue;
            }
            if entry.side == SIDE_BID {
                if entry.price_int > best_bid {
                    best_bid = entry.price_int;
                }
            } else if entry.side == SIDE_ASK {
                if best_ask == 0 || entry.price_int < best_ask {
                    best_ask = entry.price_int;
                }
            }
        }
        (best_bid, best_ask)
    }

    /// Recompute BBO from order map when anchor is not available.
    pub fn recompute_provisional_bbo_from_orders(&mut self) {
        let (bid, ask) = self.raw_bbo_from_orders();
        self.best_bid_idx = -1;
        self.best_ask_idx = -1;
        self.best_bid = bid;
        self.best_ask = ask;
    }

    /// Repair best bid by scanning down from start_idx.
    pub fn repair_best_bid_idx(&mut self, start_idx: i64) {
        let mut idx = start_idx.max(0).min(self.n_ticks as i64 - 1);
        while idx >= 0 && self.depth_bid[idx as usize] <= 0 {
            idx -= 1;
        }
        self.best_bid_idx = idx;
        self.best_bid = if idx >= 0 {
            self.idx_to_price_int(idx)
        } else {
            0
        };
    }

    /// Repair best ask by scanning up from start_idx.
    pub fn repair_best_ask_idx(&mut self, start_idx: i64) {
        let mut idx = start_idx.max(0).min(self.n_ticks as i64 - 1);
        while (idx as usize) < self.n_ticks && self.depth_ask[idx as usize] <= 0 {
            idx += 1;
        }
        if (idx as usize) >= self.n_ticks {
            idx = -1;
        }
        self.best_ask_idx = idx;
        self.best_ask = if idx >= 0 {
            self.idx_to_price_int(idx)
        } else {
            0
        };
    }

    /// Recompute best bid/ask from depth arrays after bulk rebuild.
    pub fn recompute_best_from_depth_arrays(&mut self) {
        let mut best_bid_idx = -1i64;
        for i in (0..self.n_ticks).rev() {
            if self.depth_bid[i] > 0 {
                best_bid_idx = i as i64;
                break;
            }
        }
        let mut best_ask_idx = -1i64;
        for i in 0..self.n_ticks {
            if self.depth_ask[i] > 0 {
                best_ask_idx = i as i64;
                break;
            }
        }
        self.best_bid_idx = best_bid_idx;
        self.best_ask_idx = best_ask_idx;
        self.best_bid = if best_bid_idx >= 0 {
            self.idx_to_price_int(best_bid_idx)
        } else {
            0
        };
        self.best_ask = if best_ask_idx >= 0 {
            self.idx_to_price_int(best_ask_idx)
        } else {
            0
        };
    }

    /// Apply signed qty delta to one depth array slot and maintain BBO.
    pub fn apply_depth_delta(
        &mut self,
        side: u8,
        idx: usize,
        delta: i64,
    ) -> Result<(), String> {
        if delta == 0 {
            return Ok(());
        }
        let arr = if side == SIDE_BID {
            &mut self.depth_bid
        } else {
            &mut self.depth_ask
        };
        let cur = arr[idx];
        let new_val = cur + delta;
        if new_val < 0 {
            return Err(format!(
                "Depth underflow at idx={idx} side={} cur={cur} delta={delta}",
                side as char
            ));
        }
        arr[idx] = new_val;

        let best_idx = if side == SIDE_BID {
            self.best_bid_idx
        } else {
            self.best_ask_idx
        };

        if side == SIDE_BID {
            if new_val > 0 && (best_idx < 0 || idx as i64 > best_idx) {
                self.best_bid_idx = idx as i64;
                self.best_bid = self.idx_to_price_int(idx as i64);
            } else if cur > 0 && new_val == 0 && idx as i64 == best_idx {
                self.repair_best_bid_idx(idx as i64);
            }
        } else {
            if new_val > 0 && (best_idx < 0 || (idx as i64) < best_idx) {
                self.best_ask_idx = idx as i64;
                self.best_ask = self.idx_to_price_int(idx as i64);
            } else if cur > 0 && new_val == 0 && idx as i64 == best_idx {
                self.repair_best_ask_idx(idx as i64);
            }
        }
        Ok(())
    }

    /// Rebuild depth arrays and BBO from current order map.
    pub fn rebuild_depth_from_orders(&mut self) {
        self.depth_bid.iter_mut().for_each(|x| *x = 0);
        self.depth_ask.iter_mut().for_each(|x| *x = 0);
        self.best_bid_idx = -1;
        self.best_ask_idx = -1;
        self.best_bid = 0;
        self.best_ask = 0;

        if self.anchor_tick_idx < 0 {
            for entry in self.orders.values_mut() {
                entry.idx = -1;
            }
            self.recompute_provisional_bbo_from_orders();
            return;
        }

        // Phase 1: compute indices for all orders
        let order_ids: Vec<u64> = self.orders.keys().cloned().collect();
        for oid in &order_ids {
            let idx_opt = {
                let entry = &self.orders[oid];
                self.price_to_idx(entry.price_int)
            };
            let entry = self.orders.get_mut(oid).unwrap();
            entry.idx = idx_opt.map(|i| i as i64).unwrap_or(-1);
        }

        // Phase 2: accumulate depth
        for entry in self.orders.values() {
            if entry.idx >= 0 {
                let i = entry.idx as usize;
                if entry.side == SIDE_BID {
                    self.depth_bid[i] += entry.qty;
                } else {
                    self.depth_ask[i] += entry.qty;
                }
            }
        }

        self.recompute_best_from_depth_arrays();
    }

    /// Clear all orders and depth.
    pub fn clear(&mut self) {
        self.orders.clear();
        self.depth_bid.iter_mut().for_each(|x| *x = 0);
        self.depth_ask.iter_mut().for_each(|x| *x = 0);
        self.best_bid_idx = -1;
        self.best_ask_idx = -1;
        self.best_bid = 0;
        self.best_ask = 0;
    }

    /// Add a new order. Returns (idx, qty_added_to_grid).
    pub fn add_order(
        &mut self,
        side: u8,
        price_int: i64,
        qty: i64,
        order_id: u64,
    ) -> Result<(i64, i64), String> {
        let idx_opt = self.price_to_idx(price_int);
        let idx_val = idx_opt.map(|i| i as i64).unwrap_or(-1);

        self.orders.insert(
            order_id,
            OrderEntry {
                side,
                price_int,
                qty,
                idx: idx_val,
            },
        );

        if idx_val >= 0 {
            self.apply_depth_delta(side, idx_val as usize, qty)?;
        } else if self.anchor_tick_idx < 0 {
            // Pre-anchor: maintain provisional BBO by raw price
            if side == SIDE_BID {
                if price_int > self.best_bid {
                    self.best_bid = price_int;
                }
            } else if self.best_ask == 0 || price_int < self.best_ask {
                self.best_ask = price_int;
            }
        }

        Ok((idx_val, qty))
    }

    /// Cancel an order. Returns (idx, qty_removed_from_grid) if existed.
    pub fn cancel_order(&mut self, order_id: u64) -> Option<(i64, i64)> {
        let entry = self.orders.remove(&order_id)?;
        if entry.idx >= 0 {
            let _ = self.apply_depth_delta(entry.side, entry.idx as usize, -entry.qty);
            Some((entry.idx, entry.qty))
        } else {
            if self.anchor_tick_idx < 0 {
                self.recompute_provisional_bbo_from_orders();
            }
            Some((-1, 0))
        }
    }

    /// Modify an existing order. Returns Some(((old_idx, old_qty), (new_idx, new_qty))) if existed pre-modify.
    pub fn modify_order(
        &mut self,
        order_id: u64,
        side: u8,
        price_int: i64,
        qty: i64,
    ) -> Result<Option<((i64, i64), (i64, i64))>, String> {
        let old = self.orders.remove(&order_id);

        let old_info = if let Some(ref e) = old {
            if e.idx >= 0 {
                self.apply_depth_delta(e.side, e.idx as usize, -e.qty)?;
            }
            Some((e.idx, e.qty))
        } else {
            None
        };

        let new_idx = self
            .price_to_idx(price_int)
            .map(|i| i as i64)
            .unwrap_or(-1);
        self.orders.insert(
            order_id,
            OrderEntry {
                side,
                price_int,
                qty,
                idx: new_idx,
            },
        );

        if new_idx >= 0 {
            self.apply_depth_delta(side, new_idx as usize, qty)?;
        } else if self.anchor_tick_idx < 0 {
            self.recompute_provisional_bbo_from_orders();
        }

        Ok(old_info.map(|(oi, oq)| ((oi, oq), (new_idx, qty))))
    }

    /// Fill an order. Returns (idx, effective_fill) if existed.
    pub fn fill_order(&mut self, order_id: u64, fill_size: i64) -> Option<(i64, i64)> {
        // Read fields immutably to avoid borrow conflict with apply_depth_delta.
        let (idx, side, current_qty) = {
            let e = self.orders.get(&order_id)?;
            (e.idx, e.side, e.qty)
        };

        let effective_fill = fill_size.min(current_qty);

        if idx >= 0 {
            let _ = self.apply_depth_delta(side, idx as usize, -effective_fill);
        }

        // Update qty; remove if exhausted.
        let exhausted = {
            let entry = self.orders.get_mut(&order_id)?;
            entry.qty -= effective_fill;
            entry.qty <= 0
        };

        if exhausted {
            self.orders.remove(&order_id);
            if self.anchor_tick_idx < 0 {
                self.recompute_provisional_bbo_from_orders();
            }
        }

        Some((idx, effective_fill))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_book() -> OrderBook {
        let mut book = OrderBook::new(500, 250_000_000, false);
        // Set anchor at tick 86000 (e.g., ES at $21500 with tick=$0.25)
        book.anchor_tick_idx = 86000;
        book
    }

    fn price_for_idx(book: &OrderBook, idx: usize) -> i64 {
        book.idx_to_price_int(idx as i64)
    }

    #[test]
    fn test_add_cancel_bbo() {
        let mut book = make_book();
        let center = 250 as usize; // idx 250 = center slot
        let bid_price = price_for_idx(&book, center - 1);
        let ask_price = price_for_idx(&book, center + 1);

        book.add_order(SIDE_BID, bid_price, 5, 1001).unwrap();
        book.add_order(SIDE_ASK, ask_price, 3, 1002).unwrap();

        assert_eq!(book.best_bid, bid_price);
        assert_eq!(book.best_ask, ask_price);

        book.cancel_order(1001);
        assert_eq!(book.best_bid, 0);
        assert_eq!(book.best_bid_idx, -1);
    }

    #[test]
    fn test_modify_qty() {
        let mut book = make_book();
        let p = price_for_idx(&book, 250);
        book.add_order(SIDE_ASK, p, 10, 2001).unwrap();
        assert_eq!(book.depth_ask[250], 10);

        // Modify same price, smaller qty (cancel partial)
        book.modify_order(2001, SIDE_ASK, p, 6).unwrap();
        assert_eq!(book.depth_ask[250], 6);
        assert_eq!(book.best_ask, p);
    }

    #[test]
    fn test_fill_removes_order() {
        let mut book = make_book();
        let p = price_for_idx(&book, 260);
        book.add_order(SIDE_ASK, p, 4, 3001).unwrap();
        book.fill_order(3001, 4);
        assert_eq!(book.depth_ask[260], 0);
        assert!(!book.orders.contains_key(&3001));
    }

    #[test]
    fn test_snapshot_clear() {
        let mut book = make_book();
        let p = price_for_idx(&book, 250);
        book.add_order(SIDE_BID, p, 10, 9001).unwrap();
        book.clear();
        assert_eq!(book.best_bid, 0);
        assert!(book.orders.is_empty());
        assert_eq!(book.depth_bid[250], 0);
    }
}
