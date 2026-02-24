//! DBN file event source for canonical vacuum-pressure streaming.
//!
//! Ports Python replay_source.py: _resolve_dbn_path(), iter_mbo_events().
//! Exposes to Python as vp_engine.iter_mbo_events and vp_engine.resolve_dbn_path.

use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use dbn::decode::dbn::Decoder as DbnDecoder;
use dbn::decode::{DbnMetadata, DecodeRecordRef};
use dbn::{MboMsg, Record, RecordRef};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

const NULL_PRICE: i64 = i64::MAX;
const F_SNAPSHOT: u8 = 32;
const F_LAST: u8 = 128;

/// Resolve the raw .dbn file path for a given symbol and date.
///
/// For futures, searches parent symbol directories (MNQH6 → MNQ, etc.).
/// For equities, uses the symbol directly.
///
/// Prefers uncompressed .dbn over .dbn.zst.
pub fn resolve_dbn_path_inner(
    lake_root: &Path,
    product_type: &str,
    symbol: &str,
    dt: &str,
) -> Result<PathBuf, String> {
    let date_compact = dt.replace('-', "");

    let parent = if product_type == "future_mbo" {
        // Try progressively shorter prefixes to find existing directory
        let mut found = symbol.to_string();
        for i in (1..=symbol.len()).rev() {
            let candidate = &symbol[..i];
            let raw_dir = lake_root
                .join("raw")
                .join("source=databento")
                .join(format!("product_type={product_type}"))
                .join(format!("symbol={candidate}"))
                .join("table=market_by_order_dbn");
            if raw_dir.exists() {
                found = candidate.to_string();
                break;
            }
        }
        found
    } else {
        symbol.to_string()
    };

    let raw_dir = lake_root
        .join("raw")
        .join("source=databento")
        .join(format!("product_type={product_type}"))
        .join(format!("symbol={parent}"))
        .join("table=market_by_order_dbn");

    if !raw_dir.exists() {
        return Err(format!(
            "Raw DBN directory not found: {}\nDownload raw data first with batch_download scripts.",
            raw_dir.display()
        ));
    }

    // Find .dbn files matching the date
    let entries: Vec<PathBuf> = fs::read_dir(&raw_dir)
        .map_err(|e| format!("Cannot read {}: {e}", raw_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            name.contains(&date_compact)
                && (name.ends_with(".dbn") || name.ends_with(".dbn.zst"))
        })
        .collect();

    if entries.is_empty() {
        return Err(format!(
            "No .dbn files found for date {dt} in {}/",
            raw_dir.display()
        ));
    }

    // Prefer uncompressed .dbn over .dbn.zst
    for path in &entries {
        if path
            .extension()
            .map(|e| e == "dbn")
            .unwrap_or(false)
        {
            return Ok(path.clone());
        }
    }
    Ok(entries[0].clone())
}

/// Compute session window nanoseconds (UTC) for a given date and product_type.
///
/// Futures: full UTC day (00:00 UTC → next-day 00:00 UTC).
/// Equities: 02:00 ET → 16:00 ET (handles EST/EDT automatically).
fn session_window_ns(dt: &str, product_type: &str) -> Result<(i64, i64), String> {
    let parts: Vec<&str> = dt.split('-').collect();
    if parts.len() != 3 {
        return Err(format!("Invalid date string: {dt}"));
    }
    let year: i64 = parts[0].parse().map_err(|_| "bad year")?;
    let month: u32 = parts[1].parse().map_err(|_| "bad month")?;
    let day: u32 = parts[2].parse().map_err(|_| "bad day")?;

    let midnight_utc = days_from_epoch(year, month, day) * 86_400 * 1_000_000_000i64;

    let is_futures =
        product_type == "future_mbo" || product_type == "future_option_mbo";

    if is_futures {
        Ok((midnight_utc, midnight_utc + 86_400 * 1_000_000_000i64))
    } else {
        // Equities: 02:00 ET → 16:00 ET
        // ET offset: EDT (UTC-4, offset=-240 min), EST (UTC-5, offset=-300 min)
        let offset_min = us_east_offset_minutes(year, month, day);
        // UTC = local - offset → UTC_ns = midnight_utc + (local_hour * 60 - offset_min) * 60 * 1e9
        let start_ns = midnight_utc + (2 * 60 - offset_min) as i64 * 60 * 1_000_000_000i64;
        let end_ns = midnight_utc + (16 * 60 - offset_min) as i64 * 60 * 1_000_000_000i64;
        Ok((start_ns, end_ns))
    }
}

/// Days since 1970-01-01 (Gregorian proleptic calendar).
fn days_from_epoch(year: i64, month: u32, day: u32) -> i64 {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let y = if month <= 2 { year - 1 } else { year };
    let m = month as i64;
    let d = day as i64;
    let m_adj = if month <= 2 { m + 9 } else { m - 3 };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * m_adj + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Day of week for a given date. 0 = Sunday, 6 = Saturday.
fn day_of_week(year: i64, month: u32, day: u32) -> u32 {
    // 1970-01-01 was a Thursday (=4)
    let days = days_from_epoch(year, month, day);
    ((days + 4).rem_euclid(7)) as u32
}

/// Day-of-month of the nth Sunday in (year, month). n=1 = first Sunday.
fn nth_sunday(year: i64, month: u32, n: u32) -> u32 {
    let dow_first = day_of_week(year, month, 1); // day of week of 1st
    let days_to_first_sunday = if dow_first == 0 { 0 } else { 7 - dow_first };
    let first_sunday = 1 + days_to_first_sunday;
    first_sunday + (n - 1) * 7
}

/// US Eastern time offset in minutes (negative = behind UTC).
/// Returns -240 (EDT) or -300 (EST).
fn us_east_offset_minutes(year: i64, month: u32, day: u32) -> i32 {
    // DST: 2nd Sunday of March → 1st Sunday of November (US rules since 2007)
    let dst_start_day = nth_sunday(year, 3, 2); // 2nd Sunday of March
    let dst_end_day = nth_sunday(year, 11, 1); // 1st Sunday of November

    let current = days_from_epoch(year, month, day);
    let dst_start = days_from_epoch(year, 3, dst_start_day);
    let dst_end = days_from_epoch(year, 11, dst_end_day);

    if current >= dst_start && current < dst_end {
        -240 // EDT
    } else {
        -300 // EST
    }
}

/// Build instrument_id → symbol map from DBN metadata mappings.
fn build_iid_to_symbol(metadata: &dbn::Metadata) -> HashMap<u32, String> {
    let mut map = HashMap::new();
    for mapping in &metadata.mappings {
        for interval in &mapping.intervals {
            // interval.symbol is the instrument_id as a string when stype_out=InstrumentId
            if let Ok(iid) = interval.symbol.parse::<u32>() {
                map.insert(iid, mapping.raw_symbol.clone());
            }
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Enum to hold either a plain or zstd decoder without lifetime issues
// ---------------------------------------------------------------------------

enum DbFile {
    Plain(DbnDecoder<std::fs::File>),
    Zstd(DbnDecoder<zstd::stream::Decoder<'static, BufReader<std::fs::File>>>),
}

impl DbFile {
    fn open(path: &Path) -> Result<Self, String> {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if name.ends_with(".dbn.zst") {
            let dec = DbnDecoder::from_zstd_file(path)
                .map_err(|e| format!("open zstd dbn: {e}"))?;
            Ok(DbFile::Zstd(dec))
        } else {
            let dec = DbnDecoder::from_file(path)
                .map_err(|e| format!("open dbn: {e}"))?;
            Ok(DbFile::Plain(dec))
        }
    }

    fn metadata(&self) -> &dbn::Metadata {
        match self {
            DbFile::Plain(d) => d.metadata(),
            DbFile::Zstd(d) => d.metadata(),
        }
    }

    fn decode_next_mbo(
        &mut self,
    ) -> Result<Option<(i64, u8, u8, i64, i64, u64, u8, u32)>, String> {
        // Returns (ts_event, action_byte, side_byte, price, size, order_id, flags_raw, instrument_id)
        loop {
            let maybe_ref: Option<RecordRef<'_>> = match self {
                DbFile::Plain(d) => d
                    .decode_record_ref()
                    .map_err(|e| format!("decode error: {e}"))?,
                DbFile::Zstd(d) => d
                    .decode_record_ref()
                    .map_err(|e| format!("decode error: {e}"))?,
            };

            let rec: RecordRef<'_> = match maybe_ref {
                None => return Ok(None),
                Some(r) => r,
            };

            // Fast path: check rtype before downcasting
            if rec.header().rtype != dbn::rtype::MBO {
                continue;
            }

            let mbo: &MboMsg = match rec.get::<MboMsg>() {
                None => continue,
                Some(m) => m,
            };

            // Extract all primitives while borrow is alive
            let ts_event = mbo.hd.ts_event as i64;
            let action_byte = mbo.action as u8;
            let side_byte = mbo.side as u8;
            let price = mbo.price;
            let size = mbo.size as i64;
            let order_id = mbo.order_id;
            let flags_raw = mbo.flags.raw();
            let instrument_id = mbo.hd.instrument_id;

            return Ok(Some((
                ts_event,
                action_byte,
                side_byte,
                price,
                size,
                order_id,
                flags_raw,
                instrument_id,
            )));
        }
    }
}

// ---------------------------------------------------------------------------
// Python iterator class
// ---------------------------------------------------------------------------

#[pyclass(module = "vp_engine")]
pub struct DbnMboIterator {
    db: DbFile,
    iid_to_symbol: HashMap<u32, String>,
    symbol: String,
    product_type: String,
    session_start_ns: i64,
    session_end_ns: i64,
    skip_to_ns: i64,
    snapshot_active: bool,
    snapshot_anchor_ts: i64,
    done: bool,
}

#[pymethods]
impl DbnMboIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(
        &mut self,
    ) -> PyResult<Option<(i64, String, String, i64, i64, u64, u8)>> {
        if self.done {
            return Ok(None);
        }

        loop {
            let raw = match self.db.decode_next_mbo() {
                Err(e) => {
                    self.done = true;
                    return Err(PyRuntimeError::new_err(e));
                }
                Ok(None) => {
                    self.done = true;
                    return Ok(None);
                }
                Ok(Some(r)) => r,
            };

            let (
                mut ts_event,
                action_byte,
                side_byte,
                mut price,
                size,
                order_id,
                flags_raw,
                instrument_id,
            ) = raw;

            let is_snapshot = (flags_raw & F_SNAPSHOT) != 0;
            let is_last = (flags_raw & F_LAST) != 0;
            let is_clear = action_byte == b'R';

            // Snapshot state machine (timestamp normalization)
            if is_clear && is_snapshot {
                self.snapshot_active = true;
                self.snapshot_anchor_ts = ts_event;
            } else if self.snapshot_active && is_last {
                self.snapshot_active = false;
            }
            // Normalize snapshot timestamps to prevent gap-fill explosion
            if is_snapshot && self.snapshot_anchor_ts > 0 {
                ts_event = self.snapshot_anchor_ts;
            }

            // Symbol filter via instrument_id
            let rec_symbol = self
                .iid_to_symbol
                .get(&instrument_id)
                .cloned()
                .unwrap_or_default();

            // For futures, drop spread symbols (contain '-')
            if self.product_type == "future_mbo" && rec_symbol.contains('-') {
                continue;
            }

            // Filter to requested contract symbol
            if !rec_symbol.is_empty() && rec_symbol != self.symbol {
                continue;
            }

            // Session window filter
            let in_window =
                self.session_start_ns <= ts_event && ts_event < self.session_end_ns;

            if !in_window && !is_snapshot && !is_clear {
                continue;
            }

            // Null price handling
            if price == NULL_PRICE {
                if action_byte == b'A' || action_byte == b'M' {
                    continue; // Skip null price on Add/Modify
                }
                price = 0;
            }

            // Fast-forward: skip non-snapshot non-clear events before skip_to_ns
            if self.skip_to_ns > 0 && ts_event < self.skip_to_ns {
                if !is_snapshot && !is_clear {
                    continue;
                }
            }

            // Convert action/side bytes to strings
            let action_str = match action_byte {
                b'A' => "A",
                b'C' => "C",
                b'M' => "M",
                b'R' => "R",
                b'T' => "T",
                b'F' => "F",
                _ => "N",
            }
            .to_string();

            let side_str = match side_byte {
                b'B' => "B",
                b'A' => "A",
                _ => "N",
            }
            .to_string();

            return Ok(Some((
                ts_event, action_str, side_str, price, size, order_id, flags_raw,
            )));
        }
    }
}

// ---------------------------------------------------------------------------
// Public Python functions
// ---------------------------------------------------------------------------

/// Resolve the raw .dbn file path for a given lake_root, product_type, symbol, and date.
///
/// Returns the path as a string. Raises RuntimeError if not found.
#[pyfunction]
pub fn resolve_dbn_path(
    lake_root: &str,
    product_type: &str,
    symbol: &str,
    dt: &str,
) -> PyResult<String> {
    resolve_dbn_path_inner(Path::new(lake_root), product_type, symbol, dt)
        .map(|p| p.to_string_lossy().to_string())
        .map_err(PyRuntimeError::new_err)
}

/// Iterate MBO events from a raw .dbn file.
///
/// Same calling convention as Python replay_source.iter_mbo_events:
///   (lake_root, product_type, symbol, dt, skip_to_ns=0)
///
/// Yields (ts_event, action, side, price, size, order_id, flags) tuples.
#[pyfunction]
#[pyo3(signature = (lake_root, product_type, symbol, dt, skip_to_ns = 0))]
pub fn iter_mbo_events(
    lake_root: &str,
    product_type: &str,
    symbol: &str,
    dt: &str,
    skip_to_ns: i64,
) -> PyResult<DbnMboIterator> {
    let dbn_path =
        resolve_dbn_path_inner(Path::new(lake_root), product_type, symbol, dt)
            .map_err(PyRuntimeError::new_err)?;

    let db = DbFile::open(&dbn_path)
        .map_err(PyRuntimeError::new_err)?;

    let iid_to_symbol = build_iid_to_symbol(db.metadata());

    let (session_start_ns, session_end_ns) =
        session_window_ns(dt, product_type).map_err(PyRuntimeError::new_err)?;

    Ok(DbnMboIterator {
        db,
        iid_to_symbol,
        symbol: symbol.to_string(),
        product_type: product_type.to_string(),
        session_start_ns,
        session_end_ns,
        skip_to_ns,
        snapshot_active: false,
        snapshot_anchor_ts: 0,
        done: false,
    })
}
