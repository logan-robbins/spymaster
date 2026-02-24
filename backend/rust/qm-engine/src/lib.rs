//! qm_engine — Platform-layer Rust extension (order book, EMA chain, DBN source).
//!
//! Exposes:
//!   - AbsoluteTickEngine: tick-level book + EMA derivative + force engine
//!   - iter_mbo_events:    DBN file iterator (same signature as Python replay_source)
//!   - resolve_dbn_path:   path resolver for book cache computation
//!   - DbnMboIterator:     iterator class (indirectly via iter_mbo_events)

mod book;
mod dbn_source;
mod ema;
mod engine;

use pyo3::prelude::*;

#[pymodule]
fn qm_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<engine::AbsoluteTickEngine>()?;
    m.add_class::<dbn_source::DbnMboIterator>()?;
    m.add_function(wrap_pyfunction!(dbn_source::iter_mbo_events, m)?)?;
    m.add_function(wrap_pyfunction!(dbn_source::resolve_dbn_path, m)?)?;
    Ok(())
}
