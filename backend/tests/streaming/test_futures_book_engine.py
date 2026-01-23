import pandas as pd

from src.data_eng.stages.silver.future_mbo.book_engine import (
    F_LAST,
    F_SNAPSHOT,
    HUD_MAX_TICKS,
    TICK_INT,
    WINDOW_NS,
    compute_futures_surfaces_1s,
)


def test_futures_engine_window_valid_and_bounds():
    base = 100 * TICK_INT
    rows = [
        {"ts_event": 100_000_000, "action": "R", "side": "N", "price": 0, "size": 0, "order_id": 0, "sequence": 1, "flags": F_SNAPSHOT},
        {"ts_event": 200_000_000, "action": "A", "side": "B", "price": base, "size": 10, "order_id": 1, "sequence": 2, "flags": 0},
        {"ts_event": 300_000_000, "action": "A", "side": "A", "price": base + 2 * TICK_INT, "size": 12, "order_id": 2, "sequence": 3, "flags": 0},
        {"ts_event": 400_000_000, "action": "T", "side": "N", "price": base + TICK_INT, "size": 1, "order_id": 0, "sequence": 4, "flags": 0},
        {"ts_event": 500_000_000, "action": "A", "side": "A", "price": base + 3 * TICK_INT, "size": 8, "order_id": 3, "sequence": 5, "flags": F_LAST},
        {"ts_event": 1_100_000_000, "action": "M", "side": "B", "price": base + TICK_INT, "size": 10, "order_id": 1, "sequence": 6, "flags": 0},
        {"ts_event": 1_200_000_000, "action": "C", "side": "A", "price": 0, "size": 0, "order_id": 2, "sequence": 7, "flags": 0},
        {"ts_event": 1_300_000_000, "action": "T", "side": "N", "price": base + TICK_INT, "size": 1, "order_id": 0, "sequence": 8, "flags": 0},
    ]
    df = pd.DataFrame(rows)

    df_snap, df_wall, df_radar = compute_futures_surfaces_1s(df)

    assert len(df_snap) == 2

    w0_end = WINDOW_NS
    w1_end = 2 * WINDOW_NS

    snap_w0 = df_snap[df_snap["window_end_ts_ns"] == w0_end].iloc[0]
    snap_w1 = df_snap[df_snap["window_end_ts_ns"] == w1_end].iloc[0]

    assert bool(snap_w0["book_valid"]) is True

    wall_w0 = df_wall[df_wall["window_end_ts_ns"] == w0_end]
    wall_w1 = df_wall[df_wall["window_end_ts_ns"] == w1_end]

    assert not wall_w0.empty
    assert wall_w0["window_valid"].unique().tolist() == [False]
    assert wall_w1["window_valid"].unique().tolist() == [True]

    assert (df_wall["rel_ticks"].abs() <= HUD_MAX_TICKS).all()

    bid_row = wall_w1[(wall_w1["side"] == "B") & (wall_w1["price_int"] == snap_w1["best_bid_price_int"])]
    ask_row = wall_w1[(wall_w1["side"] == "A") & (wall_w1["price_int"] == snap_w1["best_ask_price_int"])]

    assert not bid_row.empty
    assert not ask_row.empty
    assert bid_row.iloc[0]["depth_qty_end"] == float(snap_w1["best_bid_qty"])
    assert ask_row.iloc[0]["depth_qty_end"] == float(snap_w1["best_ask_qty"])

    assert df_radar["window_end_ts_ns"].tolist() == [w0_end, w1_end]
