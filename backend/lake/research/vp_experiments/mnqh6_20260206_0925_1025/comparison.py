"""Compare all experiment results and rank by TP rate and PnL.

Reads results.json from each agent workspace and produces a ranked summary.
"""
from __future__ import annotations

import json
from pathlib import Path

DATASET_ID = "mnqh6_20260206_0925_1025"
BASE_DIR = Path(__file__).resolve().parent
AGENTS_DIR = BASE_DIR / "agents"
BASELINE_TP_RATE = 0.285  # current regime mode baseline

AGENT_NAMES = ["ads", "spg", "erd", "pfp", "jad", "iirc"]


def load_results(agent_name: str) -> dict | None:
    path = AGENTS_DIR / agent_name / "outputs" / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    print("=" * 80)
    print(f"EXPERIMENT COMPARISON — {DATASET_ID}")
    print(f"Baseline: TP={BASELINE_TP_RATE:.1%} (8t TP / 4t SL, breakeven=33.3%)")
    print("=" * 80)

    summaries = []

    for name in AGENT_NAMES:
        data = load_results(name)
        if data is None:
            print(f"\n  {name.upper()}: NO RESULTS")
            continue

        summaries.append({
            "agent": name,
            "experiment": data["experiment_name"],
            "best_threshold": data.get("best_threshold"),
            "best_tp_rate": data.get("best_tp_rate"),
            "best_n_signals": data.get("best_n_signals"),
            "best_mean_pnl_ticks": data.get("best_mean_pnl_ticks"),
            "best_events_per_hour": data.get("best_events_per_hour"),
            "all_results": data.get("results_by_threshold", []),
        })

    # --- Per-experiment detail ---
    print("\n" + "-" * 80)
    print("PER-EXPERIMENT RESULTS (all thresholds)")
    print("-" * 80)

    for s in summaries:
        print(f"\n  {s['agent'].upper()} — {s['experiment']}")
        print(f"  {'thr':>10s}  {'n':>5s}  {'TP%':>6s}  {'SL%':>6s}  {'PnL':>7s}  {'evts/hr':>8s}  {'vs base':>8s}")
        for r in s["all_results"]:
            tp = r.get("tp_rate", float("nan"))
            sl = r.get("sl_rate", float("nan"))
            pnl = r.get("mean_pnl_ticks", float("nan"))
            n = r.get("n_signals", 0)
            eph = r.get("events_per_hour", 0)
            thr = r.get("threshold", 0)
            delta = tp - BASELINE_TP_RATE if tp == tp else float("nan")
            tp_s = f"{tp:.1%}" if tp == tp else "N/A"
            sl_s = f"{sl:.1%}" if sl == sl else "N/A"
            pnl_s = f"{pnl:+.2f}t" if pnl == pnl else "N/A"
            delta_s = f"{delta:+.1%}" if delta == delta else "N/A"
            print(f"  {thr:>10.6f}  {n:>5d}  {tp_s:>6s}  {sl_s:>6s}  {pnl_s:>7s}  {eph:>8.1f}  {delta_s:>8s}")

    # --- Ranking by best TP rate (min 5 signals) ---
    print("\n" + "=" * 80)
    print("RANKING BY BEST TP RATE (min 5 signals)")
    print("=" * 80)

    # For each experiment, find best result with n >= 5
    ranked = []
    for s in summaries:
        valid = [r for r in s["all_results"] if r.get("n_signals", 0) >= 5]
        if not valid:
            continue
        best = max(valid, key=lambda r: r.get("tp_rate", 0))
        ranked.append({
            "agent": s["agent"],
            "experiment": s["experiment"],
            "tp_rate": best["tp_rate"],
            "n_signals": best["n_signals"],
            "mean_pnl_ticks": best.get("mean_pnl_ticks", 0),
            "events_per_hour": best.get("events_per_hour", 0),
            "threshold": best.get("threshold", 0),
        })

    ranked.sort(key=lambda x: x["tp_rate"], reverse=True)

    print(f"\n  {'#':>2s}  {'Agent':>6s}  {'TP%':>6s}  {'n':>5s}  {'PnL':>7s}  {'evts/hr':>8s}  {'thr':>10s}  {'vs base':>8s}  {'Experiment'}")
    for i, r in enumerate(ranked):
        delta = r["tp_rate"] - BASELINE_TP_RATE
        beats = "BEATS" if r["tp_rate"] > 0.333 else "below"
        print(
            f"  {i+1:>2d}  {r['agent']:>6s}  {r['tp_rate']:.1%}  {r['n_signals']:>5d}  "
            f"{r['mean_pnl_ticks']:+.2f}t  {r['events_per_hour']:>8.1f}  "
            f"{r['threshold']:>10.6f}  {delta:+.1%}  {r['experiment']}  [{beats}]"
        )

    # --- Ranking by PnL (economically meaningful) ---
    print("\n" + "=" * 80)
    print("RANKING BY MEAN PnL TICKS (min 20 signals)")
    print("=" * 80)

    pnl_ranked = []
    for s in summaries:
        valid = [r for r in s["all_results"]
                 if r.get("n_signals", 0) >= 20 and r.get("mean_pnl_ticks") == r.get("mean_pnl_ticks")]
        if not valid:
            continue
        best = max(valid, key=lambda r: r.get("mean_pnl_ticks", float("-inf")))
        pnl_ranked.append({
            "agent": s["agent"],
            "experiment": s["experiment"],
            "tp_rate": best["tp_rate"],
            "n_signals": best["n_signals"],
            "mean_pnl_ticks": best["mean_pnl_ticks"],
            "events_per_hour": best.get("events_per_hour", 0),
            "threshold": best.get("threshold", 0),
        })

    pnl_ranked.sort(key=lambda x: x["mean_pnl_ticks"], reverse=True)

    print(f"\n  {'#':>2s}  {'Agent':>6s}  {'PnL':>7s}  {'TP%':>6s}  {'n':>5s}  {'evts/hr':>8s}  {'thr':>10s}  {'Experiment'}")
    for i, r in enumerate(pnl_ranked):
        print(
            f"  {i+1:>2d}  {r['agent']:>6s}  {r['mean_pnl_ticks']:+.2f}t  {r['tp_rate']:.1%}  "
            f"{r['n_signals']:>5d}  {r['events_per_hour']:>8.1f}  "
            f"{r['threshold']:>10.6f}  {r['experiment']}"
        )

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    above_breakeven = [r for r in ranked if r["tp_rate"] > 0.333]
    print(f"\n  Experiments beating 33.3% breakeven: {len(above_breakeven)}/{len(ranked)}")
    above_baseline = [r for r in ranked if r["tp_rate"] > BASELINE_TP_RATE]
    print(f"  Experiments beating 28.5% baseline: {len(above_baseline)}/{len(ranked)}")

    if above_breakeven:
        best = above_breakeven[0]
        print(f"\n  TOP PERFORMER: {best['agent'].upper()} ({best['experiment']})")
        print(f"    TP rate: {best['tp_rate']:.1%} ({best['tp_rate'] - BASELINE_TP_RATE:+.1%} vs baseline)")
        print(f"    Signals: {best['n_signals']} ({best['events_per_hour']:.0f}/hr)")
        print(f"    Mean PnL: {best['mean_pnl_ticks']:+.2f} ticks/trade")
        print(f"    Threshold: {best['threshold']}")


if __name__ == "__main__":
    main()
