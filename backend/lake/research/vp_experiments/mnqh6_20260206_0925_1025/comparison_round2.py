"""Comparison script for Round 2 ML experiments."""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
AGENTS = ["svm_sp", "gbm_mf", "knn_cl", "lsvm_der", "xgb_snap", "pca_ad"]
BASELINE_TP = 0.285
BREAKEVEN_TP = 0.333


def main() -> None:
    print("=" * 80)
    print("Round 2: ML Experiment Comparison")
    print("=" * 80)

    experiments = []
    for agent in AGENTS:
        results_path = BASE / "agents" / agent / "outputs" / "results.json"
        if not results_path.exists():
            print(f"  WARNING: {agent} has no results.json")
            continue
        data = json.loads(results_path.read_text())
        experiments.append(data)
        print(f"  Loaded {agent}: {data['experiment_name']}")

    print()

    # Per-experiment detail
    print("-" * 80)
    print("Per-Experiment Detail:")
    print("-" * 80)
    for exp in experiments:
        name = exp["agent_name"].upper()
        print(f"\n  {name} ({exp['experiment_name']}):")
        print(f"    Model: {exp['params'].get('model', 'N/A')}")
        print(f"    Features: {exp['params'].get('n_features', 'N/A')}")
        print(f"    {'Threshold':>12s}  {'Signals':>7s}  {'TP%':>6s}  {'PnL':>8s}  {'Ev/hr':>7s}")
        for r in exp["results_by_threshold"]:
            tp = r["tp_rate"]
            tp_str = f"{tp:.1%}" if not (isinstance(tp, float) and tp != tp) else "N/A"
            pnl = r.get("mean_pnl_ticks", float("nan"))
            pnl_str = f"{pnl:+.2f}t" if not (isinstance(pnl, float) and pnl != pnl) else "N/A"
            n = r["n_signals"]
            evhr = r.get("events_per_hour", 0)
            thr = r["threshold"]
            print(f"    {str(thr):>12s}  {n:>7d}  {tp_str:>6s}  {pnl_str:>8s}  {evhr:>7.0f}")

    # Ranking by TP rate (min 5 signals)
    print("\n" + "=" * 80)
    print("Ranking by Best TP Rate (min 5 signals):")
    print("=" * 80)

    best_list = []
    for exp in experiments:
        name = exp["agent_name"]
        tp = exp.get("best_tp_rate")
        n = exp.get("best_n_signals", 0)
        pnl = exp.get("best_mean_pnl_ticks", 0)
        evhr = exp.get("best_events_per_hour", 0)
        thr = exp.get("best_threshold")
        if tp is not None and n >= 5:
            best_list.append((name, tp, n, pnl, evhr, thr))

    best_list.sort(key=lambda x: x[1], reverse=True)

    print(f"  {'#':>2s}  {'Agent':>10s}  {'TP%':>6s}  {'Signals':>7s}  {'PnL':>8s}  {'Ev/hr':>7s}  {'Threshold':>12s}  {'vs Base':>8s}  {'vs BE':>8s}")
    for i, (name, tp, n, pnl, evhr, thr) in enumerate(best_list, 1):
        vs_base = f"+{(tp - BASELINE_TP)*100:.1f}pp"
        vs_be = "BEATS" if tp > BREAKEVEN_TP else "BELOW"
        print(f"  {i:>2d}  {name.upper():>10s}  {tp:.1%}  {n:>7d}  {pnl:+.2f}t  {evhr:>7.0f}  {str(thr):>12s}  {vs_base:>8s}  {vs_be:>8s}")

    # Ranking by PnL (min 20 signals)
    print("\n" + "=" * 80)
    print("Ranking by Mean PnL (min 20 signals):")
    print("=" * 80)

    pnl_list = []
    for exp in experiments:
        # Find best PnL threshold with min 20 signals
        valid = [r for r in exp["results_by_threshold"]
                 if r["n_signals"] >= 20
                 and not (isinstance(r["mean_pnl_ticks"], float) and r["mean_pnl_ticks"] != r["mean_pnl_ticks"])]
        if valid:
            best_pnl = max(valid, key=lambda r: r["mean_pnl_ticks"])
            pnl_list.append((
                exp["agent_name"],
                best_pnl["mean_pnl_ticks"],
                best_pnl["tp_rate"],
                best_pnl["n_signals"],
                best_pnl.get("events_per_hour", 0),
                best_pnl["threshold"],
            ))

    pnl_list.sort(key=lambda x: x[1], reverse=True)

    print(f"  {'#':>2s}  {'Agent':>10s}  {'PnL':>8s}  {'TP%':>6s}  {'Signals':>7s}  {'Ev/hr':>7s}  {'Threshold':>12s}")
    for i, (name, pnl, tp, n, evhr, thr) in enumerate(pnl_list, 1):
        print(f"  {i:>2d}  {name.upper():>10s}  {pnl:+.2f}t  {tp:.1%}  {n:>7d}  {evhr:>7.0f}  {str(thr):>12s}")

    # Cross-round comparison with Round 1
    print("\n" + "=" * 80)
    print("Cross-Round Comparison (Round 1 vs Round 2 best results):")
    print("=" * 80)

    round1_agents = ["ads", "spg", "erd", "pfp", "jad", "iirc"]
    all_best = []

    for agent in round1_agents:
        rp = BASE / "agents" / agent / "outputs" / "results.json"
        if rp.exists():
            d = json.loads(rp.read_text())
            tp = d.get("best_tp_rate")
            n = d.get("best_n_signals", 0)
            pnl = d.get("best_mean_pnl_ticks", 0)
            if tp and n >= 5:
                all_best.append(("R1", agent, tp, n, pnl))

    for name, tp, n, pnl, evhr, thr in best_list:
        all_best.append(("R2", name, tp, n, pnl))

    all_best.sort(key=lambda x: x[2], reverse=True)

    print(f"  {'#':>2s}  {'Round':>5s}  {'Agent':>10s}  {'TP%':>6s}  {'Signals':>7s}  {'PnL':>8s}")
    for i, (rnd, name, tp, n, pnl) in enumerate(all_best, 1):
        print(f"  {i:>2d}  {rnd:>5s}  {name.upper():>10s}  {tp:.1%}  {n:>7d}  {pnl:+.2f}t")


if __name__ == "__main__":
    main()
