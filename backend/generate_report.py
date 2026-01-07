import re

def parse_report(file_path):
    features = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    start = False
    for line in lines:
        if line.startswith("Feature"):
            start = True
            continue
        if line.startswith("---"):
            continue
        if not start:
            continue
        if not line.strip():
            continue
            
        parts = re.split(r'\s{2,}', line.strip())
        # Re-split by pipe if formatted that way
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
        
        # Expected parts: Feature, Status, Min, Max, Mean, % Zeros
        if len(parts) >= 6:
            features.append({
                "name": parts[0],
                "status": parts[1],
                "min": parts[2],
                "max": parts[3],
                "mean": parts[4],
                "zeros_pct": parts[5]
            })
    return features

features = parse_report("feature_report.txt")

with open("FEATURE_BY_FEATURE.md", "w") as f:
    f.write("# Feature Analysis Report\n\n")
    f.write("Analysis of Silver Stage Outputs for Future (ESU5, 2025-06-04)\n\n")
    
    f.write("| Count | Feature Name | Status | Min | Max | Mean | % Zeros | Notes |\n")
    f.write("|-------|--------------|--------|-----|-----|------|---------|-------|\n")
    
    for i, feat in enumerate(features):
        note = ""
        status = feat["status"]
        
        # Add notes based on analysis
        if "rvol_" in feat["name"] and float(feat["mean"]) > 1e6:
            note = "⚠️ **Suspicious Magnitude**: Values > 1e6. Likely division by near-zero mean."
        elif "size_at_level" in feat["name"] and "setup" in feat["name"] and float(feat["mean"]) == 0:
            note = "⚠️ **Logic Error**: All zeros despite expected data. Likely missing upstream column."
        elif feat["name"] == "bar5s_meta_clear_cnt_sum" and float(feat["mean"]) == 0:
            note = "⚠️ **Data Gaps**: All zeros. Verify if 'clear' messages exist in source."
        elif "is_pm_low" in feat["name"] or "is_or_" in feat["name"]:
            note = "✅ Expected (filtered dataset)"
        elif "dist_to_level" in feat["name"] and float(feat["zeros_pct"]) > 90:
             note = "ℹ️ Sparse? Check logic."
        
        f.write(f"| {i+1} | {feat['name']} | {status} | {feat['min']} | {feat['max']} | {feat['mean']} | {feat['zeros_pct']} | {note} |\n")

    f.write("\n## Summary of Findings\n")
    f.write("1. **RVOL Feature Explosion**: `rvol_` features exhibit massive values (e.g., 1e9), indicating division by zero (or epsilon) when historical profile data is missing (early dates).\n")
    f.write("2. **Missing Setup Features**: `bar5s_setup_size_at_level_*` features are all zeros. Code inspection reveals dependency on `bar5s_lvl_size_at_level_eob` which is never computed (only `bar5s_lvl_total_size_at_level_eob` exists).\n")
    f.write("3. **Meta Clear Counts**: `bar5s_meta_clear_cnt_sum` is always 0. Requires verification against raw data.\n")

