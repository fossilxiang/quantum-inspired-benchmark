
import numpy as np, pandas as pd

def summarize_results(df, ref_col="reference_value"):
    df["gap"] = df.apply(lambda r: (r["value"] - r[ref_col]) / max(1.0, abs(r[ref_col])) * 100, axis=1)
    df["success"] = df["gap"].apply(lambda g: g <= 1.0)
    groups = df.groupby(["algorithm", "problem", "n", "instance"])
    s = groups.agg(mean_gap=("gap","mean"), median_gap=("gap","median"), std_gap=("gap","std"),
                   success_rate=("success","mean"), mean_time=("time","mean"), n_runs=("gap","count")).reset_index()
    tts_rows = []
    for key, grp in groups:
        st = grp[grp["success"]]["time"].values
        tts = float(np.median(st)) if len(st) > 0 else float("inf")
        tts_rows.append({"algorithm":key[0],"problem":key[1],"n":key[2],"instance":key[3],"tts":tts})
    tts_df = pd.DataFrame(tts_rows)
    return s.merge(tts_df, on=["algorithm","problem","n","instance"])

def cross_summary(s):
    return s.groupby(["algorithm","problem","n"]).agg(
        mean_gap=("mean_gap","mean"), std_gap_across=("mean_gap","std"),
        success_rate=("success_rate","mean"), tts_median=("tts",lambda x: float(np.median(x))),
        mean_time=("mean_time","mean")).reset_index()
