"""Shared preprocessing + anomaly detection for AstroWatch (OPS-SAT segment-feature CSV)."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

META = ["segment", "anomaly", "train", "channel"]
LOG_COLS = ["var", "diff_var", "diff2_var", "var_div_duration", "var_div_len", "std"]
LOG1P_COLS = ["len", "duration", "gaps_squared", "len_weighted", "n_peaks", "diff_peaks", "diff2_peaks"]

def validate(df):
    missing = [c for c in ["segment", "channel"] if c not in df.columns]
    if missing:
        raise ValueError(f"Required column(s) missing: {missing}")
    feats = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    if len(feats) < 2:
        raise ValueError("Fewer than two numeric feature columns found.")
    return feats

def transform(df, feats):
    X = df[feats].astype(float).copy()
    for c in LOG_COLS:
        if c in X: X[c] = np.log10(X[c].clip(lower=1e-30))
    if "mean" in X: X["mean"] = np.sign(X["mean"]) * np.log1p(np.abs(X["mean"]) * 1e6)
    for c in LOG1P_COLS:
        if c in X: X[c] = np.log1p(X[c].clip(lower=0))
    return X

def detect(df, feats, contamination=0.2, n_estimators=300, seed=42):
    X = transform(df, feats)
    fit_mask = (df["train"] == 1) if "train" in df else pd.Series(True, index=df.index)
    sc = RobustScaler().fit(X[fit_mask]); Z = sc.transform(X)
    iso = IsolationForest(n_estimators=n_estimators, random_state=seed).fit(Z[fit_mask])
    score = -iso.score_samples(Z)
    thr = np.quantile(score[fit_mask], 1 - contamination)
    out = df.copy(); out["score"] = score; out["flag"] = (score >= thr).astype(int)
    return out, thr
