import joblib, numpy as np, pandas as pd
from feature_pipeline import build_features, TARGETS

def load_bundle(path):
    return joblib.load(path)

def forecast(hist, times, bundle, interval=True):
    # hist: raw data with weather (+ load up to 24h before times). times: DatetimeIndex to forecast.
    feats = build_features(hist, bundle["horizon"]); X = feats.loc[times, bundle["features"]].copy(); Xs = X.copy()
    if bundle.get("scaler") is not None:
        c = bundle["scaled_columns"]; Xs[c] = bundle["scaler"].transform(X[c])
    out = {}
    for t in TARGETS:
        spec = bundle["models"][t]
        ref = feats.loc[times, spec["reference"]].values if spec["reference"] else 0.0
        p = np.mean([m.predict(Xs if sc else X) for m, sc in spec["members"]], axis=0) + ref
        out[t] = p
        if interval:
            q = np.array([bundle["intervals"][t][int(h)] for h in times.hour])
            out[t + "_low"] = np.clip(p - q, 0, None); out[t + "_high"] = p + q
    return pd.DataFrame(out, index=times)
