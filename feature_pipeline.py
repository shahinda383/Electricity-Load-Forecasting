import numpy as np, pandas as pd
WEATHER = ["temp", "humidity", "wind", "gen_diffuse", "diffuse"]
ZONES = ["zone1", "zone2", "zone3"]; TARGETS = ZONES + ["total"]
HOLIDAYS = pd.to_datetime(['2017-01-01', '2017-01-11', '2017-05-01', '2017-06-25', '2017-06-26', '2017-07-30', '2017-08-14', '2017-08-20', '2017-08-21', '2017-09-01', '2017-09-02', '2017-09-21', '2017-11-06', '2017-11-18', '2017-12-01'])
RAMADAN = (pd.Timestamp("2017-05-27"), pd.Timestamp("2017-06-24"))

def calendar_features(idx):
    f = pd.DataFrame(index=idx)
    f["hour"] = idx.hour; f["slot"] = idx.hour * 6 + idx.minute // 10; f["dow"] = idx.dayofweek
    f["day"] = idx.day; f["month"] = idx.month; f["doy"] = idx.dayofyear
    f["week"] = idx.isocalendar().week.values.astype(int); f["is_weekend"] = (f.dow >= 5).astype(int)
    f["is_daytime"] = ((f.hour >= 7) & (f.hour < 19)).astype(int)
    f["is_evening_peak"] = ((f.hour >= 18) & (f.hour <= 22)).astype(int)
    for n, v, p in [("slot", f.slot, 144), ("dow", f.dow, 7), ("month", f.month, 12), ("doy", f.doy, 365)]:
        f[n + "_sin"] = np.sin(2 * np.pi * v / p); f[n + "_cos"] = np.cos(2 * np.pi * v / p)
    d = idx.normalize()
    f["is_holiday"] = d.isin(HOLIDAYS).astype(int)
    f["is_pre_holiday"] = (d + pd.Timedelta(days=1)).isin(HOLIDAYS).astype(int)
    f["is_post_holiday"] = (d - pd.Timedelta(days=1)).isin(HOLIDAYS).astype(int)
    f["is_ramadan"] = ((d >= RAMADAN[0]) & (d <= RAMADAN[1])).astype(int)
    f["is_day_off"] = ((f.is_weekend == 1) | (f.is_holiday == 1)).astype(int)
    return f

def weather_features(w, f):
    f[WEATHER] = w[WEATHER]
    f["cdd"] = (w.temp - 15).clip(lower=0); f["hdd"] = (10 - w.temp).clip(lower=0)
    f["temp_x_hum"] = w.temp * w.humidity / 100; f["diffuse_ratio"] = w.diffuse / (w.gen_diffuse + 1)
    f["is_daylight"] = (w.gen_diffuse > 5).astype(int); f["wind_high"] = (w.wind > 1).astype(int)
    for k in (6, 36, 144): f["temp_rmean%d" % k] = w.temp.rolling(k, min_periods=1).mean()
    f["temp_max144"] = w.temp.rolling(144, min_periods=1).max(); f["temp_min144"] = w.temp.rolling(144, min_periods=1).min()
    f["temp_change6"] = (w.temp - w.temp.shift(6)).fillna(0)
    f["hum_rmean36"] = w.humidity.rolling(36, min_periods=1).mean()
    f["gen_diffuse_rmean36"] = w.gen_diffuse.rolling(36, min_periods=1).mean()
    f["cdd_rmean144"] = f.cdd.rolling(144, min_periods=1).mean()
    return f

def load_features(h, horizon):
    lags = sorted({horizon + k for k in (0, 1, 2, 6, 36)} | {l for l in (144, 288, 1008) if l >= horizon}); o = {}
    for c in TARGETS:
        for l in lags: o["%s_lag%d" % (c, l)] = h[c].shift(l)
        s = h[c].shift(horizon)
        for k in (6, 36, 144): o["%s_rmean%d" % (c, k)] = s.rolling(k).mean()
        o[c + "_rstd36"] = s.rolling(36).std(); o[c + "_rmax144"] = s.rolling(144).max(); o[c + "_rmin144"] = s.rolling(144).min()
        days = [h[c].shift(144 * k) for k in range(1, 8) if 144 * k >= horizon]
        o[c + "_slot_mean7d"] = pd.concat(days, axis=1).mean(axis=1)
    return pd.DataFrame(o, index=h.index)

def build_features(hist, horizon=144):
    h = hist.copy()
    if "total" not in h: h["total"] = h[ZONES].sum(axis=1)
    f = weather_features(h, calendar_features(h.index))
    f = pd.concat([f, load_features(h, horizon)], axis=1)
    for z in ZONES: f[z + "_share_lag"] = h[z].shift(horizon) / h.total.shift(horizon)
    f["cdd_x_evening"] = f.cdd * f.is_evening_peak; f["cdd_x_dayoff"] = f.cdd * f.is_day_off
    f["lag_vs_week"] = f["total_lag%d" % horizon] - f["total_lag1008"]
    return f
