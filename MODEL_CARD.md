# Model Card — Tetouan Electricity Load Forecasting
**Task:** day-ahead (24h) forecast of power load, every 10 minutes, for zone1/zone2/zone3 and total.
**Data:** Tetouan city, 2017 (52,416 rows). **Models:** zone1: HistGB+ExtraTrees+Ridge, zone2: HistGB+ExtraTrees, zone3: ExtraTrees, total: ExtraTrees.
**Honest test (Dec 2017), total:** MAPE 1.89%, MAE 1158, R2 0.985.
**Inputs needed at forecast time:** weather (forecast) for the target time + load history up to 24h before it (and ~9 days of history).
**Limits:** only one year of training data; Zone 3 changes regime and is the least reliable; holidays are the weakest days.
**Retraining:** recommended monthly; monitor MAPE by hour and by zone.
