import os, json, io, html
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from inference import load_bundle, forecast

# ---------------------- Page setup ----------------------
st.set_page_config(page_title="WATT / Electricity Intelligence", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "history_2017.csv")
MODEL_PATH = os.path.join(ROOT, "model_production.pkl")
META_PATH = os.path.join(ROOT, "metadata.json")

@st.cache_data
def load_history():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    return df

@st.cache_resource
def load_model():
    return load_bundle(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

hist = load_history()
bundle = load_model()
meta = load_metadata()

# ---------------------- Theme / language ----------------------
with st.sidebar:
    st.markdown("## ⚡ WATT")
    st.caption("POWER INTELLIGENCE STUDIO")
    lang = st.radio("Language / اللغة", ["English", "العربية"], horizontal=True)
    dark = st.toggle("Dark mode / الوضع الداكن", value=False)
    st.divider()
    page = st.radio("Navigate", ["Command Center", "Prediction Studio", "Forecast Lab", "Model Observatory", "Scenario Studio", "AI Analyst", "Export Center"] if lang=="English" else ["مركز القيادة", "استوديو التنبؤ المستقبلي", "مختبر التنبؤ", "مرصد النماذج", "رادار الشذوذ", "استوديو السيناريوهات", "المحلل الذكي", "مركز التقارير"])
    st.divider()
    st.markdown("**Forecast horizon**")
    st.markdown("24 hours · 144 steps · 10-minute resolution")
    st.caption("Tetouan City · 2017 reference dataset")

ar = lang == "العربية"
labels = {
"Command Center": "مركز القيادة", "Prediction Studio":"استوديو التنبؤ المستقبلي", "Forecast Lab":"مختبر التنبؤ", "Model Observatory":"مرصد النماذج",
"Anomaly Radar":"رادار الشذوذ", "Scenario Studio":"استوديو السيناريوهات", "AI Analyst":"المحلل الذكي", "Export Center":"مركز التقارير"
}
english_pages = {v:k for k,v in labels.items()}
page_en = english_pages.get(page,page)
bg = "#071521" if dark else "#F3F7FA"
panel = "#102535" if dark else "#FFFFFF"
ink = "#EAF4FA" if dark else "#152B3A"
muted = "#9BB2C2" if dark else "#607789"
accent = "#42D6C5"
st.markdown(f"""
<style>
:root {{ color-scheme: {'dark' if dark else 'light'}; }}
.stApp {{ background: {bg}; color: {ink}; }}
[data-testid="stSidebar"] {{ background: {'#081925' if dark else '#E8F1F6'}; border-right: 1px solid {'#244252' if dark else '#D6E3EB'}; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }}
h1,h2,h3 {{ letter-spacing: -0.035em; color: {ink}; }}
.hero {{ padding: 30px 32px; border-radius: 24px; background: linear-gradient(120deg, {'#0E2B3D,#124B56' if dark else '#D9F3F0,#E5EEF9'}); border: 1px solid {'#245366' if dark else '#D0E7E9'}; margin-bottom: 20px; }}
.eyebrow {{ color: {accent if dark else '#087F79'}; font-size: 0.75rem; font-weight: 800; letter-spacing: .18em; }}
.hero-title {{ font-size: clamp(2rem,4vw,3.4rem); font-weight: 850; line-height: 1.03; margin: 10px 0; color: {ink}; }}
.hero-sub {{ color: {muted}; font-size: 1rem; max-width: 850px; }}
.metric {{ padding: 18px 20px; border-radius: 18px; background: {panel}; border: 1px solid {'#203E50' if dark else '#DFE8EF'}; min-height: 125px; }}
.metric-label {{ color: {muted}; font-size: .78rem; text-transform: uppercase; letter-spacing: .10em; }}
.metric-value {{ color: {ink}; font-size: 1.85rem; font-weight: 800; margin-top: 8px; }}
.metric-note {{ color: {muted}; font-size: .8rem; margin-top: 3px; }}
.section-kicker {{ color: {accent if dark else '#087F79'}; text-transform: uppercase; font-size: .75rem; font-weight: 800; letter-spacing: .15em; }}
.insight {{ border-left: 3px solid {accent}; background: {panel}; padding: 14px 17px; border-radius: 0 14px 14px 0; margin: 8px 0; }}
.small-muted {{ color: {muted}; font-size: .84rem; }}
div[data-testid="stMetric"] {{ background:{panel}; padding:16px 18px; border:1px solid {'#203E50' if dark else '#DFE8EF'}; border-radius:16px; }}
.stButton>button {{ border-radius: 12px; font-weight: 700; min-height: 2.7rem; }}
</style>
""", unsafe_allow_html=True)

def t(en, ar_text):
    return ar_text if ar else en

def hero(kicker, title, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div></div>', unsafe_allow_html=True)

def metric_card(label, value, note=""):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def plot_layout(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=8,r=8,t=38,b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=ink), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#294352" if dark else "#E5EDF2", zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False, "scrollZoom":True})

TARGETS = ["zone1","zone2","zone3","total"]
target_labels = {"total":"City total","zone1":"Zone 1","zone2":"Zone 2","zone3":"Zone 3"}
metrics = meta["honest_test_metrics_dec2017"]
total_m = metrics["total"]
day_count = hist.index.normalize().nunique()
peak = hist["total"].max()
avg = hist["total"].mean()
peak_time = hist["total"].idxmax()
daily = hist["total"].resample("D").mean()

# ---------------------- Command Center ----------------------
if page_en == "Command Center":
    hero("WATT / POWER INTELLIGENCE", t("From raw demand to operational foresight.", "من بيانات الاستهلاك إلى رؤية تشغيلية استباقية."),
         t("A model-backed control room for Tetouan electricity demand — built around real backtest results, interpretable signals, and uncertainty-aware forecasts.",
           "لوحة تحكم ذكية لاستهلاك الكهرباء في تطوان، مبنية على نتائج الاختبار الفعلية وتفسير العوامل وعدم اليقين في التوقعات."))
    c = st.columns(4)
    with c[0]: metric_card(t("TEST MAPE · TOTAL","خطأ MAPE للاختبار"), f'{total_m["MAPE_%"]:.2f}%', t("December 2017 holdout","اختبار ديسمبر 2017"))
    with c[1]: metric_card(t("TEST R² · TOTAL","معامل R² للاختبار"), f'{total_m["R2"]:.3f}', t("Out-of-time evaluation","تقييم زمني منفصل"))
    with c[2]: metric_card(t("MAE · TOTAL","متوسط الخطأ المطلق"), f'{total_m["MAE"]:,.0f}', t("Load units / 10 min","وحدة حمل لكل 10 دقائق"))
    with c[3]: metric_card(t("FORECAST HORIZON","أفق التنبؤ"), "24 h", t("144 × 10-minute steps","144 خطوة كل 10 دقائق"))
    st.markdown("")
    left,right = st.columns([1.65,1])
    with left:
        st.markdown(f'<div class="section-kicker">{t("THE CITY, IN MOTION","المدينة بالأرقام")}</div>',unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=daily.index,y=daily.values,mode="lines",name=t("Daily average load","متوسط الحمل اليومي"),line=dict(color="#16A89A",width=2.5),fill="tozeroy",fillcolor="rgba(22,168,154,.12)"))
        fig.add_hline(y=avg,line_dash="dot",line_color="#E6A23C",annotation_text=t("Annual mean","المتوسط السنوي"))
        fig.update_layout(title=t("Demand rhythm across 2017","إيقاع الطلب خلال 2017"),xaxis_title="",yaxis_title=t("Load","الحمل"))
        plot_layout(fig,350)
    with right:
        st.markdown(f'<div class="section-kicker">{t("MODEL SCORECARD","بطاقة أداء النماذج")}</div>',unsafe_allow_html=True)
        mm=pd.DataFrame([{"Target":target_labels[k],"MAPE (%)":v["MAPE_%"],"R²":v["R2"],"MAE":v["MAE"]} for k,v in metrics.items()])
        fig=px.bar(mm,x="MAPE (%)",y="Target",orientation="h",text="MAPE (%)",color="MAPE (%)",color_continuous_scale=["#30BFAE","#F3C969","#E77C62"])
        fig.update_layout(title=t("Lower MAPE is better","كلما قل MAPE كان أفضل"),coloraxis_showscale=False)
        plot_layout(fig,350)
    st.markdown(f'<div class="section-kicker">{t("SIGNALS WORTH KNOWING","إشارات مهمة")}</div>',unsafe_allow_html=True)
    i1,i2,i3=st.columns(3)
    with i1: st.markdown(f'<div class="insight"><b>{t("The total-load model is strong","نموذج الحمل الكلي قوي")}</b><br><span class="small-muted">{t("December holdout MAPE is 1.89% and R² is 0.985; these are historical test metrics, not a guarantee for future months.","حقق اختبار ديسمبر MAPE قدره 1.89% وR² قدره 0.985؛ وهي نتائج تاريخية وليست ضمانًا للمستقبل.")}</span></div>',unsafe_allow_html=True)
    with i2: st.markdown(f'<div class="insight"><b>{t("Weather is not the whole story","الطقس ليس العامل الوحيد")}</b><br><span class="small-muted">{t("Notebook analysis attributes about 50.2% of importance to past load, 43.1% to calendar, and 6.7% to weather for the total model.","تحليل النوتبوك يقدّر أهمية الحمل السابق بنحو 50.2% والتقويم 43.1% والطقس 6.7% لنموذج الإجمالي.")}</span></div>',unsafe_allow_html=True)
    with i3: st.markdown(f'<div class="insight"><b>{t("Zone 3 needs extra caution","المنطقة الثالثة تحتاج حذرًا")}</b><br><span class="small-muted">{t("Its December MAPE is 4.25%, and the notebook flags a regime that available features do not fully explain.","بلغ MAPE للمنطقة الثالثة 4.25% في ديسمبر، ويشير التحليل إلى تغيرات لا تفسرها الخصائص المتاحة بالكامل.")}</span></div>',unsafe_allow_html=True)
    st.caption(t("Data note: the supplied dataset covers 2017 only. Forecast replay uses real timestamps and saved model bundles; this dashboard does not pretend that 2017 data is live telemetry.",
                 "ملاحظة البيانات: الداتا المتاحة تغطي عام 2017 فقط. وضع إعادة التنبؤ يستخدم التوقيتات الفعلية والموديلات المحفوظة، ولا يعرض بيانات 2017 على أنها بيانات مباشرة."))


# ---------------------- Prediction Studio: real future inference ----------------------
elif page_en == "Prediction Studio":
    hero("PREDICTION STUDIO",
         t("Forecast the next 24 hours — using your latest data.",
           "تنبأ بالـ24 ساعة القادمة باستخدام أحدث بياناتك."),
         t("This page runs the saved production model on future timestamps. It requires recent 10-minute load history and weather inputs for every future step; the app will not silently treat 2017 as live data.",
           "هذه الصفحة تشغّل موديل الإنتاج المحفوظ على توقيتات مستقبلية. يلزم سجل حمل حديث كل 10 دقائق ومدخلات طقس لكل خطوة مستقبلية؛ ولن نعرض بيانات 2017 على أنها بيانات مباشرة."))

    st.markdown("### " + t("1 · Load recent history", "١ · ارفع سجل الاستهلاك الحديث"))
    st.caption(t("CSV columns: datetime, temp, humidity, wind, gen_diffuse, diffuse, zone1, zone2, zone3. The app derives total if missing.",
                 "أعمدة CSV: datetime, temp, humidity, wind, gen_diffuse, diffuse, zone1, zone2, zone3. التطبيق يحسب total إذا لم يكن موجودًا."))
    hist_file = st.file_uploader(t("Upload recent load + weather history (CSV)", "ارفع سجل الحمل والطقس الحديث (CSV)"),
                                 type=["csv"], key="future_history_upload")
    use_demo_history = st.checkbox(t("Use bundled 2017 data for a demonstration only",
                                     "استخدم بيانات 2017 المرفقة للتجربة فقط"), value=(hist_file is None))
    source_hist = None
    history_label = ""
    if hist_file is not None:
        try:
            raw_h = pd.read_csv(hist_file)
            if "datetime" not in raw_h.columns:
                st.error(t("History CSV must contain a 'datetime' column.", "ملف التاريخ يجب أن يحتوي عمود datetime."))
            else:
                raw_h["datetime"] = pd.to_datetime(raw_h["datetime"], errors="coerce")
                raw_h = raw_h.dropna(subset=["datetime"]).drop_duplicates("datetime").sort_values("datetime")
                source_hist = raw_h.set_index("datetime")
                history_label = hist_file.name
        except Exception as e:
            st.error(t("Could not read the history CSV: ", "تعذر قراءة ملف التاريخ: ") + str(e))
    elif use_demo_history:
        source_hist = hist.copy()
        history_label = "Bundled Tetouan 2017 demo"
    if source_hist is not None:
        required_hist = ["temp", "humidity", "wind", "gen_diffuse", "diffuse", "zone1", "zone2", "zone3"]
        missing_hist = [c for c in required_hist if c not in source_hist.columns]
        if missing_hist:
            st.error(t("Missing history columns: ", "أعمدة ناقصة في سجل التاريخ: ") + ", ".join(missing_hist))
            source_hist = None
        else:
            for c in required_hist:
                source_hist[c] = pd.to_numeric(source_hist[c], errors="coerce")
            if "total" not in source_hist.columns:
                source_hist["total"] = source_hist[["zone1", "zone2", "zone3"]].sum(axis=1)
            else:
                source_hist["total"] = pd.to_numeric(source_hist["total"], errors="coerce")
            source_hist = source_hist[~source_hist.index.duplicated(keep="last")].sort_index()
            source_hist = source_hist[required_hist + ["total"]]
            if len(source_hist):
                st.info(t(f"Selected history: {history_label} · {source_hist.index.min()} → {source_hist.index.max()} · {len(source_hist):,} rows",
                          f"السجل المختار: {history_label} · من {source_hist.index.min()} إلى {source_hist.index.max()} · {len(source_hist):,} صف"))

    st.markdown("### " + t("2 · Supply future weather", "٢ · أدخل الطقس المتوقع"))
    st.write(t("The model needs five weather fields at each of the 144 future timestamps. Upload a forecast CSV, or use an explicitly labelled constant-weather assumption for a demonstration.",
               "الموديل يحتاج خمس خصائص طقس لكل واحدة من الـ144 خطوة المستقبلية. ارفع ملف توقعات الطقس، أو استخدم افتراض طقس ثابتًا معلنًا للتجربة فقط."))
    weather_mode = st.radio(t("Future weather input", "طريقة إدخال الطقس"),
                            [t("Upload weather forecast CSV", "رفع ملف توقعات الطقس"),
                             t("Use constant assumptions (demo only)", "استخدام افتراضات ثابتة (للتجربة فقط)")],
                            horizontal=True, key="future_weather_mode")
    weather_file = None
    future_weather = None
    future_idx = None
    if source_hist is not None and len(source_hist):
        last_ts = source_hist.index.max()
        future_idx = pd.date_range(last_ts + pd.Timedelta(minutes=10), periods=144, freq="10min")
        template = pd.DataFrame({"datetime": future_idx,
                                 "temp": [np.nan]*144, "humidity": [np.nan]*144, "wind": [np.nan]*144,
                                 "gen_diffuse": [np.nan]*144, "diffuse": [np.nan]*144})
        st.download_button(t("Download 24-hour weather CSV template", "تحميل قالب طقس الـ24 ساعة"),
                           template.to_csv(index=False).encode("utf-8"),
                           file_name="watt_future_weather_template.csv", mime="text/csv")
        if weather_mode == t("Upload weather forecast CSV", "رفع ملف توقعات الطقس"):
            weather_file = st.file_uploader(t("Upload weather forecast CSV (144 rows)", "ارفع ملف توقعات الطقس (144 صفًا)"),
                                            type=["csv"], key="future_weather_upload")
            if weather_file is not None:
                try:
                    raw_w = pd.read_csv(weather_file)
                    if "datetime" not in raw_w.columns:
                        st.error(t("Weather CSV must contain a 'datetime' column.", "ملف الطقس يجب أن يحتوي عمود datetime."))
                    else:
                        raw_w["datetime"] = pd.to_datetime(raw_w["datetime"], errors="coerce")
                        raw_w = raw_w.dropna(subset=["datetime"]).drop_duplicates("datetime").set_index("datetime").sort_index()
                        weather_cols = ["temp", "humidity", "wind", "gen_diffuse", "diffuse"]
                        missing_w = [c for c in weather_cols if c not in raw_w.columns]
                        if missing_w:
                            st.error(t("Missing weather columns: ", "أعمدة طقس ناقصة: ") + ", ".join(missing_w))
                        elif not raw_w.index.equals(future_idx):
                            st.error(t("Weather timestamps must exactly match the 144 timestamps shown in the template, at 10-minute intervals.",
                                       "يجب أن تطابق توقيتات الطقس الـ144 توقيتًا في القالب تمامًا، بفاصل 10 دقائق."))
                        elif raw_w[weather_cols].apply(pd.to_numeric, errors="coerce").isna().any().any():
                            st.error(t("Weather values contain blanks or non-numeric values.", "قيم الطقس تحتوي فراغات أو قيمًا غير رقمية."))
                        else:
                            future_weather = raw_w[weather_cols].astype(float)
                except Exception as e:
                    st.error(t("Could not read the weather forecast: ", "تعذر قراءة ملف الطقس: ") + str(e))
        else:
            st.warning(t("Demonstration only: the selected weather values are assumptions, not an external weather forecast.",
                         "للتجربة فقط: قيم الطقس المختارة افتراضات وليست توقعات طقس من مصدر خارجي."))
            last_window = source_hist.tail(144)
            defaults = {
                "temp": float(last_window["temp"].mean()),
                "humidity": float(last_window["humidity"].mean()),
                "wind": float(last_window["wind"].mean()),
                "gen_diffuse": float(last_window["gen_diffuse"].mean()),
                "diffuse": float(last_window["diffuse"].mean())
            }
            w1, w2, w3 = st.columns(3)
            with w1:
                demo_temp = st.slider(t("Assumed temperature (°C)", "الحرارة المفترضة (°م)"), -5.0, 50.0,
                                      float(np.clip(defaults["temp"], -5, 50)), 0.5)
                demo_hum = st.slider(t("Assumed humidity (%)", "الرطوبة المفترضة (%)"), 0.0, 100.0,
                                     float(np.clip(defaults["humidity"], 0, 100)), 1.0)
            with w2:
                demo_wind = st.slider(t("Assumed wind", "الرياح المفترضة"), 0.0, 20.0,
                                      float(np.clip(defaults["wind"], 0, 20)), 0.1)
                demo_gen = st.number_input(t("Assumed global/diffuse radiation feature", "إشعاع gen_diffuse المفترض"),
                                           min_value=0.0, value=float(max(0, defaults["gen_diffuse"])), step=0.1)
            with w3:
                demo_diffuse = st.number_input(t("Assumed diffuse radiation", "الإشعاع diffuse المفترض"),
                                               min_value=0.0, value=float(max(0, defaults["diffuse"])), step=0.1)
            future_weather = pd.DataFrame({
                "temp": demo_temp, "humidity": demo_hum, "wind": demo_wind,
                "gen_diffuse": demo_gen, "diffuse": demo_diffuse
            }, index=future_idx)

    st.markdown("### " + t("3 · Generate the future forecast", "٣ · أنشئ التنبؤ المستقبلي"))
    st.caption(t("Required history: at least 9 days, complete 10-minute timestamps, no gaps in the final 9-day window.",
                 "السجل المطلوب: 9 أيام على الأقل، توقيتات مكتملة كل 10 دقائق، دون فجوات في آخر 9 أيام."))
    ready = source_hist is not None and future_idx is not None and future_weather is not None
    if ready:
        expected_history = pd.date_range(source_hist.index.max() - pd.Timedelta(minutes=10* (9*144-1)),
                                         source_hist.index.max(), freq="10min")
        last9 = source_hist.loc[expected_history[0]:expected_history[-1]]
        history_ok = len(source_hist) >= 9*144 and last9.index.equals(expected_history)
        required_numeric = ["temp", "humidity", "wind", "gen_diffuse", "diffuse", "zone1", "zone2", "zone3", "total"]
        history_ok = history_ok and not source_hist[required_numeric].tail(9*144).isna().any().any()
        weather_ok = future_weather.index.equals(future_idx) and not future_weather.isna().any().any()
        if not history_ok:
            st.error(t("History is incomplete: provide 9 continuous days of complete 10-minute readings through the latest timestamp.",
                       "السجل غير مكتمل: وفّر 9 أيام متصلة من القراءات كل 10 دقائق حتى أحدث توقيت."))
        elif not weather_ok:
            st.warning(t("Complete the future weather CSV before forecasting.", "أكمل ملف الطقس المستقبلي قبل التنبؤ."))
        elif st.button(t("⚡ Run 24-hour future forecast", "⚡ شغّل التنبؤ المستقبلي لـ24 ساعة"), type="primary", use_container_width=True):
            try:
                combined = source_hist.copy()
                future_block = future_weather.copy()
                for c in ["zone1", "zone2", "zone3", "total"]:
                    future_block[c] = np.nan
                combined = pd.concat([combined, future_block], axis=0)
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                with st.spinner(t("Running the saved production model…", "جارٍ تشغيل موديل الإنتاج المحفوظ…")):
                    pred_future = forecast(combined, future_idx, bundle, interval=True)
                out_future = pred_future.copy()
                out_future.index.name = "datetime"
                out_future = out_future.rename(columns={
                    "zone1":"Zone 1 forecast", "zone1_low":"Zone 1 lower", "zone1_high":"Zone 1 upper",
                    "zone2":"Zone 2 forecast", "zone2_low":"Zone 2 lower", "zone2_high":"Zone 2 upper",
                    "zone3":"Zone 3 forecast", "zone3_low":"Zone 3 lower", "zone3_high":"Zone 3 upper",
                    "total":"Total forecast", "total_low":"Total lower", "total_high":"Total upper"
                })
                st.session_state["watt_future_forecast"] = out_future
                st.session_state["watt_future_forecast_meta"] = {
                    "created_at": pd.Timestamp.now().isoformat(),
                    "history_label": history_label,
                    "weather_mode": "uploaded forecast" if weather_mode == t("Upload weather forecast CSV", "رفع ملف توقعات الطقس") else "constant demo assumptions"
                }
                st.success(t("Future forecast generated successfully.", "تم إنشاء التنبؤ المستقبلي بنجاح."))
            except Exception as e:
                st.error(t("Forecast failed. Check the feature columns and model files. Details: ", "فشل التنبؤ. تحقق من الأعمدة وملفات الموديل. التفاصيل: ") + str(e))

    saved_future = st.session_state.get("watt_future_forecast")
    if saved_future is not None:
        st.markdown("### " + t("Forecast results", "نتائج التنبؤ"))
        st.caption(t("Generated: " + st.session_state.get("watt_future_forecast_meta", {}).get("created_at", ""),
                     "وقت الإنشاء: " + st.session_state.get("watt_future_forecast_meta", {}).get("created_at", "")))
        target = st.selectbox(t("Series to display", "السلسلة المعروضة"),
                              ["total", "zone1", "zone2", "zone3"], index=0, key="future_result_target",
                              format_func=lambda x: target_labels[x])
        forecast_col = {"total":"Total forecast","zone1":"Zone 1 forecast","zone2":"Zone 2 forecast","zone3":"Zone 3 forecast"}[target]
        low_col = {"total":"Total lower","zone1":"Zone 1 lower","zone2":"Zone 2 lower","zone3":"Zone 3 lower"}[target]
        high_col = {"total":"Total upper","zone1":"Zone 1 upper","zone2":"Zone 2 upper","zone3":"Zone 3 upper"}[target]
        show = saved_future[[forecast_col, low_col, high_col]]
        c1, c2, c3 = st.columns(3)
        with c1: st.metric(t("Average forecast", "متوسط الحمل المتوقع"), f"{show[forecast_col].mean():,.0f}")
        with c2: st.metric(t("Peak forecast", "أعلى حمل متوقع"), f"{show[forecast_col].max():,.0f}")
        with c3: st.metric(t("Peak timestamp", "توقيت الذروة"), str(show[forecast_col].idxmax()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=show.index, y=show[high_col], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=show.index, y=show[low_col], line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(22,168,154,.16)", name=t("Empirical 90% band", "نطاق معايرة تجريبي 90%")))
        fig.add_trace(go.Scatter(x=show.index, y=show[forecast_col], name=t("Forecast", "التوقع"),
                                 line=dict(color="#13A99A", width=2.6)))
        fig.update_layout(title=t("Next 24 hours · 144 steps", "الـ24 ساعة القادمة · 144 خطوة"),
                          yaxis_title=t("Predicted load", "الحمل المتوقع"))
        plot_layout(fig, 440)
        st.dataframe(saved_future.round(2), use_container_width=True)
        st.download_button(t("Download full 24-hour forecast CSV", "تحميل توقعات الـ24 ساعة CSV"),
                           saved_future.to_csv().encode("utf-8"), file_name="watt_future_forecast.csv",
                           mime="text/csv", type="primary")
        st.info(t("The displayed band is an empirical hour-specific calibration interval, not a guaranteed statistical confidence interval. If constant-weather demo mode was used, results are conditional on those assumptions.",
                  "النطاق المعروض معايرة تجريبية حسب الساعة وليس فاصل ثقة مضمونًا. إذا استخدمت افتراضات الطقس الثابتة، فالنتائج مشروطة بهذه الافتراضات."))

# ---------------------- Forecast Lab ----------------------
elif page_en == "Forecast Lab":
    hero("FORECAST LAB",t("See tomorrow before it arrives.","شوف استهلاك بكرة قبل ما يوصل."),t("Replay the production model on a selected historical day, inspect the uncertainty band, and compare prediction with actual observed load.","شغّل نموذج الإنتاج على يوم تاريخي محدد، وشاهد نطاق عدم اليقين وقارن التوقع بالاستهلاك الفعلي."))
    st.info(t("This is a historical replay because the package contains 2017 weather and load data. A genuine future forecast needs 24-hour weather forecasts at 10-minute resolution plus recent load history.","ده اختبار إعادة تشغيل تاريخي لأن الحزمة تحتوي على بيانات 2017. التنبؤ الحقيقي للمستقبل يحتاج توقعات طقس لـ24 ساعة كل 10 دقائق بالإضافة إلى سجل الحمل الحديث."))
    date_min=hist.index.min().date()+timedelta(days=9)
    date_max=hist.index.max().date()
    col1,col2,col3=st.columns([1,1,1])
    with col1: chosen_date=st.date_input(t("Replay date","تاريخ الاختبار"),value=date(2017,12,15),min_value=date_min,max_value=date_max)
    with col2: target=st.selectbox(t("Load series","سلسلة الحمل"),TARGETS,index=3,format_func=lambda x: target_labels[x])
    with col3: show_actual=st.toggle(t("Show actual values","إظهار القيم الفعلية"),value=True)
    times=hist.loc[str(chosen_date)].index
    if len(times):
        pred=forecast(hist,times,bundle,interval=True)
        actual=hist.loc[times,target]
        dfp=pd.DataFrame({"Forecast":pred[target],"Lower band":pred[target+"_low"],"Upper band":pred[target+"_high"]},index=times)
        if show_actual: dfp["Actual"] = actual
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=dfp.index,y=dfp["Upper band"],mode="lines",line=dict(width=0),showlegend=False,name="Upper"))
        fig.add_trace(go.Scatter(x=dfp.index,y=dfp["Lower band"],mode="lines",line=dict(width=0),fill="tonexty",fillcolor="rgba(22,168,154,.16)",name=t("90% calibration band","نطاق معايرة 90%")))
        fig.add_trace(go.Scatter(x=dfp.index,y=dfp["Forecast"],name=t("Model forecast","توقع النموذج"),line=dict(color="#13A99A",width=2.6)))
        if show_actual: fig.add_trace(go.Scatter(x=dfp.index,y=dfp["Actual"],name=t("Observed load","الحمل الفعلي"),line=dict(color="#E8A23A",width=1.7)))
        fig.update_layout(title=f'{target_labels[target]} · {chosen_date}',yaxis_title=t("Load","الحمل"))
        plot_layout(fig,470)
        err=actual-pred[target]
        c=st.columns(4)
        with c[0]: st.metric(t("Day MAE","MAE اليوم"),f"{np.mean(np.abs(err)):,.0f}")
        with c[1]: st.metric(t("Day RMSE","RMSE اليوم"),f"{np.sqrt(np.mean(err**2)):,.0f}")
        with c[2]: st.metric(t("Day MAPE","MAPE اليوم"),f"{np.mean(np.abs(err)/actual.clip(lower=1))*100:.2f}%")
        with c[3]: st.metric(t("Within band","داخل النطاق"),f"{((actual>=pred[target+'_low'])&(actual<=pred[target+'_high'])).mean()*100:.1f}%")
        st.download_button(t("Download this day's forecast CSV","تحميل توقعات اليوم CSV"),dfp.to_csv(index_label="datetime").encode("utf-8"),file_name=f"forecast_{target}_{chosen_date}.csv",mime="text/csv")
    else: st.warning(t("No rows available for this date.","لا توجد بيانات لهذا التاريخ."))

# ---------------------- Model Observatory ----------------------
elif page_en == "Model Observatory":
    hero("MODEL OBSERVATORY",t("Trust the score. Understand the caveats.","افهم الأداء وحدوده."),t("Transparent, time-aware evaluation from the December 2017 holdout. Metrics are reported from the project metadata rather than invented in the UI.","تقييم زمني واضح من اختبار ديسمبر 2017. المقاييس مأخوذة من نتائج المشروع المحفوظة وليست أرقامًا افتراضية."))
    dfm=pd.DataFrame([{"Series":target_labels[k],"MAPE (%)":v["MAPE_%"],"MAE":v["MAE"],"RMSE":v["RMSE"],"R²":v["R2"],"MAE improvement vs naive (%)":v["MAE improvement %"]} for k,v in metrics.items()])
    st.dataframe(dfm,hide_index=True,use_container_width=True)
    a,b=st.columns(2)
    with a:
        fig=px.bar(dfm,x="Series",y="MAPE (%)",text="MAPE (%)",color="Series",color_discrete_sequence=["#13A99A","#5B8DEF","#E7A63B","#A58AE8"])
        fig.update_layout(title=t("Error by load series (lower is better)","الخطأ حسب سلسلة الحمل — الأقل أفضل"),showlegend=False)
        plot_layout(fig,360)
    with b:
        fig=px.bar(dfm,x="Series",y="R²",text="R²",color="Series",color_discrete_sequence=["#13A99A","#5B8DEF","#E7A63B","#A58AE8"])
        fig.update_layout(title=t("Explained variance score","درجة التباين المفسر"),showlegend=False,yaxis_range=[0,1])
        plot_layout(fig,360)
    st.markdown("### "+t("Ensemble recipe","تركيبة النماذج"))
    chosen=meta["chosen_models"]
    for k in TARGETS:
        st.write(f"**{target_labels[k]}:** " + " + ".join(chosen[k]))
    st.markdown(f'<div class="insight"><b>{t("Evaluation integrity","نزاهة التقييم")}</b><br><span class="small-muted">{t("The notebook uses a chronological split: training through October, validation in November, and a held-out December test. Production is retrained on all 2017 data, so use the backtest bundle for honest evaluation and the production bundle for deployment.","النوتبوك يستخدم تقسيمًا زمنيًا: تدريب حتى أكتوبر، تحقق في نوفمبر، واختبار منفصل في ديسمبر. نموذج الإنتاج أعيد تدريبه على عام 2017 بالكامل؛ لذلك نستخدم حزمة backtest للتقييم النزيه وحزمة production للنشر.")}</span></div>',unsafe_allow_html=True)
    st.caption(t("Intervals are empirical calibration bands based on saved historical errors, not formal guarantees under distribution shift.","نطاقات التنبؤ معايرة من أخطاء تاريخية محفوظة وليست ضمانًا إحصائيًا عند تغيّر توزيع البيانات."))


# ---------------------- Scenario Studio ----------------------
elif page_en == "Scenario Studio":
    hero("SCENARIO STUDIO",t("Change an assumption. See what moves.","غيّر افتراضًا وشاهد تأثيره."),t("A controlled sensitivity experiment using actual model input features. This is a replay-based what-if, not a causal claim about weather or behavior.","تجربة حساسية مضبوطة تستخدم خصائص فعلية تدخل للنموذج. هي محاكاة افتراضية وليست إثباتًا لعلاقة سببية."))
    st.info(t("Scenario adjustments are applied to weather values in the selected historical day, while all other inputs stay fixed. The output is only as realistic as the assumed weather scenario.","يتم تعديل الطقس في يوم تاريخي مختار مع تثبيت بقية المدخلات. واقعية النتيجة تعتمد على واقعية افتراضات الطقس."))
    d=st.date_input(t("Baseline day","اليوم الأساسي"),value=date(2017,12,15),min_value=hist.index.min().date()+timedelta(days=9),max_value=hist.index.max().date())
    s1,s2,s3=st.columns(3)
    with s1: temp_delta=st.slider(t("Temperature shift (°C)","تغير الحرارة (°م)"),-8.0,8.0,0.0,0.5)
    with s2: humidity_delta=st.slider(t("Humidity shift (percentage points)","تغير الرطوبة (نقطة مئوية)"),-20,20,0,2)
    with s3: wind_scale=st.slider(t("Wind multiplier","معامل الرياح"),0.5,2.0,1.0,0.1)
    ts=hist.loc[str(d)].index
    if len(ts):
        baseline=forecast(hist,ts,bundle,interval=False)
        scenario_hist=hist.copy()
        scenario_hist.loc[ts,"temp"] = (scenario_hist.loc[ts,"temp"]+temp_delta).clip(-5,50)
        scenario_hist.loc[ts,"humidity"] = (scenario_hist.loc[ts,"humidity"]+humidity_delta).clip(0,100)
        scenario_hist.loc[ts,"wind"] = (scenario_hist.loc[ts,"wind"]*wind_scale).clip(lower=0)
        alt=forecast(scenario_hist,ts,bundle,interval=False)
        target=st.selectbox(t("Compare series","السلسلة للمقارنة"),TARGETS,index=3,format_func=lambda x: target_labels[x])
        delta=alt[target]-baseline[target]
        cc=st.columns(3)
        with cc[0]: st.metric(t("Baseline mean","متوسط الأساس"),f"{baseline[target].mean():,.0f}")
        with cc[1]: st.metric(t("Scenario mean","متوسط السيناريو"),f"{alt[target].mean():,.0f}",f"{delta.mean():+,.0f}")
        with cc[2]: st.metric(t("Average change","متوسط التغير"),f"{delta.mean()/max(baseline[target].mean(),1)*100:+.2f}%")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=ts,y=baseline[target],name=t("Baseline","الأساس"),line=dict(color="#8193A5",width=2)))
        fig.add_trace(go.Scatter(x=ts,y=alt[target],name=t("What-if scenario","سيناريو افتراضي"),line=dict(color="#13A99A",width=2.5)))
        fig.update_layout(title=t("Model response to changed weather inputs","استجابة النموذج لتغير الطقس"),yaxis_title=t("Predicted load","الحمل المتوقع"))
        plot_layout(fig,400)
        st.caption(t("Because the model uses rolling weather features, a scenario can influence nearby rows too. Treat this as model sensitivity, not a real-world causal forecast.","لأن النموذج يستخدم خصائص طقس متحركة، قد يؤثر السيناريو على الصفوف المجاورة أيضًا. اعتبره تحليل حساسية للنموذج وليس توقعًا سببيًا مؤكدًا."))

# ---------------------- AI Analyst / Gemini Chatbot ----------------------
elif page_en == "AI Analyst":
    hero("WATT AI ANALYST", t("Ask the evidence. Understand the forecast.", "اسأل الأدلة وافهم التوقع."),
         t("Gemini assistant grounded in saved model metrics and the latest forecast generated in this app. It must flag missing evidence instead of inventing answers.",
           "مساعد Gemini يستند إلى مقاييس الموديل والتوقعات التي أنشأها التطبيق، ويصرّح بنقص الأدلة بدل اختلاق الإجابات."))
    st.markdown(t("Configure `GEMINI_API_KEY` in `.streamlit/secrets.toml` or as an environment variable. Do not paste the key into this file or commit it to GitHub.",
                  "اضبط `GEMINI_API_KEY` داخل `.streamlit/secrets.toml` أو كمتغير بيئة. لا تضع المفتاح داخل هذا الملف أو ترفعه إلى GitHub."))
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    try:
        if not gemini_key:
            gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    future_ctx = st.session_state.get("watt_future_forecast")
    future_meta = st.session_state.get("watt_future_forecast_meta", {})
    project_context = {
        "project": "Tetouan City electricity load forecasting",
        "training_data_period": "2017 only",
        "forecast_horizon": "24 hours, 144 ten-minute steps",
        "honest_test_metrics_december_2017": metrics,
        "selected_models": meta.get("chosen_models", {}),
        "feature_importance_summary_from_project_analysis": {
            "past_load_pct": 50.2, "calendar_pct": 43.1, "weather_pct": 6.7
        },
        "limitations": [
            "Only one year of source history (2017)",
            "Zone 3 was the least reliable zone in December test",
            "Holidays were a known weak spot",
            "Future forecast requires continuous recent load history and future weather at every target timestamp",
            "Saved intervals are empirical hour-specific calibration bands, not guaranteed confidence intervals"
        ],
        "latest_generated_forecast_metadata": future_meta,
        "latest_generated_forecast_sample": (
            future_ctx.head(12).reset_index().to_dict(orient="records") if future_ctx is not None else None
        )
    }
    if future_ctx is None:
        st.info(t("No future forecast has been generated in this session yet. You can still ask about the documented model and evaluation metrics.",
                  "لم يتم إنشاء توقع مستقبلي في هذه الجلسة بعد. يمكنك السؤال عن الموديل ومقاييس التقييم المحفوظة."))
    if "watt_chat_history" not in st.session_state:
        st.session_state["watt_chat_history"] = []
    for msg in st.session_state["watt_chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(t("Ask about the forecast, errors, zones, assumptions…",
                             "اسأل عن التوقع أو الأخطاء أو المناطق أو الافتراضات…"))
    if prompt:
        st.session_state["watt_chat_history"].append({"role":"user", "content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        if not gemini_key:
            answer = t("Gemini is not configured yet. Add GEMINI_API_KEY to `.streamlit/secrets.toml` or your environment, then restart Streamlit.",
                       "لم يتم إعداد Gemini بعد. أضف GEMINI_API_KEY إلى `.streamlit/secrets.toml` أو متغيرات البيئة ثم أعد تشغيل Streamlit.")
        else:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                recent_chat = st.session_state["watt_chat_history"][-10:]
                contents = (
                    "You are WATT, a careful electricity-load forecasting assistant. "
                    "Answer in the user's language. Use only the PROJECT EVIDENCE JSON below. "
                    "If evidence is insufficient, say so clearly and list the missing data. "
                    "Never invent live readings, forecasts, sources, model results, or causal claims. "
                    "Distinguish historical test metrics from future forecast outputs. "
                    "The latest forecast sample is only a sample, not necessarily all 144 rows.\\n\\n"
                    "PROJECT EVIDENCE JSON:\\n" + json.dumps(project_context, ensure_ascii=False, default=str) +
                    "\\n\\nCONVERSATION:\\n" +
                    "\\n".join(f'{m["role"].upper()}: {m["content"]}' for m in recent_chat)
                )
                with st.spinner(t("Gemini is checking the project evidence…", "Gemini يراجع أدلة المشروع…")):
                    response = client.models.generate_content(
                        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                        contents=contents
                    )
                answer = response.text or t("Gemini returned an empty response.", "رد Gemini فارغ.")
            except Exception as e:
                answer = t("Gemini request failed. Check your API key, model access, package installation, and network. Details: ",
                           "فشل طلب Gemini. تحقق من المفتاح وصلاحية الموديل وتثبيت المكتبة والاتصال. التفاصيل: ") + str(e)
        st.session_state["watt_chat_history"].append({"role":"assistant", "content":answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    if st.button(t("Clear chat", "مسح المحادثة"), key="clear_watt_chat"):
        st.session_state["watt_chat_history"] = []
        st.rerun()
    st.markdown("### " + t("Evidence cards", "بطاقات الأدلة"))
    st.json({"December test metrics": metrics, "Models": meta.get("chosen_models", {}),
             "Future forecast available in session": future_ctx is not None,
             "Known limitations": project_context["limitations"]})

# ---------------------- Export Center ----------------------
elif page_en == "Export Center":
    hero("EXPORT CENTER",t("Take the evidence with you.","خد الأدلة معاك."),t("Export forecasts, flagged anomalies, and a compact PDF report for review or presentation.","صدّر التوقعات والتنبيهات وتقرير PDF مختصر للمراجعة أو العرض."))
    target=st.selectbox(t("Report series","السلسلة في التقرير"),TARGETS,index=3,format_func=lambda x: target_labels[x])
    d=st.date_input(t("Report date","تاريخ التقرير"),value=date(2017,12,15),min_value=hist.index.min().date()+timedelta(days=9),max_value=hist.index.max().date())
    ts=hist.loc[str(d)].index
    if len(ts):
        pred=forecast(hist,ts,bundle,interval=True)
        actual=hist.loc[ts,target]
        report_df=pred[[target,target+"_low",target+"_high"]].copy()
        report_df["actual"]=actual
        report_df["residual"]=actual-pred[target]
        st.dataframe(report_df.head(12).round(2),use_container_width=True)
        st.download_button(t("Download forecast + actual CSV","تحميل التوقعات والقيم الفعلية CSV"),report_df.to_csv(index_label="datetime").encode(),file_name=f"watt_report_{target}_{d}.csv",mime="text/csv")
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            buffer=io.BytesIO()
            doc=SimpleDocTemplate(buffer,pagesize=A4,rightMargin=16*mm,leftMargin=16*mm,topMargin=15*mm,bottomMargin=15*mm)
            styles=getSampleStyleSheet()
            story=[Paragraph("WATT | Electricity Load Forecast Report",styles["Title"]),Spacer(1,6),
                   Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} · Series: {target_labels[target]} · Replay date: {d}",styles["Normal"]),Spacer(1,10),
                   Paragraph("Model reference (December 2017 holdout)",styles["Heading2"])]
            m=metrics[target]
            story.append(Paragraph(f"MAPE: {m['MAPE_%']:.2f}% · MAE: {m['MAE']:,.0f} · RMSE: {m['RMSE']:,.0f} · R²: {m['R2']:.3f}",styles["Normal"]))
            story.append(Spacer(1,8))
            story.append(Paragraph("Replay summary",styles["Heading2"]))
            story.append(Paragraph(f"Day MAE: {np.mean(np.abs(report_df.residual)):.1f} · Day RMSE: {np.sqrt(np.mean(report_df.residual**2)):.1f} · Points outside saved interval: {int(((actual<report_df[target+'_low'])|(actual>report_df[target+'_high'])).sum())}",styles["Normal"]))
            story.append(Spacer(1,8))
            story.append(Paragraph("Sample forecast rows",styles["Heading2"]))
            sample=report_df.iloc[::max(1,len(report_df)//12)].head(12).reset_index()
            table_data=[["Datetime","Forecast","Lower","Upper","Actual"]]+[[str(r["datetime"]),f"{r[target]:.0f}",f"{r[target+'_low']:.0f}",f"{r[target+'_high']:.0f}",f"{r['actual']:.0f}"] for _,r in sample.iterrows()]
            tab=Table(table_data,repeatRows=1,colWidths=[42*mm,26*mm,24*mm,24*mm,24*mm])
            tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#123448")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.25,colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),7),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EEF6F7")]),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
            story.append(tab); story.append(Spacer(1,10))
            story.append(Paragraph("Limitations: the source history covers 2017. This is a historical replay, not a live operational forecast. Empirical bands can under-cover when demand patterns shift.",styles["Normal"]))
            doc.build(story)
            st.download_button(t("Download PDF report","تحميل تقرير PDF"),buffer.getvalue(),file_name=f"WATT_{target}_{d}.pdf",mime="application/pdf",type="primary")
        except ImportError:
            st.warning(t("Install reportlab to enable PDF export: pip install reportlab","ثبّت reportlab لتفعيل تصدير PDF: pip install reportlab"))
