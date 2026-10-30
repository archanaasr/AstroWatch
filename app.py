"""AstroWatch dashboard. Run: streamlit run app.py"""
import pandas as pd, plotly.express as px, streamlit as st
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from astrowatch_core import validate, detect

st.set_page_config(page_title="AstroWatch", layout="wide")
st.title("AstroWatch – Satellite Telemetry Anomaly Analytics")

up = st.sidebar.file_uploader("Dataset CSV (default: dataset.csv)", type="csv")
contam = st.sidebar.slider("Flag rate (top-score share)", 0.05, 0.40, 0.20, 0.01)

@st.cache_data
def load(src):
    return pd.read_csv(src)
try:
    raw = load(up if up else "dataset.csv")
    feats = validate(raw)
except FileNotFoundError:
    st.error("dataset.csv not found. Place it next to app.py or upload a CSV."); st.stop()
except Exception as e:
    st.error(f"Input problem: {e}"); st.stop()

@st.cache_data
def run(df, c): return detect(df, feats, contamination=c)
df, thr = run(raw, contam)
has_lbl = "anomaly" in df
page = st.sidebar.radio("Page", ["Mission Overview", "Telemetry Explorer", "Anomaly Analysis"])
chans = sorted(df.channel.unique())

if page == "Mission Overview":
    c = st.columns(5)
    c[0].metric("Segments", len(df)); c[1].metric("Channels", len(chans))
    c[2].metric("Flagged", int(df.flag.sum())); c[3].metric("Flagged %", f"{df.flag.mean():.1%}")
    c[4].metric("Missing cells", int(raw.isna().sum().sum()))
    st.caption(f"Duplicate rows: {int(raw.duplicated().sum())} | Numeric features: {len(feats)}")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x="channel", color=df.flag.map({0:"Not flagged",1:"Flagged"}), title="Segments per channel"), width="stretch")
    b.plotly_chart(px.histogram(df, x="score", nbins=50, title="Anomaly score distribution"), width="stretch")
    if has_lbl: st.info(f"Labelled anomalies in file: {int(df.anomaly.sum())} ({df.anomaly.mean():.1%})")

elif page == "Telemetry Explorer":
    sel = st.sidebar.multiselect("Channels", chans, default=chans)
    f = st.sidebar.selectbox("Feature", feats, index=feats.index("std") if "std" in feats else 0)
    lo, hi = st.sidebar.slider("Segment range", int(df.segment.min()), int(df.segment.max()), (int(df.segment.min()), int(df.segment.max())))
    d = df[df.channel.isin(sel) & df.segment.between(lo, hi)]
    if d.empty: st.warning("No records match the filters."); st.stop()
    st.plotly_chart(px.line(d, x="segment", y=f, color="channel", markers=True, log_y=(d[f] > 0).all() and f in ("std","var","duration")), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(d, x=f, color="channel", title="Distribution"), width="stretch")
    b.plotly_chart(px.box(d, x="channel", y=f, title="Box plot"), width="stretch")
    st.plotly_chart(px.imshow(d[feats].corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heatmap"), width="stretch")

else:
    sel = st.sidebar.multiselect("Channels", chans, default=chans)
    status = st.sidebar.radio("Status", ["All", "Flagged only", "Not flagged"])
    d = df[df.channel.isin(sel)]
    if status == "Flagged only": d = d[d.flag == 1]
    elif status == "Not flagged": d = d[d.flag == 0]
    if d.empty: st.warning("No records match the filters."); st.stop()
    st.plotly_chart(px.scatter(d, x="segment", y="score", color=d.flag.map({0:"Not flagged",1:"Flagged"}), symbol="channel", hover_data=feats[:6]).add_hline(y=thr, line_dash="dash"), width="stretch")
    f = st.selectbox("Feature to inspect", feats, index=feats.index("std") if "std" in feats else 0)
    st.plotly_chart(px.scatter(d, x="segment", y=f, color=d.flag.map({0:"Not flagged",1:"Flagged"}), log_y=(d[f] > 0).all() and f in ("std","var","duration")), width="stretch")
    st.dataframe(d.sort_values("score", ascending=False).head(200))
    st.caption("Flags are algorithmic candidates, not confirmed spacecraft faults.")
    if has_lbl and "train" in df:
        t = df[df.train == 0]
        p, r, f1, _ = precision_recall_fscore_support(t.anomaly, t.flag, average="binary", zero_division=0)
        st.subheader("Agreement with labels (official test split)")
        st.write(f"Precision {p:.3f} | Recall {r:.3f} | F1 {f1:.3f}")
        st.write(pd.DataFrame(confusion_matrix(t.anomaly, t.flag), index=["True nominal","True anomaly"], columns=["Pred nominal","Pred anomaly"]))
