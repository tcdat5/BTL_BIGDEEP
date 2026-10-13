"""Dashboard phát hiện bất thường mạng.

    streamlit run dashboard/app.py                                   # nguồn replay (mặc định)
    streamlit run dashboard/app.py -- --source stream --stream-dir results/stream/default

replay: phát lại data/demo_test.parquet, chấm điểm bằng ONNX (onnxruntime) — không cần Kafka/Spark/PyTorch.
stream: đọc thư mục output của Spark Structured Streaming (streaming/stream_detect.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import config as C  # noqa: E402
from dashboard import view_model as vm  # noqa: E402

TICK_S = 1.0
ORDERS = {"Trộn ngẫu nhiên (seed 42)": "shuffle", "Theo thời gian gốc (orig_ts)": "orig_ts"}
SOURCES = {"replay": "Phát lại tập test (replay)", "stream": "Spark Streaming (stream)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--source", choices=sorted(SOURCES), default="replay")
    p.add_argument("--demo", default="data/demo_test.parquet")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--stream-dir", default=None, help="mặc định streaming.output_dir")
    a, _ = p.parse_known_args()
    cfg = C.load_config(a.config)
    a.scaler = C.get(cfg, "models.scaler", "models/scaler.json")
    a.stream_dir = a.stream_dir or C.get(cfg, "streaming.output_dir")
    a.max_files = C.get(cfg, "dashboard.max_files", 2000)
    a.refresh_s = float(C.get(cfg, "dashboard.refresh_s", 3))
    return a


@st.cache_data(show_spinner=False)
def load_demo(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def get_scorer(onnx_path: str, scaler_path: str):
    from streaming.scorer import OnnxScorer
    return OnnxScorer(onnx_path, scaler_path)


def wide(fn) -> dict:
    """Tham số "rộng hết khung": Streamlit >= 1.46 dùng width="stretch" (use_container_width bị deprecate),
    bản cũ (1.39 trên cụm) dùng use_container_width=True."""
    ver = tuple(int(x) for x in st.__version__.split(".")[:2])
    return {"width": "stretch"} if ver >= (1, 46) else {"use_container_width": True}


def chart_theme(chart):
    return chart.configure_axis(labelFontSize=14, titleFontSize=15).configure_legend(
        labelFontSize=14, titleFontSize=14).configure_title(fontSize=16)


def now_ms() -> int:
    return int(time.time() * 1000)


# ------------------------------------------------------------------ sidebar
args = parse_args()
st.set_page_config(page_title="Phát hiện bất thường mạng", layout="wide")
st.markdown("<style>html, body, [class*='st-'] {font-size: 18px;}</style>", unsafe_allow_html=True)
st.title("Phát hiện bất thường mạng thời gian thực — Deep Autoencoder")

ss = st.session_state
source = st.sidebar.radio("Nguồn dữ liệu", list(SOURCES), format_func=SOURCES.get,
                          index=list(SOURCES).index(args.source), key="source")
models = vm.list_models(args.models_dir)
if not models:
    st.error("Chưa có model trong `%s` (cần `<kind>.onnx` + `threshold_<kind>.json`). "
             "Giải nén kaggle_outputs.zip vào thư mục repo." % args.models_dir)
    st.stop()
model = st.sidebar.selectbox("Model", models, format_func=lambda m: m["label"], key="model")
scaler_path = str(Path(args.models_dir) / "scaler.json")
if not Path(scaler_path).exists():
    scaler_path = args.scaler

valscores = vm.load_valscores(args.results_dir, model["name"])
modes = ["Ngưỡng đã lưu"] + (["Theo percentile val Benign"] if valscores is not None else []) + ["Tự chỉnh"]
mode = st.sidebar.radio("Cách đặt ngưỡng", modes, key="thr_mode")
thr = model["threshold"]
if mode == "Theo percentile val Benign":
    pct = st.sidebar.slider("Percentile score trên val Benign", 90.0, 99.99,
                            float(model["percentile"] or 99.0), 0.01, key="thr_pct")
    thr = vm.threshold_from_percentile(valscores, pct)
elif mode == "Tự chỉnh":
    if model["kind"] == "mlp":
        thr = st.sidebar.slider("Ngưỡng xác suất", 0.01, 0.99, float(model["threshold"]), 0.01, key="thr_prob")
    else:
        base = float(np.log10(max(model["threshold"], 1e-12)))
        v = st.sidebar.slider("Ngưỡng (log10)", round(base - 2, 2), round(base + 2, 2), round(base, 2), 0.01,
                              key="thr_log")
        thr = float(10 ** v)
st.sidebar.markdown("**Ngưỡng đang dùng:** `%.6g`  \n(ngưỡng đã lưu: `%.6g`)" % (thr, model["threshold"]))

# ------------------------------------------------------------------ data
frame = pd.DataFrame(columns=vm.FRAME_COLUMNS)
running = False
if source == "replay":
    if not Path(args.demo).exists():
        st.error("Không tìm thấy `%s` — giải nén kaggle_outputs.zip (có data/demo_test.parquet)." % args.demo)
        st.stop()
    demo = load_demo(args.demo)
    order_label = st.sidebar.radio("Thứ tự phát lại", list(ORDERS), key="order")
    speed = st.sidebar.slider("Tốc độ (flow/giây)", 10, 5000, 500, 10, key="speed")
    if "feed" not in ss or ss.feed.order_name != ORDERS[order_label] or len(ss.feed.df) != len(demo):
        ss.feed = vm.ReplayFeed(demo, order=ORDERS[order_label])
        ss.cache = vm.ScoreCache(len(demo))
        ss.running = False
    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("▶ Start", key="start"):
        ss.running, ss.last_tick = True, time.time()
    if c2.button("⏸ Pause", key="pause"):
        ss.running = False
    if c3.button("↺ Reset", key="reset"):
        ss.feed.reset()
        ss.running = False
    if st.sidebar.button("Phát nhanh 1.000 flow", key="burst"):
        ss.feed.emit(1000, now_ms())
    if ss.get("running"):
        now = time.time()
        n = int(speed * min(now - ss.get("last_tick", now), 5.0))
        ss.feed.emit(n, now_ms())
        ss.last_tick = now
        if ss.feed.done:
            ss.running = False
    running = bool(ss.get("running"))
    scorer = get_scorer(model["onnx"], scaler_path)
    scores = ss.cache.get(model["onnx"], scorer, ss.feed.df, ss.feed.emitted())
    frame = vm.replay_frame(ss.feed, scores)
    st.sidebar.progress(ss.feed.pos / max(len(demo), 1),
                        text="Đã phát %s / %s flow" % (format(ss.feed.pos, ","), format(len(demo), ",")))
else:
    from streaming.output_io import load_output
    frame = vm.stream_frame(load_output(args.stream_dir, max_files=args.max_files))
    st.sidebar.caption("Đọc `%s` (tự làm mới mỗi %gs)" % (args.stream_dir, args.refresh_s))

# ------------------------------------------------------------------ tabs
tab_mon, tab_eval, tab_res = st.tabs(["Giám sát", "Đánh giá", "Kết quả nghiên cứu"])

with tab_mon:
    if len(frame) == 0:
        st.info("Chưa có flow nào được xử lý. Bấm **Start** (hoặc **Phát nhanh**) ở thanh bên.")
    else:
        k = vm.kpis(frame, thr)
        c = st.columns(4)
        c[0].metric("Flow đã xử lý", format(k["total"], ","))
        c[1].metric("Cảnh báo", format(k["alerts"], ","))
        c[2].metric("Tỷ lệ cảnh báo", "%.2f%%" % (100 * k["alert_rate"]))
        c[3].metric("Throughput (10 s gần nhất)", "%.0f flow/s" % k["throughput"])
        ps = vm.per_second(frame, thr).melt("time", var_name="loại", value_name="số lượng")
        ps["loại"] = ps["loại"].map({"flows": "flow", "alerts": "cảnh báo"})
        st.subheader("Số flow và cảnh báo theo thời gian")
        st.altair_chart(chart_theme(alt.Chart(ps).mark_line(point=True).encode(
            x=alt.X("time:T", title="thời gian"), y=alt.Y("số lượng:Q"), color="loại:N")), **wide(st.altair_chart))
        st.subheader("Score theo thời gian (3.000 flow gần nhất) — đường đỏ là ngưỡng")
        tl = vm.score_timeline(frame)
        tl["cảnh báo"] = tl["score"] >= thr
        log_y = model["kind"] != "mlp"
        if log_y:
            tl["score"] = tl["score"].clip(lower=1e-6)
        pts = alt.Chart(tl).mark_circle(size=18).encode(
            x=alt.X("seq:Q", title="flow (cũ → mới)"),
            y=alt.Y("score:Q", title="score", scale=alt.Scale(type="log" if log_y else "linear")),
            color=alt.Color("cảnh báo:N", scale=alt.Scale(domain=[False, True], range=["#9ca3af", "#dc2626"])))
        rule = alt.Chart(pd.DataFrame({"thr": [thr]})).mark_rule(color="#dc2626", strokeDash=[6, 4], size=2) \
            .encode(y="thr:Q")
        st.altair_chart(chart_theme(pts + rule), **wide(st.altair_chart))
        st.subheader("Flow đáng ngờ nhất (score cao nhất, 5.000 flow gần nhất)")
        st.dataframe(vm.top_suspicious(frame), hide_index=True, **wide(st.dataframe))

with tab_eval:
    st.caption("Dùng nhãn thật của các flow đã phát — hệ thống thật không có nhãn. Kéo thanh trượt ngưỡng để xem thay đổi.")
    if len(frame) == 0:
        st.info("Chưa có flow nào.")
    else:
        ev = vm.evaluation(frame, thr)
        m = ev["metrics"]
        c = st.columns(4)
        c[0].metric("Precision", "%.4f" % m["precision"])
        c[1].metric("Recall", "%.4f" % m["recall"])
        c[2].metric("F1", "%.4f" % m["f1"])
        c[3].metric("FPR", "%.4f" % m["fpr"])
        left, right = st.columns([2, 3])
        left.subheader("Ma trận nhầm lẫn")
        left.dataframe(ev["confusion"], **wide(st.dataframe))
        right.subheader("Recall theo loại tấn công")
        if len(ev["per_attack"]):
            right.altair_chart(chart_theme(alt.Chart(ev["per_attack"]).mark_bar(color="#2563eb").encode(
                x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("attack:N", title=None, sort="-x"),
                tooltip=["attack", "recall", "n"])), **wide(st.altair_chart))
            right.dataframe(ev["per_attack"], hide_index=True, **wide(st.dataframe))
        else:
            right.info("Chưa có flow tấn công nào được phát.")

with tab_res:
    assets = vm.research_assets(args.reports_dir)
    st.subheader("So sánh 10 model (tập test chung)")
    st.markdown(assets["comparison_md"] or "_chưa có `reports/model_comparison.md`_")
    st.subheader("Hình kết quả")
    if not assets["figures"]:
        st.markdown("_chưa có hình trong `%s`_" % args.reports_dir)
    for path, cap in assets["figures"]:
        st.image(path, caption=cap)
    st.subheader("Khám phá dữ liệu (EDA) và không gian latent")
    if not assets["eda"]:
        st.markdown("_chưa có `reports/eda/eda_summary.md`_")
    for path, cap in assets["eda"]:
        st.image(path, caption=cap)

# ------------------------------------------------------------------ auto refresh
if source == "replay" and running:
    time.sleep(TICK_S)
    st.rerun()
elif source == "stream":
    time.sleep(args.refresh_s)
    st.rerun()
