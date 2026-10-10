# -*- coding: utf-8 -*-
"""
app.py - Em xây dựng giao diện dự đoán nguy cơ tiểu đường bằng Streamlit (bước 7 của đề)
Chạy:  streamlit run app.py
Sinh viên: Hoàng Tiến Minh - Lớp CNTT 3.2 - MSSV 25ACN046
Môn Trí tuệ nhân tạo - Giảng viên Thu Hà - ĐH Kinh doanh và Công nghệ Hà Nội
"""
import json
import os
import subprocess
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import ml_utils as U

ROOT = os.path.dirname(os.path.abspath(__file__))
# Tương thích nhiều phiên bản Streamlit: bản mới dùng width="stretch", bản cũ dùng use_container_width
try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _v = (0, 0)
WIDE = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}
P = lambda *a: os.path.join(ROOT, *a)
BLUE, ORANGE, AMBER, GRAY, INK = "#2a78d6", "#eb6834", "#c98500", "#8a8985", "#1f2937"

st.set_page_config(page_title="Dự đoán nguy cơ tiểu đường", page_icon="🩺", layout="wide")

# Logo trường (nhúng base64 để hiển thị trong HTML). Không có file thì bỏ qua, ứng dụng vẫn chạy.
import base64
LOGO_PATH = P("assets", "logo_truong.jpg")
try:
    with open(LOGO_PATH, "rb") as _f:
        LOGO_B64 = base64.b64encode(_f.read()).decode()
except OSError:
    LOGO_B64 = ""
LOGO_IMG = f'<img src="data:image/jpeg;base64,{LOGO_B64}" alt="Logo trường">' if LOGO_B64 else ""

# ----------------------------------------------------------------------
# GIAO DIỆN (CSS)
# ----------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1180px;}
.hero {background: linear-gradient(120deg,#16427a 0%,#2a78d6 55%,#1baf7a 120%); color:#fff;
       border-radius: 18px; padding: 26px 32px; margin-bottom: 18px;}
.hero h1 {margin:0; font-size: 1.9rem; font-weight: 800; letter-spacing:-.3px; color:#fff;}
.hero p {margin: 6px 0 0 0; opacity:.92; font-size: 1rem;}
.hero-row {display:flex; align-items:center; justify-content:space-between; gap:24px;}
.hero-logo {background:#fff; border-radius:16px; padding:6px 10px; flex:0 0 auto; box-shadow:0 2px 10px rgba(0,0,0,.18);}
.hero-logo img {height:92px; display:block;}
.student {display:flex; flex-wrap:wrap; gap:8px 22px; margin-top:14px; padding-top:12px;
          border-top:1px solid rgba(255,255,255,.3); font-size:.88rem;}
.student b {font-weight:700;}
.side-card {background:#f3f6fb; border-radius:14px; padding:12px 14px; font-size:.86rem; line-height:1.55; border-left:4px solid #16427a;}
.side-card img {width:100%; background:#fff; border-radius:10px; margin-bottom:8px;}
.footer {text-align:center; color:#5b6472; font-size:.82rem; margin-top:28px; padding-top:14px; border-top:1px solid #e3e8f0;}
.chip {display:inline-block; background:rgba(255,255,255,.18); border-radius:999px; padding:3px 12px;
       font-size:.82rem; margin: 10px 8px 0 0;}
.card {background:#fff; border:1px solid #e3e8f0; border-radius:16px; padding:18px 22px; box-shadow:0 1px 3px rgba(16,24,40,.05);}
.kpi {background:#f3f6fb; border-radius:14px; padding:14px 16px; text-align:center;}
.kpi .v {font-size:1.7rem; font-weight:800; color:#16427a;}
.kpi .l {font-size:.82rem; color:#5b6472; margin-top:2px;}
.risk {border-radius:18px; padding:22px 26px; color:#fff;}
.risk .t {font-size:.95rem; opacity:.95; letter-spacing:.4px; font-weight:600;}
.risk .p {font-size:3.1rem; font-weight:800; line-height:1.05; margin:2px 0;}
.risk .m {font-size:1.12rem; font-weight:700;}
.bar {height:12px; border-radius:99px; background:rgba(255,255,255,.28); margin-top:14px; overflow:hidden;}
.bar > div {height:100%; background:#fff; border-radius:99px;}
.flag {display:flex; gap:10px; align-items:flex-start; padding:9px 12px; border-radius:10px; margin-bottom:7px; font-size:.93rem;}
.flag.ok {background:#eef6ff; border-left:4px solid #2a78d6;}
.flag.warn {background:#fff7e0; border-left:4px solid #c98500;}
.flag.high {background:#ffeee6; border-left:4px solid #eb6834;}
.small {color:#5b6472; font-size:.84rem;}
.disclaimer {background:#f3f6fb; border-radius:12px; padding:12px 16px; font-size:.86rem; color:#4b5563; margin-top:10px;}
div[data-testid="stTabs"] button {font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# NẠP MÔ HÌNH (tự huấn luyện nếu chưa có)
# ----------------------------------------------------------------------
MODEL_PATH = P("model", "model_tieu_duong.joblib")


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Chưa có mô hình - đang huấn luyện lần đầu (khoảng 1-3 phút)..."):
            subprocess.run([sys.executable, P("train_model.py")], cwd=ROOT, check=True)
    try:
        art = joblib.load(MODEL_PATH)
    except Exception:  # mô hình lưu bằng phiên bản scikit-learn khác -> huấn luyện lại trên máy này
        with st.spinner("Phiên bản thư viện khác nhau - đang huấn luyện lại mô hình..."):
            subprocess.run([sys.executable, P("train_model.py")], cwd=ROOT, check=True)
        art = joblib.load(MODEL_PATH)
    with open(P("model", "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    return art["pipeline"], art["meta"], metrics


try:
    pipe, meta, M = load_artifacts()
except Exception as e:  # noqa
    st.error("Không nạp được mô hình. Hãy chạy `python train_model.py` trước, rồi mở lại ứng dụng.")
    st.exception(e)
    st.stop()

FEATS, NUM_F = meta["features"], meta["num_features"]
FM = M["final_metrics"]

# ----------------------------------------------------------------------
# HÀM HỖ TRỢ
# ----------------------------------------------------------------------
def predict_proba(row: dict) -> float:
    return float(pipe.predict_proba(pd.DataFrame([row])[FEATS])[0, 1])


def explain(row: dict):
    """Giải thích cục bộ: thay từng đặc trưng bằng giá trị điển hình (trung vị / mode của tập Train)
    rồi xem xác suất thay đổi bao nhiêu. Dương = đặc trưng đẩy nguy cơ lên."""
    base = predict_proba(row)
    out = []
    for f in FEATS:
        r = dict(row)
        r[f] = np.nan  # Pipeline sẽ điền bằng trung vị / mode
        out.append((f, base - predict_proba(r)))
    return base, sorted(out, key=lambda x: -abs(x[1]))


def fmt(f, v):
    name, unit = U.LABELS[f]
    return f"{v}" if f == "GioiTinh" else f"{v:g} {unit}".strip()


def level_of(p):
    if p >= 0.65:
        return "high", "CÓ NGUY CƠ CAO", "#eb6834", "🔴"
    if p > 0.35:
        return "mid", "CẦN THEO DÕI THÊM", "#c98500", "🟠"
    return "low", "NGUY CƠ THẤP", "#2a78d6", "🔵"


ADVICE = {
    "high": ["Nên đi khám và làm lại xét nghiệm đường huyết lúc đói, HbA1C với bác sĩ nội tiết sớm nhất có thể.",
             "Hạn chế đồ ngọt, nước có đường, tinh bột tinh chế; tăng rau xanh và chất xơ.",
             "Vận động ít nhất 150 phút mỗi tuần (đi bộ nhanh, đạp xe), kiểm soát cân nặng.",
             "Theo dõi các dấu hiệu: khát nhiều, tiểu nhiều, sụt cân không rõ lý do, mệt mỏi, vết thương lâu lành."],
    "mid": ["Các chỉ số đang ở vùng giới hạn - nên xét nghiệm lại sau 1-3 tháng và hỏi ý kiến bác sĩ.",
            "Duy trì chế độ ăn cân đối, giảm đường và tinh bột tinh chế, tập thể dục đều đặn.",
            "Kiểm tra thêm HbA1C nếu chưa có kết quả."],
    "low": ["Kết quả hiện tại cho thấy nguy cơ thấp. Hãy duy trì lối sống lành mạnh.",
            "Kiểm tra sức khỏe định kỳ mỗi 6-12 tháng, đặc biệt nếu có người thân bị tiểu đường."],
}

PRESETS = {
    "Người khỏe mạnh": dict(Tuoi=35, GioiTinh="Nam", Glucose=5.0, Cholesterol=4.8, Triglycerid=1.2, Creatinine=80.0, Ure=5.0, AST=22.0, ALT=20.0),
    "Tiền tiểu đường": dict(Tuoi=52, GioiTinh="Nữ", Glucose=6.6, Cholesterol=5.9, Triglycerid=2.0, Creatinine=75.0, Ure=5.5, AST=30.0, ALT=33.0),
    "Nguy cơ cao": dict(Tuoi=62, GioiTinh="Nữ", Glucose=11.9, Cholesterol=5.5, Triglycerid=2.1, Creatinine=80.1, Ure=7.7, AST=61.0, ALT=29.4),
}
DEFAULT = PRESETS["Người khỏe mạnh"]
for k, v in DEFAULT.items():
    st.session_state.setdefault(k, v)


def set_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# ----------------------------------------------------------------------
# THANH BÊN
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
<div class="side-card">{LOGO_IMG}
<b>Đại học Kinh doanh và Công nghệ Hà Nội</b><br>
Môn: Trí tuệ nhân tạo<br>
SV: <b>Hoàng Tiến Minh</b><br>
MSSV: <b>25ACN046</b> · Lớp: <b>CNTT 3.2</b><br>
GVHD: <b>Thu Hà</b>
</div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 🩺 Thông tin mô hình")
    st.markdown(f"**Mô hình:** {M['best']}")
    st.markdown(f"**Số bệnh nhân (sau làm sạch):** {M['report']['so_dong_sau_lam_sach']:,}")
    st.markdown(f"**Train / Val / Test:** {M['split']['Train']['n']:,} / {M['split']['Validation']['n']:,} / {M['split']['Test']['n']:,}")
    st.markdown(f"**Đặc trưng dùng:** {len(FEATS)}")
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{FM['Recall']*100:.1f}%")
    c2.metric("F1-score", f"{FM['F1']*100:.1f}%")
    c1.metric("Accuracy", f"{FM['Accuracy']*100:.1f}%")
    c2.metric("ROC-AUC", f"{FM['ROC_AUC']:.3f}")
    st.caption("Các chỉ số trên tính trên tập Test (15% dữ liệu, mô hình chưa từng nhìn thấy).")
    st.divider()
    st.markdown("**Nạp ví dụ nhanh**")
    for nm in PRESETS:
        st.button(nm, on_click=set_preset, args=(nm,), **WIDE)

# ----------------------------------------------------------------------
# TIÊU ĐỀ
# ----------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <div class="hero-row">
    <div>
      <h1>🩺 Hệ thống AI dự đoán nguy cơ tiểu đường</h1>
      <p>Nhập thông số sức khỏe của người bệnh, hệ thống sẽ ước tính xác suất có nguy cơ tiểu đường, đưa ra cảnh báo và giải thích lý do.</p>
      <span class="chip">Mô hình: {M['best']}</span>
      <span class="chip">Recall {FM['Recall']*100:.1f}%</span>
      <span class="chip">F1 {FM['F1']*100:.1f}%</span>
      <span class="chip">{M['report']['so_dong_sau_lam_sach']:,} bệnh nhân</span>
    </div>
    <div class="hero-logo">{LOGO_IMG}</div>
  </div>
  <div class="student">
    <span>👤 Sinh viên: <b>Hoàng Tiến Minh</b></span>
    <span>🆔 MSSV: <b>25ACN046</b></span>
    <span>🎓 Lớp: <b>CNTT 3.2</b></span>
    <span>👩‍🏫 Giảng viên hướng dẫn: <b>Thu Hà</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🩺 Dự đoán", "📂 Dự đoán hàng loạt", "📊 Đánh giá mô hình", "🧹 Dữ liệu & tiền xử lý", "📘 Hướng dẫn"])

# ======================================================================
# TAB 1: DỰ ĐOÁN
# ======================================================================
with tab1:
    left, right = st.columns([5, 6], gap="large")
    R = meta["ranges"]
    with left:
        st.subheader("Thông tin người bệnh")
        a, b = st.columns(2)
        tuoi = a.number_input("Tuổi (năm)", 0, 110, step=1, key="Tuoi")
        gt = b.selectbox("Giới tính", ["Nam", "Nữ"], key="GioiTinh")
        glu = st.number_input("Glucose - đường huyết (mmol/L)", 1.0, 40.0, step=0.1, format="%.1f", key="Glucose",
                              help="Chỉ số quan trọng nhất. ≥ 7,0 mmol/L lúc đói đạt ngưỡng chẩn đoán tiểu đường.")
        a, b = st.columns(2)
        chol = a.number_input("Cholesterol (mmol/L)", 1.0, 20.0, step=0.1, format="%.1f", key="Cholesterol")
        tg = b.number_input("Triglycerid (mmol/L)", 0.2, 40.0, step=0.1, format="%.1f", key="Triglycerid")
        a, b = st.columns(2)
        cre = a.number_input("Creatinine (µmol/L)", 10.0, 1500.0, step=1.0, format="%.1f", key="Creatinine")
        ure = b.number_input("Ure (mmol/L)", 0.5, 70.0, step=0.1, format="%.1f", key="Ure")
        a, b = st.columns(2)
        ast = a.number_input("AST - GOT (U/L)", 1.0, 1000.0, step=1.0, format="%.1f", key="AST")
        alt = b.number_input("ALT - GPT (U/L)", 1.0, 1000.0, step=1.0, format="%.1f", key="ALT")

        with st.expander("Xét nghiệm bổ sung (không đưa vào mô hình, chỉ dùng cảnh báo tham khảo)"):
            st.caption("HbA1C, HDL, LDL, Acid Uric bị loại khỏi mô hình vì trong dữ liệu gần như toàn bộ đã bị điền sẵn một giá trị cố định "
                       "(xem tab Dữ liệu & tiền xử lý). Nếu có kết quả, bạn vẫn có thể nhập để nhận cảnh báo theo ngưỡng y khoa.")
            extra = {}
            for key, lo, hi, step, default in [("HbA1C", 3.0, 20.0, 0.1, 5.5), ("HDL", 0.2, 5.0, 0.1, 1.3),
                                                ("LDL", 0.5, 12.0, 0.1, 2.8), ("AcidUric", 50.0, 1200.0, 5.0, 350.0)]:
                c1, c2 = st.columns([1, 2])
                use = c1.checkbox(U.LABELS[key][0], key=f"use_{key}")
                if use:
                    extra[key] = c2.number_input(f"{U.LABELS[key][0]} ({U.LABELS[key][1]})", lo, hi, default, step, key=f"val_{key}", label_visibility="collapsed")
        go = st.button("🔍 Dự đoán nguy cơ", type="primary", **WIDE)

    row = {"Tuoi": tuoi, "GioiTinh": gt, "Glucose": glu, "Cholesterol": chol, "Triglycerid": tg,
           "Creatinine": cre, "Ure": ure, "AST": ast, "ALT": alt}
    row = {k: row[k] for k in FEATS}  # đúng thứ tự đặc trưng mô hình đã học

    with right:
        st.subheader("Kết quả dự đoán")
        if not go and "last" not in st.session_state:
            st.markdown('<div class="card"><b>Chưa có kết quả.</b><br><span class="small">Nhập thông tin ở bên trái hoặc chọn một ví dụ nhanh ở thanh bên, '
                        'rồi bấm <b>Dự đoán nguy cơ</b>.</span></div>', unsafe_allow_html=True)
        else:
            if go:
                st.session_state["last"] = dict(row=row, extra=extra)
            cur = st.session_state["last"]
            row_, extra_ = cur["row"], cur["extra"]
            p, contrib = explain(row_)
            lv, label, color, icon = level_of(p)
            st.markdown(f"""
            <div class="risk" style="background:{color};">
              <div class="t">{icon} XÁC SUẤT CÓ NGUY CƠ TIỂU ĐƯỜNG</div>
              <div class="p">{p*100:.1f}%</div>
              <div class="m">{label}</div>
              <div class="bar"><div style="width:{p*100:.1f}%"></div></div>
              <div class="small" style="color:#fff;opacity:.85;margin-top:8px;">Ngưỡng: ≤ 35% thấp · 35–65% cần theo dõi · ≥ 65% cao</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### Vì sao có kết quả này?")
            top = [c for c in contrib if abs(c[1]) >= 0.01][:3]
            if top:
                for f, d in top:
                    nm = U.LABELS[f][0]
                    verb = "làm tăng" if d > 0 else "làm giảm"
                    st.markdown(f"- **{nm} = {fmt(f, row_[f])}** {verb} xác suất khoảng **{abs(d)*100:.0f} điểm %** so với giá trị điển hình.")
            else:
                st.markdown("- Không có chỉ số nào làm thay đổi đáng kể xác suất so với giá trị điển hình của tập huấn luyện.")
            fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(contrib) + 0.9))
            names = [U.LABELS[f][0] for f, _ in contrib][::-1]
            vals = [d * 100 for _, d in contrib][::-1]
            ax.barh(names, vals, color=[ORANGE if v > 0 else BLUE for v in vals], height=0.62)
            ax.axvline(0, color=GRAY, lw=1)
            ax.set_xlabel("Ảnh hưởng đến xác suất (điểm %)  ← giảm | tăng →", fontsize=9)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            ax.tick_params(labelsize=9); ax.grid(axis="x", color="#e6e5e0", lw=.8); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.caption("Cách tính: thay lần lượt từng chỉ số bằng giá trị điển hình (trung vị của tập Train) rồi xem xác suất thay đổi bao nhiêu.")

    if (go or "last" in st.session_state):
        cur = st.session_state["last"]
        row_, extra_ = cur["row"], cur["extra"]
        p, _ = explain(row_)
        lv = level_of(p)[0]
        st.divider()
        c1, c2 = st.columns([6, 5], gap="large")
        with c1:
            st.markdown("#### ⚠️ Cảnh báo theo từng chỉ số xét nghiệm")
            allv = {k: v for k, v in row_.items() if k != "GioiTinh"}
            allv.update(extra_)
            flags = U.clinical_flags(allv, row_["GioiTinh"])
            order = {"high": 0, "warn": 1, "ok": 2}
            icons = {"high": "🔴", "warn": "🟠", "ok": "✅"}
            if lv == "low" and any(n == "Glucose" and l != "ok" for n, l, _ in flags):
                st.info("Mô hình học từ nhãn gần như quyết định bởi Glucose ≥ 7,0 nên mức tiền tiểu đường (5,6–6,9) vẫn được xếp nguy cơ thấp. "
                        "Hãy đọc cảnh báo từng chỉ số bên dưới.")
            for name, level, text in sorted(flags, key=lambda x: order[x[1]]):
                st.markdown(f'<div class="flag {level}"><div>{icons[level]}</div><div><b>{name}:</b> {text}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown("#### 💡 Khuyến nghị")
            for t in ADVICE[lv]:
                st.markdown(f"- {t}")
            st.markdown('<div class="disclaimer"><b>Lưu ý:</b> Đây là công cụ học tập, chỉ mang tính tham khảo, '
                        'không thay thế chẩn đoán của bác sĩ. Ngưỡng cảnh báo là ngưỡng phổ biến, mỗi phòng xét nghiệm có thể khác nhau.</div>', unsafe_allow_html=True)

# ======================================================================
# TAB 2: DỰ ĐOÁN HÀNG LOẠT
# ======================================================================
with tab2:
    st.subheader("Dự đoán cho nhiều bệnh nhân cùng lúc")
    st.write("Tải lên file CSV có các cột: " + ", ".join(f"`{c}`" for c in U.CATEGORICAL_COLS + U.NUMERIC_COLS) +
             ". Hệ thống tự làm sạch (đồng bộ giới tính, đổi đơn vị Glucose, loại giá trị lỗi) rồi dự đoán.")
    template = pd.DataFrame([PRESETS["Nguy cơ cao"], PRESETS["Người khỏe mạnh"]])
    st.download_button("⬇️ Tải file CSV mẫu", template.to_csv(index=False).encode("utf-8-sig"), "mau_du_doan.csv", "text/csv")
    up = st.file_uploader("Chọn file CSV", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up, encoding="utf-8-sig")
            miss = [c for c in ["Tuoi", "GioiTinh", "Glucose"] if c not in raw.columns]
            if miss:
                st.error("Thiếu cột bắt buộc: " + ", ".join(miss))
            else:
                orig_cols = list(raw.columns)
                for c in U.NUMERIC_COLS + U.CATEGORICAL_COLS:
                    if c not in raw.columns:
                        raw[c] = np.nan
                raw[U.TARGET] = np.nan
                cl, _ = U.clean_dataframe(raw, batch=True)
                pr = pipe.predict_proba(cl[FEATS])[:, 1]
                res = raw[orig_cols].copy()
                res.insert(0, "MucDo", [level_of(x)[1] for x in pr])
                res.insert(0, "XacSuat_NguyCo(%)", (pr * 100).round(1))
                k1, k2, k3 = st.columns(3)
                k1.metric("Số bệnh nhân", len(res))
                k2.metric("Nguy cơ cao", int((pr >= .65).sum()))
                k3.metric("Cần theo dõi", int(((pr > .35) & (pr < .65)).sum()))
                st.dataframe(res.sort_values("XacSuat_NguyCo(%)", ascending=False).reset_index(drop=True), **WIDE, height=300)
                st.download_button("⬇️ Tải kết quả (CSV)", res.to_csv(index=False).encode("utf-8-sig"), "ket_qua_du_doan.csv", "text/csv")
        except Exception as e:  # noqa
            st.error("Không đọc được file. Hãy kiểm tra định dạng CSV.")
            st.exception(e)

# ======================================================================
# TAB 3: ĐÁNH GIÁ MÔ HÌNH
# ======================================================================
with tab3:
    st.subheader("Kết quả đánh giá trên tập Test")
    cols = st.columns(5)
    for c, (k, lab) in zip(cols, [("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"), ("F1", "F1-score"), ("ROC_AUC", "ROC-AUC")]):
        v = FM[k]
        c.markdown(f'<div class="kpi"><div class="v">{v*100:.1f}%</div><div class="l">{lab}</div></div>' if k != "ROC_AUC"
                   else f'<div class="kpi"><div class="v">{v:.3f}</div><div class="l">{lab}</div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("##### So sánh các mô hình")
    rows = []
    for r in M["results"]:
        rows.append({"Mô hình": r["Mo_hinh"] + (" ⭐" if r["Mo_hinh"] == M["best"] else ""),
                     "F1 (Val)": r["Validation"]["F1"], "Recall (Val)": r["Validation"]["Recall"],
                     "Accuracy (Test)": r["Test"]["Accuracy"], "Precision (Test)": r["Test"]["Precision"],
                     "Recall (Test)": r["Test"]["Recall"], "F1 (Test)": r["Test"]["F1"], "ROC-AUC (Test)": r["Test"]["ROC_AUC"],
                     "F1 CV 5-fold": f"{r['CV_F1_mean']:.3f} ± {r['CV_F1_std']:.3f}"})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ["F1 (Val)", "Recall (Val)", "Accuracy (Test)", "Precision (Test)", "Recall (Test)", "F1 (Test)", "ROC-AUC (Test)"]}),
                 **WIDE, hide_index=True)
    st.caption("⭐ = mô hình được chọn: F1 cao nhất trên tập Validation (hòa thì xét Recall, rồi ROC-AUC). Tập Test chỉ dùng để báo cáo cuối.")
    a, b = st.columns(2)
    a.image(P("figures", "06_so_sanh_mo_hinh.png"), **WIDE)
    b.image(P("figures", "08_roc.png"), **WIDE)
    a, b = st.columns(2)
    a.image(P("figures", "07_ma_tran_nham_lan.png"), **WIDE)
    b.image(P("figures", "09_do_quan_trong.png"), **WIDE)
    rule, ab = M["rule_glucose7"], M["ablation_no_glucose"]
    st.markdown(f"""
    <div class="card"><b>Nhận xét quan trọng</b><br><span class="small">
    • Luật đơn giản <i>"Glucose ≥ 7,0 thì có nguy cơ"</i> đạt Accuracy {rule['Accuracy']*100:.1f}% và F1 {rule['F1']*100:.1f}% trên tập Test, gần bằng mô hình học máy. Nhãn trong dữ liệu gần như được quyết định bởi Glucose.<br>
    • Khi bỏ Glucose, cùng mô hình chỉ còn F1 {ab['F1']*100:.1f}% và ROC-AUC {ab['ROC_AUC']:.2f}: các chỉ số còn lại dự đoán yếu.<br>
    • Vì vậy kết quả cao không có nghĩa mô hình "thông minh hơn" bác sĩ; điểm mạnh của hệ thống là cho xác suất, giải thích và cảnh báo nhiều chỉ số cùng lúc.</span></div>
    """, unsafe_allow_html=True)
    st.markdown("##### Ảnh hưởng của ngưỡng quyết định")
    st.dataframe(pd.DataFrame(M["thresholds"]).rename(columns={"nguong": "Ngưỡng"}).style.format({"Ngưỡng": "{:.2f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                 **WIDE, hide_index=True)
    st.caption("FN = bỏ sót người có nguy cơ (nguy hiểm nhất khi sàng lọc); FP = báo động nhầm. Hạ ngưỡng thì Recall tăng, FP tăng.")

# ======================================================================
# TAB 4: DỮ LIỆU & TIỀN XỬ LÝ
# ======================================================================
with tab4:
    rep = M["report"]
    st.subheader("Tiền xử lý dữ liệu")
    k = st.columns(4)
    for c, (v, l) in zip(k, [(rep["so_dong_ban_dau"], "Dòng ban đầu"), (rep["so_dong_sau_lam_sach"], "Dòng sau làm sạch"),
                              (len(rep["cot_bi_loai"]), "Cột bị loại (bị điền sẵn)"), (len(meta["features"]), "Đặc trưng dùng cho mô hình")]):
        c.markdown(f'<div class="kpi"><div class="v">{v:,}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
    st.write("")
    steps = pd.DataFrame([
        ["Dữ liệu thiếu", f"{sum(rep['thieu_sau_lam_sach'].values())} ô thiếu sau làm sạch", "Điền trung vị (số) / mode (giới tính), học trên tập Train"],
        ["Đồng bộ", f"{rep['gioi_tinh_chua_dong_nhat']} giới tính Male/Female; {rep['nhan_chua_dong_nhat']} nhãn dạng 1/0 hoặc chữ", "Đưa về Nam/Nữ và 0/1"],
        ["Dữ liệu lỗi", f"{sum(rep['gia_tri_loi_chuyen_NaN'].values())} giá trị ngoài khoảng sinh học; {rep['glucose_doi_don_vi']} Glucose theo mg/dL", "Chuyển thành thiếu rồi điền; Glucose mg/dL chia 18,016"],
        ["Dòng trùng", f"{rep['dong_trung']} dòng", "Xóa nếu có"],
        ["Ngoại lai", "Men gan, Creatinine, Triglycerid... có giá trị rất cao", "Giới hạn (cap) theo phân vị 1%–99%, giữ lại bệnh nhân"],
        ["Mã hóa", "Giới tính", "One-Hot (nhị phân 0/1)"],
        ["Chuẩn hóa", "Các cột số", "StandardScaler (trung bình 0, độ lệch chuẩn 1)"],
        ["Chia dữ liệu", f"{M['split']['Train']['n']} / {M['split']['Validation']['n']} / {M['split']['Test']['n']}", "70% Train, 15% Validation, 15% Test, giữ tỷ lệ nhãn (stratify)"],
    ], columns=["Bước", "Phát hiện", "Cách xử lý"])
    st.dataframe(steps, **WIDE, hide_index=True)
    st.info("Phát hiện đáng chú ý: " + ", ".join(rep["cot_bi_loai"]) + f" có hơn {int(rep['nguong_loai_cot']*100)}% dòng mang cùng một giá trị (đã bị điền sẵn từ trước), "
            "gần như không chứa thông tin nên bị loại khỏi mô hình.")
    a, b = st.columns(2)
    a.image(P("figures", "01_phan_bo_nhan.png"), **WIDE)
    b.image(P("figures", "02_glucose_theo_nhan.png"), **WIDE)
    a, b = st.columns(2)
    a.image(P("figures", "03_gia_tri_dien_san.png"), **WIDE)
    b.image(P("figures", "05_tuong_quan.png"), **WIDE)
    st.image(P("figures", "04_ngoai_lai_truoc_sau.png"), **WIDE)
    clean_path = P("data", "CSDL_TieuDuong_DaLamSach.csv")
    if os.path.exists(clean_path):
        dfc = pd.read_csv(clean_path, encoding="utf-8-sig")
        st.markdown("##### Xem dữ liệu đã làm sạch")
        st.dataframe(dfc.head(200), **WIDE, height=260)
        st.download_button("⬇️ Tải dữ liệu đã làm sạch", dfc.to_csv(index=False).encode("utf-8-sig"), "CSDL_TieuDuong_DaLamSach.csv", "text/csv")

# ======================================================================
# TAB 5: HƯỚNG DẪN
# ======================================================================
with tab5:
    st.subheader("Hướng dẫn sử dụng nhanh")
    st.markdown("""
1. **Nhập thông tin** ở tab *Dự đoán*: tuổi, giới tính, Glucose và các chỉ số xét nghiệm. Có thể chọn *ví dụ nhanh* ở thanh bên.
2. Bấm **Dự đoán nguy cơ**. Hệ thống hiển thị xác suất, mức cảnh báo (thấp / cần theo dõi / cao), lý do và khuyến nghị.
3. Xem **cảnh báo từng chỉ số** để biết chỉ số nào bất thường theo ngưỡng y khoa tham khảo.
4. Có nhiều bệnh nhân? Dùng tab **Dự đoán hàng loạt**, tải file CSV mẫu, điền dữ liệu rồi tải kết quả về.
5. Tab **Đánh giá mô hình** và **Dữ liệu & tiền xử lý** trình bày quy trình và kết quả học máy.

**Cách đọc mức cảnh báo:** xác suất ≥ 65% là *có nguy cơ cao*; từ 35% đến 65% là *cần theo dõi thêm*; ≤ 35% là *nguy cơ thấp*.
Ứng dụng chỉ mang tính tham khảo và học tập, không thay thế chẩn đoán của bác sĩ.
    """)

# ----------------------------------------------------------------------
# CHÂN TRANG
# ----------------------------------------------------------------------
st.markdown("""
<div class="footer">
Bài kiểm tra môn Trí tuệ nhân tạo · Đại học Kinh doanh và Công nghệ Hà Nội<br>
Sinh viên <b>Hoàng Tiến Minh</b> · MSSV 25ACN046 · Lớp CNTT 3.2 · Giảng viên hướng dẫn: <b>Thu Hà</b><br>
Ứng dụng mang tính tham khảo và học tập, không thay thế chẩn đoán của bác sĩ.
</div>
""", unsafe_allow_html=True)
