# -*- coding: utf-8 -*-
"""
ml_utils.py - Em viết các hàm dùng chung cho train_model.py và app.py
Đề kiểm tra môn Trí tuệ nhân tạo: Dự đoán nguy cơ bệnh tiểu đường
Sinh viên: Hoàng Tiến Minh - Lớp CNTT 3.2 - MSSV 25ACN046
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------------------------------------------
# 1. KHAI BÁO CỘT DỮ LIỆU
# ------------------------------------------------------------------
TARGET = "NguyCoTieuDuong"
NUMERIC_COLS = ["Tuoi", "Glucose", "HbA1C", "Cholesterol", "HDL", "LDL",
                "Triglycerid", "Creatinine", "Ure", "AcidUric", "AST", "ALT"]
CATEGORICAL_COLS = ["GioiTinh"]

# Em đặt tên hiển thị + đơn vị để dùng cho giao diện và báo cáo
LABELS = {
    "Tuoi": ("Tuổi", "năm"),
    "GioiTinh": ("Giới tính", ""),
    "Glucose": ("Đường huyết (Glucose)", "mmol/L"),
    "HbA1C": ("HbA1C", "%"),
    "Cholesterol": ("Cholesterol toàn phần", "mmol/L"),
    "HDL": ("HDL-C", "mmol/L"),
    "LDL": ("LDL-C", "mmol/L"),
    "Triglycerid": ("Triglycerid", "mmol/L"),
    "Creatinine": ("Creatinine", "µmol/L"),
    "Ure": ("Ure", "mmol/L"),
    "AcidUric": ("Acid Uric", "µmol/L"),
    "AST": ("AST (GOT)", "U/L"),
    "ALT": ("ALT (GPT)", "U/L"),
}

# Khoảng giá trị hợp lý về mặt sinh học; em coi giá trị ngoài khoảng này là dữ liệu lỗi (nhập sai).
PLAUSIBLE = {
    "Tuoi": (0, 110), "Glucose": (1.0, 40.0), "HbA1C": (3.0, 20.0),
    "Cholesterol": (1.0, 20.0), "HDL": (0.2, 5.0), "LDL": (0.5, 12.0),
    "Triglycerid": (0.2, 40.0), "Creatinine": (10.0, 1500.0), "Ure": (0.5, 70.0),
    "AcidUric": (50.0, 1200.0), "AST": (1.0, 1000.0), "ALT": (1.0, 1000.0),
}

# ------------------------------------------------------------------
# 2. LÀM SẠCH DỮ LIỆU (bước 3.1)
# ------------------------------------------------------------------
GENDER_MAP = {"nam": "Nam", "male": "Nam", "m": "Nam",
              "nữ": "Nữ", "nu": "Nữ", "female": "Nữ", "f": "Nữ"}
TARGET_MAP = {"1": 1, "true": 1, "có nguy cơ": 1, "co nguy co": 1,
              "0": 0, "false": 0, "không có nguy cơ": 0, "khong co nguy co": 0}


def clean_dataframe(raw: pd.DataFrame, batch: bool = False):
    """Làm sạch dữ liệu thô. Trả về (df_sạch, báo_cáo).

    Các bước (chưa điền dữ liệu thiếu - việc điền sẽ làm trong Pipeline,
    chỉ học trên tập Train để tránh rò rỉ dữ liệu):
      1) Xóa khoảng trắng, đồng bộ Giới tính và nhãn
      2) Ép kiểu số, giá trị không phải số -> NaN
      3) Sửa đơn vị Glucose (mg/dL -> mmol/L)
      4) Giá trị ngoài khoảng sinh học hợp lý -> NaN (dữ liệu lỗi)
      5) Xóa dòng trùng, xóa dòng không có nhãn
    """
    df = raw.copy()
    rep = {"so_dong_ban_dau": int(len(df))}

    # 1) Em đồng bộ dữ liệu
    df.columns = [c.strip() for c in df.columns]
    g = df["GioiTinh"].astype("string").str.strip().str.lower()
    rep["gioi_tinh_chua_dong_nhat"] = int(g.isin(["male", "female"]).sum())
    df["GioiTinh"] = g.map(GENDER_MAP)
    if not batch:
        t = df[TARGET].astype("string").str.strip().str.lower()
        rep["nhan_chua_dong_nhat"] = int((~t.isin(["true", "false"])).sum())  # nhãn không ở dạng TRUE/FALSE chuẩn
        df[TARGET] = t.map(TARGET_MAP)

    # 2) Em ép kiểu số
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) Glucose nhập theo mg/dL (> 50) thì em đổi sang mmol/L (chia 18,016)
    mask_mgdl = df["Glucose"] > 50
    rep["glucose_doi_don_vi"] = int(mask_mgdl.sum())
    df.loc[mask_mgdl, "Glucose"] = df.loc[mask_mgdl, "Glucose"] / 18.016

    # 4) Dữ liệu lỗi (ngoài khoảng sinh học): em chuyển thành thiếu rồi điền sau
    loi = {}
    for c, (lo, hi) in PLAUSIBLE.items():
        bad = (df[c] < lo) | (df[c] > hi)
        loi[c] = int(bad.sum())
        df.loc[bad, c] = np.nan
    rep["gia_tri_loi_chuyen_NaN"] = loi

    if batch:  # dự đoán hàng loạt: giữ nguyên số dòng để ghép kết quả
        return df, rep

    # 5) Em xóa dòng trùng và dòng thiếu nhãn
    rep["dong_trung"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    rep["dong_thieu_nhan"] = int(df[TARGET].isna().sum())
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = df.reset_index(drop=True)

    rep["so_dong_sau_lam_sach"] = int(len(df))
    rep["thieu_sau_lam_sach"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    return df, rep


def placeholder_share(df: pd.DataFrame, cols):
    """Tỷ lệ dòng mang cùng 1 giá trị phổ biến nhất (phát hiện cột bị điền sẵn)."""
    out = {}
    for c in cols:
        s = df[c].dropna()
        out[c] = float(s.value_counts(normalize=True).iloc[0]) if len(s) else 0.0
    return out


# ------------------------------------------------------------------
# 3. XỬ LÝ NGOẠI LAI (bước 3.1) - giới hạn theo phân vị, học trên Train
# ------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Giới hạn (cap) giá trị ngoại lai vào khoảng [phân vị thấp, phân vị cao].

    Dùng thay cho IQR vì nhiều cột xét nghiệm có IQR = 0 (đa số bằng nhau),
    nếu dùng IQR sẽ cắt nhầm gần hết dữ liệu. Giữ lại bệnh nhân bất thường
    thật (men gan cao...) thay vì xóa họ khỏi dữ liệu.
    """

    def __init__(self, lower=1.0, upper=99.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.lower, axis=0)
        self.hi_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, names=None):
        return np.asarray(names)


# ------------------------------------------------------------------
# 4. CẢNH BÁO LÂM SÀNG THAM KHẢO (dựa trên ngưỡng xét nghiệm phổ biến)
# ------------------------------------------------------------------
def clinical_flags(v: dict, gioi_tinh: str):
    """Trả về danh sách (tên chỉ số, mức, nhận xét). mức: ok / warn / high.
    Ngưỡng mang tính tham khảo chung; mỗi phòng xét nghiệm có thể khác nhau."""
    nam = gioi_tinh == "Nam"
    f = []

    def add(name, level, text):
        f.append((name, level, text))

    g = v.get("Glucose")
    if g is not None:
        if g >= 7.0:
            add("Glucose", "high", f"{g:.1f} mmol/L ≥ 7,0: đạt ngưỡng chẩn đoán tiểu đường (nếu đo lúc đói), cần xét nghiệm lại và gặp bác sĩ.")
        elif g >= 5.6:
            add("Glucose", "warn", f"{g:.1f} mmol/L trong khoảng 5,6–6,9: rối loạn đường huyết lúc đói (tiền tiểu đường).")
        else:
            add("Glucose", "ok", f"{g:.1f} mmol/L: trong giới hạn bình thường (< 5,6).")
    h = v.get("HbA1C")
    if h is not None:
        if h >= 6.5:
            add("HbA1C", "high", f"{h:.2f}% ≥ 6,5%: đạt ngưỡng chẩn đoán tiểu đường.")
        elif h >= 5.7:
            add("HbA1C", "warn", f"{h:.2f}% trong khoảng 5,7–6,4%: nguy cơ tiền tiểu đường.")
        else:
            add("HbA1C", "ok", f"{h:.2f}%: bình thường (< 5,7%).")
    c = v.get("Cholesterol")
    if c is not None:
        if c >= 6.2:
            add("Cholesterol", "high", f"{c:.1f} mmol/L ≥ 6,2: mỡ máu toàn phần cao.")
        elif c >= 5.2:
            add("Cholesterol", "warn", f"{c:.1f} mmol/L: giới hạn cao (5,2–6,1).")
        else:
            add("Cholesterol", "ok", f"{c:.1f} mmol/L: bình thường (< 5,2).")
    tg = v.get("Triglycerid")
    if tg is not None:
        if tg >= 2.3:
            add("Triglycerid", "high", f"{tg:.1f} mmol/L ≥ 2,3: triglycerid cao.")
        elif tg >= 1.7:
            add("Triglycerid", "warn", f"{tg:.1f} mmol/L: giới hạn cao (1,7–2,2).")
        else:
            add("Triglycerid", "ok", f"{tg:.1f} mmol/L: bình thường (< 1,7).")
    hdl = v.get("HDL")
    if hdl is not None:
        lim = 1.0 if nam else 1.3
        if hdl < lim:
            add("HDL", "warn", f"{hdl:.2f} mmol/L thấp (< {lim:.1f} với {'nam' if nam else 'nữ'}).")
        else:
            add("HDL", "ok", f"{hdl:.2f} mmol/L: đạt.")
    ldl = v.get("LDL")
    if ldl is not None:
        if ldl >= 4.1:
            add("LDL", "high", f"{ldl:.2f} mmol/L ≥ 4,1: LDL cao.")
        elif ldl >= 3.4:
            add("LDL", "warn", f"{ldl:.2f} mmol/L: giới hạn cao (3,4–4,0).")
        else:
            add("LDL", "ok", f"{ldl:.2f} mmol/L: bình thường (< 3,4).")
    cr = v.get("Creatinine")
    if cr is not None:
        lim = 115 if nam else 97
        if cr > lim:
            add("Creatinine", "warn", f"{cr:.0f} µmol/L > {lim}: cao, cần kiểm tra chức năng thận.")
        else:
            add("Creatinine", "ok", f"{cr:.0f} µmol/L: trong giới hạn tham khảo.")
    u = v.get("Ure")
    if u is not None:
        if u > 7.5:
            add("Ure", "warn", f"{u:.1f} mmol/L > 7,5: cao, cần kiểm tra chức năng thận.")
        else:
            add("Ure", "ok", f"{u:.1f} mmol/L: trong giới hạn tham khảo.")
    au = v.get("AcidUric")
    if au is not None:
        lim = 420 if nam else 360
        if au > lim:
            add("Acid Uric", "warn", f"{au:.0f} µmol/L > {lim}: cao (nguy cơ gút).")
        else:
            add("Acid Uric", "ok", f"{au:.0f} µmol/L: trong giới hạn tham khảo.")
    for k, nm in (("AST", "AST"), ("ALT", "ALT")):
        x = v.get(k)
        if x is not None:
            if x > 40:
                add(nm, "warn", f"{x:.0f} U/L > 40: men gan tăng, nên kiểm tra gan.")
            else:
                add(nm, "ok", f"{x:.0f} U/L: bình thường.")
    return f
