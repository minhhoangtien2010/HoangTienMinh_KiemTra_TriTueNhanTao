# -*- coding: utf-8 -*-
"""
train_model.py - Em huấn luyện và đánh giá mô hình dự đoán nguy cơ tiểu đường ở tệp này.
Sinh viên: Hoàng Tiến Minh - Lớp CNTT 3.2 - MSSV 25ACN046 - Giảng viên: Thu Hà
Chạy:  python train_model.py
Kết quả: thư mục model/ (mô hình + số liệu), figures/ (biểu đồ), data/ (dữ liệu đã làm sạch)

Em làm đúng theo 7 bước của đề (bước 7 là giao diện nằm ở app.py):
  Bước 1  Xác định bài toán      -> phân loại nhị phân (0 = không nguy cơ, 1 = có nguy cơ)
  Bước 2  Thu thập dữ liệu       -> data/CSDL_TieuDuong.csv
  Bước 3  Tiền xử lý             -> 3.1 làm sạch, 3.2 mã hóa + chuẩn hóa, 3.3 chia 70/15/15
  Bước 4  Chọn mô hình           -> LR, DT, RF, SVM, KNN, XGBoost
  Bước 5  Huấn luyện             -> chỉnh siêu tham số trên tập Validation
  Bước 6  Đánh giá               -> Accuracy, Precision, Recall, F1, ROC-AUC trên tập Test
"""
import json
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (ParameterGrid, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import ml_utils as U

warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # không có xgboost vẫn chạy được
    HAS_XGB = False

SEED = 42
ROOT = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(ROOT, *a)
os.makedirs(P("model"), exist_ok=True)
os.makedirs(P("figures"), exist_ok=True)

# Em chọn bảng màu xanh dương / cam / xanh ngọc để người mù màu vẫn phân biệt được
BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985", "#222222"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b9b8b3", "axes.labelcolor": "#444444",
    "xtick.color": "#555555", "ytick.color": "#555555",
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(P("figures", name), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def metrics(y, pred, proba):
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, proba),
    }


# ======================================================================
# BƯỚC 1-2. ĐỌC DỮ LIỆU
# ======================================================================
print("== Bước 2: đọc dữ liệu")
raw = pd.read_csv(P("data", "CSDL_TieuDuong.csv"), encoding="utf-8-sig")
print("Kích thước dữ liệu thô:", raw.shape)

# ======================================================================
# BƯỚC 3.1. LÀM SẠCH
# ======================================================================
print("== Bước 3.1: làm sạch dữ liệu")
df, report = U.clean_dataframe(raw)
missing_before = raw.replace("", np.nan).isna().sum().to_dict()
share = U.placeholder_share(df, U.NUMERIC_COLS)
THRESH = 0.90
dropped = [c for c, s in share.items() if s > THRESH]
USED_NUM = [c for c in U.NUMERIC_COLS if c not in dropped]
USED_CAT = list(U.CATEGORICAL_COLS)
FEATURES = USED_NUM + USED_CAT
report["ty_le_gia_tri_pho_bien"] = share
report["cot_bi_loai"] = dropped
report["nguong_loai_cot"] = THRESH
report["dac_trung_su_dung"] = FEATURES
print("Cột bị loại vì >90% là 1 giá trị cố định:", dropped)
df.to_csv(P("data", "CSDL_TieuDuong_DaLamSach.csv"), index=False, encoding="utf-8-sig")

# ======================================================================
# BƯỚC 4 (EDA) - biểu đồ phân tích dữ liệu
# ======================================================================
print("== EDA: vẽ biểu đồ")
y_all = df[U.TARGET]

# 01. Phân bố nhãn
fig, ax = plt.subplots(figsize=(5.2, 3.6))
vc = y_all.value_counts().sort_index()
bars = ax.bar(["Không có nguy cơ (0)", "Có nguy cơ (1)"], vc.values, color=[BLUE, ORANGE], width=0.55)
for b, v in zip(bars, vc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{v:,}\n({v / len(df):.1%})", ha="center", fontsize=9, color=INK)
ax.set_ylim(0, vc.max() * 1.22); ax.set_ylabel("Số bệnh nhân"); ax.grid(axis="x", visible=False)
ax.set_title("Phân bố nhãn NguyCoTieuDuong")
save(fig, "01_phan_bo_nhan.png")

# 02. Glucose theo nhãn
fig, ax = plt.subplots(figsize=(7, 3.8))
bins = np.linspace(1, 25, 49)
ax.hist(df.loc[y_all == 0, "Glucose"].dropna().clip(upper=25), bins=bins, color=BLUE, alpha=.85, label="Không có nguy cơ", edgecolor="white", linewidth=.4)
ax.hist(df.loc[y_all == 1, "Glucose"].dropna().clip(upper=25), bins=bins, color=ORANGE, alpha=.85, label="Có nguy cơ", edgecolor="white", linewidth=.4)
ax.axvline(7.0, color=INK, ls="--", lw=1.2)
ax.text(7.2, ax.get_ylim()[1] * .9, "Glucose = 7,0", fontsize=9, color=INK)
ax.set_xlabel("Glucose (mmol/L; giá trị ≥ 25 gộp vào cột cuối)"); ax.set_ylabel("Số bệnh nhân"); ax.legend(frameon=False)
ax.set_title("Glucose gần như tách hẳn hai nhóm tại ngưỡng 7,0")
save(fig, "02_glucose_theo_nhan.png")

# 03. Thiếu và bị điền sẵn
fig, ax = plt.subplots(figsize=(7.2, 4.2))
order = sorted(U.NUMERIC_COLS, key=lambda c: share[c])
ypos = np.arange(len(order))
ax.barh(ypos, [share[c] * 100 for c in order], color=[ORANGE if share[c] > THRESH else BLUE for c in order], height=.6)
for i, c in enumerate(order):
    ax.text(share[c] * 100 + 1, i, f"{share[c]:.1%}", va="center", fontsize=8.5, color=INK)
ax.axvline(THRESH * 100, color=INK, ls="--", lw=1)
ax.text(THRESH * 100 - 1, len(order) - .4, "ngưỡng 90%", ha="right", fontsize=8.5)
ax.set_yticks(ypos); ax.set_yticklabels(order); ax.set_xlim(0, 112); ax.grid(axis="y", visible=False)
ax.set_xlabel("% dòng mang cùng một giá trị phổ biến nhất")
ax.set_title("Cột nào đã bị điền sẵn một giá trị cố định?")
save(fig, "03_gia_tri_dien_san.png")

# 04. Trước/sau xử lý ngoại lai (Glucose, Creatinine, AST, ALT)
raw_num = raw.copy()
for c in U.NUMERIC_COLS:
    raw_num[c] = pd.to_numeric(raw_num[c], errors="coerce")
fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.6))
for ax, c in zip(axes, ["Triglycerid", "Creatinine", "AST", "ALT"]):
    lo, hi = np.nanpercentile(df[c], [1, 99])
    a = raw_num[c].dropna(); b = df[c].dropna().clip(lo, hi)
    bp = ax.boxplot([a, b], widths=.55, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=2.2, markerfacecolor=GRAY, markeredgecolor="none", alpha=.6),
                    medianprops=dict(color=INK, lw=1.4))
    for patch, col in zip(bp["boxes"], [GRAY, BLUE]):
        patch.set_facecolor(col); patch.set_alpha(.55); patch.set_edgecolor(col)
    ax.set_xticklabels(["Trước", "Sau"]); ax.set_title(c, fontsize=10.5); ax.grid(axis="x", visible=False)
fig.suptitle("Biểu đồ hộp trước và sau khi làm sạch và giới hạn ngoại lai (phân vị 1%–99%)", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
save(fig, "04_ngoai_lai_truoc_sau.png")

# 05. Ma trận tương quan
fig, ax = plt.subplots(figsize=(7, 5.6))
cm_df = df[U.NUMERIC_COLS + [U.TARGET]].corr()
im = ax.imshow(cm_df, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cm_df))); ax.set_xticklabels(cm_df.columns, rotation=45, ha="right", fontsize=8.5)
ax.set_yticks(range(len(cm_df))); ax.set_yticklabels(cm_df.columns, fontsize=8.5); ax.grid(False)
for i in range(len(cm_df)):
    for j in range(len(cm_df)):
        v = cm_df.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8, color="white" if abs(v) > .55 else INK)
fig.colorbar(im, ax=ax, shrink=.8)
ax.set_title("Ma trận tương quan")
save(fig, "05_tuong_quan.png")

# ======================================================================
# BƯỚC 3.2-3.3. MÃ HÓA, CHUẨN HÓA, CHIA DỮ LIỆU 70 / 15 / 15
# ======================================================================
print("== Bước 3.3: chia dữ liệu 70/15/15 (stratify)")
X = df[FEATURES]
y = df[U.TARGET]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
split_info = {k: {"n": int(len(v)), "ty_le_co_nguy_co": float(w.mean())}
              for k, v, w in [("Train", X_train, y_train), ("Validation", X_val, y_val), ("Test", X_test, y_test)]}
print(split_info)


def make_preprocessor(num_cols, cat_cols):
    """Điền thiếu (median / mode) -> giới hạn ngoại lai -> chuẩn hóa; mã hóa One-Hot cho giới tính.
    Tất cả chỉ học (fit) trên tập Train thông qua Pipeline."""
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("winsor", U.Winsorizer(1, 99)),
                    ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))])
    tr = [("num", num, num_cols)]
    if cat_cols:
        tr.append(("cat", cat, cat_cols))
    return ColumnTransformer(tr)


pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
MODELS = {
    "Logistic Regression": (lambda **k: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED, **k),
                            {"C": [0.01, 0.1, 1, 10]}),
    "Decision Tree": (lambda **k: DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **k),
                      {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 20]}),
    "Random Forest": (lambda **k: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1, **k),
                      {"max_depth": [5, 10, None], "min_samples_leaf": [1, 3]}),
    "SVM (RBF)": (lambda **k: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED, **k),
                  {"C": [0.5, 1, 5]}),
    "KNN": (lambda **k: KNeighborsClassifier(**k), {"n_neighbors": [5, 9, 15, 25]}),
}
if HAS_XGB:
    MODELS["XGBoost"] = (lambda **k: XGBClassifier(n_estimators=200, scale_pos_weight=pos_weight, eval_metric="logloss",
                                                   random_state=SEED, n_jobs=1, verbosity=0, **k),
                         {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]})


def build(name, params, feats_num=USED_NUM, feats_cat=USED_CAT):
    return Pipeline([("prep", make_preprocessor(feats_num, feats_cat)),
                     ("model", MODELS[name][0](**params))])


# ======================================================================
# BƯỚC 4-5. CHỌN MÔ HÌNH + HUẤN LUYỆN (chỉnh siêu tham số trên Validation)
# ======================================================================
print("== Bước 4-5: huấn luyện và chỉnh siêu tham số")
results, best_models = [], {}
for name, (_, grid) in MODELS.items():
    t0 = time.time()
    best = None
    for params in ParameterGrid(grid):
        pipe = build(name, params).fit(X_train, y_train)
        pv = pipe.predict_proba(X_val)[:, 1]
        m = metrics(y_val, (pv >= .5).astype(int), pv)
        key = (round(m["F1"], 4), round(m["Recall"], 4), m["ROC_AUC"])
        if best is None or key > best[0]:
            best = (key, params, pipe, m)
    _, params, pipe, mval = best
    ptr = pipe.predict_proba(X_train)[:, 1]; pte = pipe.predict_proba(X_test)[:, 1]
    mtr = metrics(y_train, (ptr >= .5).astype(int), ptr); mte = metrics(y_test, (pte >= .5).astype(int), pte)
    best_models[name] = pipe
    results.append({"Mo_hinh": name, "Sieu_tham_so": {k: (None if v is None else v) for k, v in params.items()},
                    "Train": mtr, "Validation": mval, "Test": mte, "Giay": round(time.time() - t0, 1)})
    print(f"{name:20s} val F1={mval['F1']:.4f} recall={mval['Recall']:.4f} | test F1={mte['F1']:.4f}  {params}")

# Em chọn mô hình tốt nhất theo F1 trên Validation; nếu hòa thì xét Recall, vẫn hòa thì xét ROC-AUC
results.sort(key=lambda r: (round(r["Validation"]["F1"], 4), round(r["Validation"]["Recall"], 4), r["Validation"]["ROC_AUC"]), reverse=True)
BEST = results[0]["Mo_hinh"]
final = best_models[BEST]
print("==> Mô hình được chọn:", BEST)

# Em chạy cross-validation 5 phần (Train + Validation) để kiểm tra độ ổn định
X_dev = pd.concat([X_train, X_val]); y_dev = pd.concat([y_train, y_val])
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for r in results:
    pipe = build(r["Mo_hinh"], {k: v for k, v in r["Sieu_tham_so"].items()})
    sc = cross_val_score(pipe, X_dev, y_dev, cv=skf, scoring="f1")
    r["CV_F1_mean"] = float(sc.mean()); r["CV_F1_std"] = float(sc.std())

# ======================================================================
# BƯỚC 6. ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT TRÊN TẬP TEST
# ======================================================================
print("== Bước 6: đánh giá trên tập Test")
proba_test = final.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= .5).astype(int)
cm = confusion_matrix(y_test, pred_test)
final_metrics = metrics(y_test, pred_test, proba_test)

# Em so sánh với luật đơn giản: Glucose >= 7 -> có nguy cơ
glu_filled = X_test["Glucose"].fillna(X_train["Glucose"].median())
rule_pred = (glu_filled >= 7.0).astype(int)
rule_metrics = metrics(y_test, rule_pred, glu_filled)

# Em phân tích ảnh hưởng của ngưỡng quyết định
thr_rows = []
for t in (0.35, 0.50, 0.65):
    pr = (proba_test >= t).astype(int)
    thr_rows.append({"nguong": t, "Precision": precision_score(y_test, pr, zero_division=0),
                     "Recall": recall_score(y_test, pr), "F1": f1_score(y_test, pr),
                     "FN": int(((pr == 0) & (y_test == 1)).sum()), "FP": int(((pr == 1) & (y_test == 0)).sum())})

# Em thử bỏ Glucose để xem các chỉ số còn lại dự đoán được bao nhiêu
best_params = results[0]["Sieu_tham_so"]
nog_num = [c for c in USED_NUM if c != "Glucose"]
ab = build(BEST, best_params, nog_num, USED_CAT).fit(X_train, y_train)
pab = ab.predict_proba(X_test)[:, 1]
ablation = metrics(y_test, (pab >= .5).astype(int), pab)
print("Bỏ Glucose:", {k: round(v, 3) for k, v in ablation.items()})

# Em tính độ quan trọng đặc trưng (permutation importance trên Test)
pi = permutation_importance(final, X_test, y_test, scoring="f1", n_repeats=20, random_state=SEED, n_jobs=1)
imp = pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean, "Std": pi.importances_std}).sort_values("Importance", ascending=False)

# ---------------- biểu đồ kết quả ----------------
# 06. So sánh mô hình
fig, ax = plt.subplots(figsize=(8, 4))
names = [r["Mo_hinh"] for r in results]; xx = np.arange(len(names)); w = .27
for i, (key, lab, col) in enumerate([("F1", "F1-score", BLUE), ("Recall", "Recall", ORANGE), ("ROC_AUC", "ROC-AUC", AQUA)]):
    vals = [r["Test"][key] for r in results]
    ax.bar(xx + (i - 1) * w, vals, w - .03, color=col, label=lab)
ax.set_xticks(xx); ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9); ax.set_ylim(0.5, 1.03)
ax.set_ylabel("Giá trị trên tập Test"); ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0)); ax.grid(axis="x", visible=False)
ax.set_title("So sánh các mô hình trên tập Test", pad=30)
save(fig, "06_so_sanh_mo_hinh.png")

# 07. Ma trận nhầm lẫn
fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
labs = ["Không nguy cơ", "Có nguy cơ"]
ax.set_xticks([0, 1]); ax.set_xticklabels(labs); ax.set_yticks([0, 1]); ax.set_yticklabels(labs, rotation=90, va="center")
ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.grid(False)
for (i, j), v in np.ndenumerate(cm):
    nm = [["TN", "FP"], ["FN", "TP"]][i][j]
    ax.text(j, i, f"{v}\n{nm}", ha="center", va="center", fontsize=13, fontweight="bold", color="white" if v > cm.max() * .5 else INK)
ax.set_title(f"Ma trận nhầm lẫn - {BEST}")
save(fig, "07_ma_tran_nham_lan.png")

# 08. ROC
fig, ax = plt.subplots(figsize=(5.4, 4.4))
cols = [BLUE, ORANGE, AQUA, "#4a3aa7", "#eda100", "#e34948"]
for r, col in zip(results, cols):
    p = best_models[r["Mo_hinh"]].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, color=col, lw=2 if r["Mo_hinh"] == BEST else 1.3, label=f"{r['Mo_hinh']} ({r['Test']['ROC_AUC']:.3f})")
ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1)
ax.set_xlabel("Tỷ lệ dương tính giả (FPR)"); ax.set_ylabel("Recall (TPR)"); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Đường cong ROC (tập Test)")
save(fig, "08_roc.png")

# 09. Độ quan trọng đặc trưng
fig, ax = plt.subplots(figsize=(6.6, 3.9))
im = imp.sort_values("Importance")
ax.barh(im["Feature"], im["Importance"], xerr=im["Std"], color=BLUE, height=.6, ecolor=GRAY, error_kw=dict(lw=1))
for yy, v in enumerate(im["Importance"]):
    ax.text(max(v, 0) + .015, yy, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, im["Importance"].max() * 1.18)
ax.set_xlabel("Mức giảm F1 khi xáo trộn đặc trưng"); ax.grid(axis="y", visible=False)
ax.set_title(f"Độ quan trọng đặc trưng - {BEST}")
save(fig, "09_do_quan_trong.png")

# ======================================================================
# LƯU MÔ HÌNH VÀ SỐ LIỆU
# ======================================================================
meta = {
    "best_model": BEST, "best_params": best_params, "features": FEATURES,
    "num_features": USED_NUM, "cat_features": USED_CAT, "dropped_features": dropped,
    "medians": {c: float(X_train[c].median()) for c in USED_NUM},
    "ranges": {c: [float(df[c].quantile(.01)), float(df[c].quantile(.99))] for c in U.NUMERIC_COLS},
    "modes": {"GioiTinh": X_train["GioiTinh"].mode()[0]},
}
joblib.dump({"pipeline": final, "meta": meta}, P("model", "model_tieu_duong.joblib"))
out = {"report": report, "split": split_info, "results": results, "best": BEST, "final_metrics": final_metrics,
       "confusion_matrix": cm.tolist(), "rule_glucose7": rule_metrics, "ablation_no_glucose": ablation,
       "thresholds": thr_rows, "importance": imp.to_dict("records"), "missing_before": missing_before,
       "xgboost": HAS_XGB}
json.dump(out, open(P("model", "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("\nHOÀN TẤT. Mô hình tốt nhất:", BEST, {k: round(v, 4) for k, v in final_metrics.items()})
print("Luật Glucose>=7:", {k: round(v, 4) for k, v in rule_metrics.items()})
