@echo off
chcp 65001 >nul
echo === Cai dat thu vien (chi can lam 1 lan) ===
python -m pip install -r requirements.txt
if not exist model\model_tieu_duong.joblib (
  echo === Huan luyen mo hinh lan dau ===
  python train_model.py
)
echo === Mo ung dung ===
python -m streamlit run app.py
pause
