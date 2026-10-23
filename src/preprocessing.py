"""Làm sạch dữ liệu, feature engineering, chia tập và encoding/scaling cho Telco Churn.

Luồng sử dụng:
    config = load_config()
    df = add_features(clean_data(load_raw(config)), config)
    splits = assign_splits(df, config)
    save_processed(df, splits, config)

Ở bước mô hình:
    df = load_processed(config)                     # có cột `split`
    prep = build_preprocessor(num_cols, cat_cols, bin_cols, scale=True, drop_first=True)

Module này không xử lý mất cân bằng lớp và không chọn feature – hai việc đó thuộc bước mô hình.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "src" / "config.yaml"

ID_COL = "customerID"
TARGET = "y"

INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]
YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "MultipleLines", *INTERNET_ADDONS, "PaperlessBilling"]
AUTO_PAYMENTS = ["Bank transfer (automatic)", "Credit card (automatic)"]

# Nhóm cột của bảng đã xử lý – dùng làm "menu" khi chọn feature ở bước mô hình
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "log_TotalCharges",
            "avg_past_charge", "charge_change", "num_addons"]
CAT_COLS = ["gender", "Contract", "InternetService", "PaymentMethod", "monthly_charge_band"]
BIN_COLS = ["SeniorCitizen", *YES_NO_COLS,
            "is_new_customer", "has_protection", "has_streaming", "auto_payment", "m2m_fiber", "m2m_echeck"]
ENGINEERED_COLS = ["is_new_customer", "num_addons", "has_protection", "has_streaming",
                   "auto_payment", "monthly_charge_band", "log_TotalCharges", "avg_past_charge", "charge_change",
                   "m2m_fiber", "m2m_echeck"]
# Cờ nghiệp vụ cho phân khúc / playbook: thông tin đã nằm trong biến gốc, không nên dùng cùng biến gốc trong LR
BUSINESS_FLAG_COLS = {"is_new_customer": "tenure"}
# Cột dạng nhãn khoảng – đọc lại từ CSV phải giữ kiểu chuỗi
_BAND_COLS = ["monthly_charge_band"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(config):
    return pd.read_csv(ROOT / config["paths"]["raw_data"])


def save_processed(df, splits, config):
    for key, data in [("clean_data", df), ("splits", splits)]:
        path = ROOT / config["paths"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)


def load_processed(config):
    """Bảng đã xử lý kèm cột `split` (train / val / test)."""
    df = pd.read_csv(ROOT / config["paths"]["clean_data"], dtype={c: str for c in _BAND_COLS})
    splits = pd.read_csv(ROOT / config["paths"]["splits"])
    return df.merge(splits, on=ID_COL, how="inner", validate="one_to_one")


# ---------------------------------------------------------------------------
# Bước 1 – làm sạch (quy tắc cố định, không học từ dữ liệu)
# ---------------------------------------------------------------------------
def _to_binary(s):
    out = s.map({"Yes": 1, "No": 0})
    if out.isna().any():
        raise ValueError(f"{s.name}: giá trị ngoài Yes/No: {sorted(s[out.isna()].astype(str).unique())}")
    return out.astype(int)


def clean_data(df):
    out = df.copy()

    # TotalCharges: chuỗi rỗng chỉ xuất hiện ở khách tenure = 0 (chưa có hoá đơn) → 0
    total = pd.to_numeric(out["TotalCharges"], errors="coerce")
    unexpected = total.isna() & out["tenure"].ne(0)
    if unexpected.any():
        raise ValueError(f"TotalCharges thiếu ở {unexpected.sum()} khách có tenure > 0")
    out["TotalCharges"] = total.fillna(0.0)

    # "No internet service" / "No phone service" trùng thông tin với InternetService / PhoneService
    out[INTERNET_ADDONS + ["MultipleLines"]] = out[INTERNET_ADDONS + ["MultipleLines"]].replace(
        {"No internet service": "No", "No phone service": "No"})

    for c in YES_NO_COLS:
        out[c] = _to_binary(out[c])
    out[TARGET] = _to_binary(out["Churn"])
    return out.drop(columns="Churn")


# ---------------------------------------------------------------------------
# Bước 2 – feature engineering
# ---------------------------------------------------------------------------
def add_features(df, config):
    fc = config["features"]
    out = df.copy()

    # tenure giữ dạng số: gần tuyến tính với log-odds (notebook 02 mục 4) nên không chia khoảng
    out["is_new_customer"] = out["tenure"].le(fc["new_customer_max_tenure"]).astype(int)

    out["num_addons"] = out[INTERNET_ADDONS].sum(axis=1)
    out["has_protection"] = out[["OnlineSecurity", "TechSupport"]].max(axis=1)
    out["has_streaming"] = out[["StreamingTV", "StreamingMovies"]].max(axis=1)
    out["auto_payment"] = out["PaymentMethod"].isin(AUTO_PAYMENTS).astype(int)

    # Quan hệ phi tuyến với log-odds (EDA 7.1 / 4.2): churn tăng theo cước rồi giảm ở nhóm cao nhất;
    # TotalCharges lệch phải, churn giảm nhanh ở vùng thấp rồi chậm dần
    band = pd.cut(out["MonthlyCharges"], bins=fc["monthly_charge_bins"], labels=fc["monthly_charge_labels"])
    if band.isna().any():
        raise ValueError(f"MonthlyCharges nằm ngoài monthly_charge_bins ở {band.isna().sum()} khách")
    out["monthly_charge_band"] = band.astype(str)
    out["log_TotalCharges"] = np.log1p(out["TotalCharges"])

    # Khách tenure = 0 chưa có lịch sử → coi cước quá khứ bằng cước hiện tại
    out["avg_past_charge"] = out["TotalCharges"].div(out["tenure"].replace(0, np.nan)).fillna(out["MonthlyCharges"])
    out["charge_change"] = out["MonthlyCharges"] - out["avg_past_charge"]

    m2m = out["Contract"].eq("Month-to-month")
    out["m2m_fiber"] = (m2m & out["InternetService"].eq("Fiber optic")).astype(int)
    out["m2m_echeck"] = (m2m & out["PaymentMethod"].eq("Electronic check")).astype(int)
    return out


# ---------------------------------------------------------------------------
# Bước 3 – chia train / val / test
# ---------------------------------------------------------------------------
def assign_splits(df, config):
    """Trả về DataFrame [customerID, split]. Chia 2 lần: tách test, rồi tách val từ phần còn lại."""
    sc, seed = config["split"], config["random_state"]
    if not 0 < sc["test_size"] + sc["val_size"] < 1:
        raise ValueError("test_size + val_size phải nằm trong (0, 1)")

    def split(data, size):
        return train_test_split(data, test_size=size, random_state=seed,
                                stratify=data[TARGET] if sc["stratify"] else None)

    rest, test = split(df, sc["test_size"])
    train, val = split(rest, sc["val_size"] / (1 - sc["test_size"]))   # val tính trên toàn bộ dữ liệu
    return pd.concat([
        pd.DataFrame({ID_COL: part[ID_COL], "split": name})
        for name, part in [("train", train), ("val", val), ("test", test)]
    ], ignore_index=True)


# ---------------------------------------------------------------------------
# Bước 4 – encoding / scaling
# ---------------------------------------------------------------------------
def build_preprocessor(num_cols, cat_cols, bin_cols, scale=True, drop_first=False):
    """ColumnTransformer cho danh sách cột được chọn ở bước mô hình. Fit trên train, đặt trong Pipeline.

    - num_cols → StandardScaler (scale=True) hoặc giữ nguyên
    - cat_cols → OneHotEncoder (drop_first=True cho logistic regression để tránh dummy trap)
    - bin_cols → giữ nguyên 0/1
    Cột không được liệt kê sẽ bị bỏ.
    """
    encoder = OneHotEncoder(drop="first" if drop_first else None, handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [("num", StandardScaler() if scale else "passthrough", list(num_cols)),
         ("cat", encoder, list(cat_cols)),
         ("bin", "passthrough", list(bin_cols))],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
