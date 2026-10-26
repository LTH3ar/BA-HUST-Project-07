"""Churn model baseline: logistic regression, random forest, LightGBM – tuning, calibration, evaluation.

Usage (notebook 03):
    config = pp.load_config()
    data = pp.load_processed(config)
    train, val = data[data["split"] == "train"], data[data["split"] == "val"]

    search = tune("lr", train, config)                       # CV on train, selected by log loss
    p_val = search.best_estimator_.predict_proba(val)[:, 1]
    p_cal = crossfit_calibrate(p_val, val[pp.TARGET], "sigmoid", config)   # honest evaluation on val
    model = CalibratedModel(search.best_estimator_, fit_calibrator(p_val, val[pp.TARGET], "sigmoid"))

Class imbalance is not handled with class_weight or resampling: both distort the p_churn the EV formula needs.
Imbalance is handled when choosing the threshold / number of customers to contact.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src import preprocessing as pp

MODELS = ["lr", "rf", "lgbm"]
MODEL_LABELS = {"lr": "Logistic regression", "rf": "Random forest", "lgbm": "LightGBM"}
CV_SCORING = {"neg_log_loss": "neg_log_loss", "roc_auc": "roc_auc",
              "average_precision": "average_precision", "neg_brier_score": "neg_brier_score"}

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
# Not used by any model: gender (no signal, p = 0.49, and a fairness risk), customerID (key).
#
# LR – based on the linearity checks and correlation notes (docs/eda_to_preprocessing.md sections 4.6, 7):
# - MonthlyCharges → monthly_charge_band; numeric tenure, without is_new_customer
# - no num_addons / auto_payment / has_protection / has_streaming alongside their source columns:
#   num_addons is the exact sum of the 6 add-on columns, auto_payment the sum of 2 PaymentMethod dummies → perfect collinearity
LR_FEATURES = dict(
    num_cols=["tenure"],
    cat_cols=["Contract", "InternetService", "PaymentMethod", "monthly_charge_band"],
    bin_cols=["SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
              *pp.INTERNET_ADDONS, "PaperlessBilling", "m2m_fiber", "m2m_echeck"],
)
# Candidate numeric features for LR – kept only if CV log loss improves (notebook 03 section 2)
LR_CANDIDATES = ["log_TotalCharges", "abs_charge_change"]

# Tree models find their own cut points and interactions, so they use raw forms. Drop features that are only a monotone
# transform or binning of an existing one (log_TotalCharges, monthly_charge_band, is_new_customer): trees gain nothing from them.
# Keep features that combine several columns (num_addons, auto_payment, m2m_*…) since trees need many splits to learn them.
TREE_FEATURES = dict(
    num_cols=["tenure", "MonthlyCharges", "TotalCharges", "avg_past_charge", "charge_change", "num_addons"],
    cat_cols=["Contract", "InternetService", "PaymentMethod"],
    bin_cols=["SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", *pp.INTERNET_ADDONS,
              "PaperlessBilling", "has_protection", "has_streaming", "auto_payment", "m2m_fiber", "m2m_echeck"],
)


def add_model_features(df):
    """Features created at the modelling step (not from EDA): size of the charge change, for charge_change's V-shaped signal."""
    return df.assign(abs_charge_change=df["charge_change"].abs())


def with_numeric(features, extra):
    """New feature set = `features` plus the numeric columns in `extra`."""
    return {**features, "num_cols": [*features["num_cols"], *extra]}


def xy(df):
    return df.drop(columns=[pp.TARGET]), df[pp.TARGET].to_numpy()


# ---------------------------------------------------------------------------
# Pipelines and tuning
# ---------------------------------------------------------------------------
def make_pipeline(name, config, features=None):
    """Pipeline(modelling-step features → encode/scale → classifier). The preprocessor is refit inside every CV fold."""
    seed, params = config["random_state"], config["models"][name].get("params", {})
    if name == "lr":
        prep = pp.build_preprocessor(**(features or LR_FEATURES), scale=True, drop_first=True)
        clf = LogisticRegression(max_iter=5000)
    elif name == "rf":
        prep = pp.build_preprocessor(**(features or TREE_FEATURES), scale=False, drop_first=False)
        clf = RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    elif name == "lgbm":
        prep = pp.build_preprocessor(**(features or TREE_FEATURES), scale=False, drop_first=False)
        clf = LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1, **params)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return Pipeline([("extra", FunctionTransformer(add_model_features)), ("prep", prep), ("clf", clf)])


def cv_splitter(config, n_splits=None):
    return StratifiedKFold(n_splits or config["models"]["cv_folds"], shuffle=True, random_state=config["random_state"])


def tune(name, train, config, features=None):
    """Grid search with stratified K-fold CV on train. Refit on `models.refit_metric` (log loss)."""
    grid = {f"clf__{k}": v for k, v in config["models"][name]["grid"].items()}
    search = GridSearchCV(make_pipeline(name, config, features), grid, scoring=CV_SCORING,
                          refit=config["models"]["refit_metric"], cv=cv_splitter(config), n_jobs=-1)
    X, y = xy(train)
    return search.fit(X, y)


def cv_summary(search):
    """CV scores (mean ± SE across folds) of the best parameter set. neg_* scores are flipped back to losses."""
    res, i, k = search.cv_results_, search.best_index_, search.n_splits_
    out = {}
    for key in CV_SCORING:
        folds = np.array([res[f"split{f}_test_{key}"][i] for f in range(k)])
        sign = -1 if key.startswith("neg_") else 1
        name = key.removeprefix("neg_")
        out[name], out[f"{name}_se"] = sign * folds.mean(), folds.std(ddof=1) / np.sqrt(k)
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class IdentityCalibrator:
    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p, dtype=float)


class PlattCalibrator:
    """Logistic regression on the logit scale: p_cal = σ(a · logit(p) + b), unregularised.

    a is the calibration slope: a > 1 means the raw model is too "timid" (probabilities squeezed toward the middle),
    a < 1 means it is overconfident. a = 1, b = 0 leaves probabilities unchanged.
    """

    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=np.inf).fit(_logit(p).reshape(-1, 1), y)
        self.slope_, self.intercept_ = self.lr_.coef_[0, 0], self.lr_.intercept_[0]
        return self

    def predict(self, p):
        return self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def fit(self, p, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def predict(self, p):
        return self.iso_.predict(p)


CALIBRATORS = {"none": IdentityCalibrator, "sigmoid": PlattCalibrator, "isotonic": IsotonicCalibrator}


def fit_calibrator(p, y, method):
    return CALIBRATORS[method]().fit(np.asarray(p), np.asarray(y))


def crossfit_calibrate(p, y, method, config):
    """Out-of-fold calibrated probabilities on val: each fold uses a calibrator fit on the other folds.

    Used to **evaluate** calibration without scoring on the same data the calibrator was fit on.
    The calibrator used for real (for test and the targeting table) is fit on all of val with `fit_calibrator`.
    """
    p, y = np.asarray(p), np.asarray(y)
    out = np.empty_like(p, dtype=float)
    for fit_idx, pred_idx in cv_splitter(config, config["calibration"]["cv_folds"]).split(p.reshape(-1, 1), y):
        out[pred_idx] = fit_calibrator(p[fit_idx], y[fit_idx], method).predict(p[pred_idx])
    return out


class CalibratedModel:
    """Pipeline fit on train + calibrator fit on val. `predict_churn` returns calibrated p_churn."""

    def __init__(self, pipeline, calibrator, method):
        self.pipeline, self.calibrator, self.method = pipeline, calibrator, method

    def predict_raw(self, df):
        return self.pipeline.predict_proba(df.drop(columns=[pp.TARGET], errors="ignore"))[:, 1]

    def predict_churn(self, df):
        return self.calibrator.predict(self.predict_raw(df))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _bins(p, n_bins):
    """Equal-frequency bins by p (bins hold roughly the same number of customers)."""
    return pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False).to_numpy()


def reliability_table(p, y, n_bins=10):
    """Per bin: mean predicted p versus observed churn rate."""
    df = pd.DataFrame({"p": p, "y": y, "bin": _bins(p, n_bins)})
    return df.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), churn_rate=("y", "mean"))


def expected_calibration_error(p, y, n_bins=10):
    t = reliability_table(p, y, n_bins)
    return float(np.average((t["mean_p"] - t["churn_rate"]).abs(), weights=t["n"]))


def calibration_slope(p, y):
    """Platt coefficient a on logit(p). Close to 1 is good."""
    return PlattCalibrator().fit(np.asarray(p), np.asarray(y)).slope_


def evaluate(p, y, n_bins=10):
    p, y = np.asarray(p), np.asarray(y)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(p, y, n_bins),
        "calib_slope": calibration_slope(p, y),
        "mean_p": p.mean(),
        "churn_rate": y.mean(),
    }


BOOTSTRAP_METRICS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score,
                     "log_loss": log_loss, "brier": brier_score_loss}


def bootstrap_compare(preds, y, ref, n_boot, seed):
    """Paired bootstrap on the same customer samples: 95% CI for each metric and for the difference versus model `ref`.

    `preds`: {model name: p}. A difference whose CI contains 0 cannot be told apart from `ref` on this set.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    draws = {(m, k): np.array([f(y[i], np.asarray(p)[i]) for i in idx])
             for m, p in preds.items() for k, f in BOOTSTRAP_METRICS.items()}
    rows = []
    for (m, k), d in draws.items():
        diff = d - draws[(ref, k)]
        rows.append({"model": m, "metric": k, "estimate": BOOTSTRAP_METRICS[k](y, preds[m]),
                     "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                     "diff_vs_ref": diff.mean(), "diff_ci_low": np.percentile(diff, 2.5),
                     "diff_ci_high": np.percentile(diff, 97.5)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------
def _confusion_rates(flag, y):
    tp = (flag & (y == 1)).sum()
    precision = tp / flag.sum() if flag.any() else np.nan
    recall = tp / (y == 1).sum()
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return precision, recall, f1


def threshold_table(p, y, thresholds):
    """Flag customers with p ≥ threshold: share flagged, precision, recall, F1."""
    p, y = np.asarray(p), np.asarray(y)
    rows = []
    for t in thresholds:
        flag = p >= t
        precision, recall, f1 = _confusion_rates(flag, y)
        rows.append({"threshold": t, "flagged_share": flag.mean(), "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def best_f1_threshold(p, y):
    """Threshold with the highest F1 – a technical reference point, not a decision threshold."""
    grid = np.unique(np.round(p, 3))
    f1 = threshold_table(p, y, grid)["f1"].to_numpy()
    return float(grid[np.nanargmax(f1)])


def topk_table(p, y, shares):
    """Contact the top-K% of customers by p (capacity / budget constraint).

    precision = observed churn rate among contacted customers; mean_p = churn rate the model predicts for that group.
    When the two are close, probabilities are trustworthy inside the group that gets the offer – what the EV formula needs.
    """
    p, y = np.asarray(p), np.asarray(y)
    order = np.argsort(-p, kind="stable")
    rows = []
    for s in shares:
        k = int(round(s * len(p)))
        top = order[:k]
        rows.append({"share": s, "k": k, "cutoff_p": p[top].min(), "mean_p": p[top].mean(),
                     "precision": y[top].mean(), "recall": y[top].sum() / y.sum(), "lift": y[top].mean() / y.mean()})
    return pd.DataFrame(rows)
