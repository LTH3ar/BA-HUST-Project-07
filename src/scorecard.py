"""Points-based churn scorecard and per-customer churn reasons, both read from the logistic regression.

Scorecard: score = PDO / ln 2 × logit(p). With PDO = 20, +20 points doubles the odds of churning.
Because LR is linear in its inputs, the score splits exactly into
    base points + points for each category the customer is in + points per month of tenure × tenure
                + points per unit of ln(1 + TotalCharges) × ln(1 + TotalCharges)
so anyone can score a customer by hand from the table. Valid while the LR calibrator is the identity (notebook 03 chose
"none"); with Platt scaling the score maps to the raw LR probability, and the calibrated p comes from the calibrator.

Reasons: each customer's log-odds contribution of every source feature, measured against the average customer of the
training set (coef × (x − mean of x on train)), summed over the dummies of a categorical feature.
The three largest positive contributions are the "top reasons" in the targeting table.
"""
import numpy as np
import pandas as pd

PRETTY = {
    "tenure": "tenure", "log_TotalCharges": "low lifetime charges", "Contract": "contract", "InternetService": "internet",
    "PaymentMethod": "payment", "monthly_charge_band": "monthly bill", "SeniorCitizen": "senior citizen",
    "Partner": "partner", "Dependents": "dependents", "PhoneService": "phone service",
    "MultipleLines": "multiple lines", "OnlineSecurity": "online security", "OnlineBackup": "online backup",
    "DeviceProtection": "device protection", "TechSupport": "tech support", "StreamingTV": "streaming TV",
    "StreamingMovies": "streaming movies", "PaperlessBilling": "paperless billing",
    "m2m_fiber": "month-to-month fiber", "m2m_echeck": "month-to-month e-check",
}
INTERACTIONS = ["m2m_fiber", "m2m_echeck"]


def _parts(lr_model):
    pipe = lr_model.pipeline
    return pipe, pipe.named_steps["prep"], pipe.named_steps["clf"]


def _source_map(prep):
    """Encoded column → (source feature, level or None)."""
    out = {}
    for name, trans, cols in prep.transformers_:
        if name == "remainder":
            continue
        if name == "cat":
            for col, cats in zip(cols, trans.categories_):
                for level in cats:
                    out[f"{col}_{level}"] = (col, level)
        else:
            for col in cols:
                out[col] = (col, None)
    return out


def scorecard_table(lr_model, pdo=20):
    """Points table. Categorical rows give points per level (reference level = 0); numeric rows give points per unit."""
    pipe, prep, clf = _parts(lr_model)
    factor = pdo / np.log(2)
    names = prep.get_feature_names_out()
    coef = pd.Series(clf.coef_[0], index=names)
    scaler = prep.named_transformers_["num"]
    num_cols = list(prep.transformers_[0][2])
    mean, scale = pd.Series(scaler.mean_, index=num_cols), pd.Series(scaler.scale_, index=num_cols)

    base_logit = clf.intercept_[0] - sum(coef[c] * mean[c] / scale[c] for c in num_cols)
    rows = [{"feature": "Base points", "level": "reference customer, all numeric inputs = 0",
             "points": factor * base_logit, "coef": np.nan, "odds_ratio": np.nan}]
    for c in num_cols:
        per_unit = coef[c] / scale[c]
        unit = "per month" if c == "tenure" else "per unit of ln(1 + TotalCharges)"
        rows.append({"feature": c, "level": unit, "points": factor * per_unit, "coef": per_unit,
                     "odds_ratio": np.exp(per_unit)})
    smap = _source_map(prep)
    cat_trans = prep.named_transformers_["cat"]
    for col, cats, drop_idx in zip(prep.transformers_[1][2], cat_trans.categories_, cat_trans.drop_idx_):
        for i, level in enumerate(cats):
            b = 0.0 if i == drop_idx else coef[f"{col}_{level}"]
            rows.append({"feature": col, "level": f"{level}" + (" (reference)" if i == drop_idx else ""),
                         "points": factor * b, "coef": b, "odds_ratio": np.exp(b)})
    for col in prep.transformers_[2][2]:
        rows.append({"feature": col, "level": "= 1 (yes)", "points": factor * coef[col], "coef": coef[col],
                     "odds_ratio": np.exp(coef[col])})
    assert set(names) <= set(smap)
    return pd.DataFrame(rows)


def score_customers(lr_model, df, pdo=20):
    """Scorecard score from the table (independent re-computation) and the raw LR probability, for a self-check."""
    table = scorecard_table(lr_model, pdo)
    factor = pdo / np.log(2)
    score = np.full(len(df), table.loc[0, "points"])
    tc = df.assign(log_TotalCharges=np.log1p(df["TotalCharges"]))
    for _, r in table.iloc[1:].iterrows():
        f = r["feature"]
        if r["level"].startswith("per"):
            score += r["points"] * tc[f].to_numpy(float)
        elif r["level"] == "= 1 (yes)":
            score += r["points"] * tc[f].to_numpy(float)
        else:
            level = r["level"].replace(" (reference)", "")
            score += r["points"] * (tc[f].astype(str) == level).to_numpy(float)
    p_from_score = 1 / (1 + np.exp(-score / factor))
    return score, p_from_score


def contributions(lr_model, df, train_df):
    """Log-odds contribution of each source feature versus the average training customer (rows = customers)."""
    pipe, prep, clf = _parts(lr_model)
    transform = pipe[:-1].transform
    X = transform(df.drop(columns=["y"], errors="ignore"))
    mu = transform(train_df.drop(columns=["y"], errors="ignore")).mean()
    coef = pd.Series(clf.coef_[0], index=X.columns)
    contrib = (X - mu) * coef
    smap = _source_map(prep)
    groups = [smap[c][0] for c in X.columns]
    return contrib.T.groupby(groups).sum().T.set_index(df.index)


def _describe(feature, row):
    v = row.get(feature)
    if feature == "tenure":
        return f"short tenure ({int(v)} mo)"
    if feature == "log_TotalCharges":
        return f"low lifetime charges (${row['TotalCharges']:,.0f})"
    if feature in ("Contract", "InternetService", "PaymentMethod"):
        return f"{PRETTY[feature]}: {v}"
    if feature == "monthly_charge_band":
        return f"monthly bill {v}"
    label = PRETTY.get(feature, feature)
    return label if v == 1 else f"no {label}"


def top_reasons(contrib, df, n=3, min_logit=0.05):
    """Text of the n largest positive contributions per customer ("why is this customer risky?")."""
    out = []
    for idx, row in contrib.iterrows():
        # An interaction flag at 0 only corrects its main effects; "not month-to-month e-check" is no reason to a reader
        row = row.drop([c for c in INTERACTIONS if c in row.index and df.loc[idx, c] == 0])
        top = row[row > min_logit].sort_values(ascending=False).head(n)
        out.append("; ".join(_describe(f, df.loc[idx]) for f in top.index))
    return pd.Series(out, index=contrib.index, name="top_reasons")
