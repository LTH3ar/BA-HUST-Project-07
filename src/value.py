"""Customer value proxy for retention targeting.

V_i = MonthlyCharges_i × gross_margin × annuity(retention_horizon_months, annual_discount_rate)

V is the margin a customer would still bring in over the retention horizon if a churn were prevented, i.e. the
amount at stake in the retention decision. All three parameters live in `config.yaml → business` and are assumptions.

How tenure is used (the brief asks for tenure, charges and remaining-value assumptions):
- `past_value` = TotalCharges (≈ tenure × MonthlyCharges) is reported as relationship value for segments, the playbook and
  the dashboard, so a long-standing customer can be recognised.
- It is deliberately NOT a multiplier inside V. Revenue already paid is sunk and does not change what the offer can save,
  and tenure's effect on future behaviour is already inside p_churn (tenure is the strongest single driver). Multiplying V by
  tenure would count that effect twice. What tenure-based loyalty does to *future* value is covered by the
  `retention_horizon_months` assumption and its sensitivity range.
"""
import numpy as np
import pandas as pd


def monthly_rate(annual_rate):
    return (1 + annual_rate) ** (1 / 12) - 1


def annuity_factor(months, annual_rate):
    """Present value of 1 per month for `months` months, paid at the end of each month."""
    r = monthly_rate(annual_rate)
    t = np.arange(1, int(months) + 1)
    return float(np.sum((1 + r) ** -t))


def value_proxy(df, biz):
    """Margin at stake over the retention horizon, per customer (currency units)."""
    a = annuity_factor(biz["retention_horizon_months"], biz["annual_discount_rate"])
    return df["MonthlyCharges"].to_numpy(dtype=float) * biz["gross_margin"] * a


def value_tier_edges(v_base, biz):
    """Tier edges from the quantiles of V over the whole base (fixed edges, so every split uses the same tiers)."""
    edges = np.quantile(v_base, biz["value_tiers"]["quantiles"])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def value_tier(v, edges, biz):
    return pd.cut(v, bins=edges, labels=biz["value_tiers"]["labels"]).astype(str)


def risk_tier(p, biz):
    rt = biz["risk_tiers"]
    return pd.cut(p, bins=rt["bins"], labels=rt["labels"], include_lowest=True).astype(str)


def add_value_columns(df, biz, v_base=None):
    """Add value_proxy, past_value, value_tier. Tier edges come from `v_base` (default: V of `df` itself)."""
    out = df.copy()
    out["value_proxy"] = value_proxy(out, biz)
    out["past_value"] = out["TotalCharges"]
    edges = value_tier_edges(out["value_proxy"] if v_base is None else v_base, biz)
    out["value_tier"] = value_tier(out["value_proxy"], edges, biz)
    return out
