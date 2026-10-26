"""Expected-value retention targeting simulator.

Decision: which customers get a retention offer in this month's campaign, given a fixed budget.

    EV_i = p_i × save_rate × V_i − offer_cost

- p_i: calibrated churn probability (notebook 03), V_i: value proxy (src/value.py)
- save_rate, offer_cost, budget: assumptions in config.yaml → business
- The offer is assumed to change nothing for customers who would stay anyway (no uplift data to say otherwise),
  and the offer cost is paid for every contacted customer, churner or not.

Two ways of scoring a policy on a labelled split (val / test):
- `expected_net`: Σ (p × s × V − c) over contacted customers – what the model *predicts* the campaign is worth
- `net_value`:    Σ (y × s × V − c) over contacted customers – the value *under the stated save_rate* using the customers
  who actually churned. This is not a causal estimate (save_rate is assumed), but it does not rely on the model's
  own probabilities, so it is the fair way to compare policies.

Key property: with one offer for everyone, EV ranks customers exactly like p × V (expected margin at risk), whatever the
values of save_rate, offer_cost, margin and horizon. The assumptions only decide **how far down the list** it is worth going.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import value as vl

POLICY_LABELS = {
    "ev": "Expected value (model)",
    "risk": "Highest churn risk (model)",
    "value": "Highest value only",
    "eda_rule": "EDA rule: M2M + Fiber, newest first",
    "random": "Random",
    "oracle": "Perfect foresight (upper bound)",
}
DEFAULT_POLICIES = ["ev", "risk", "value", "eda_rule", "random", "oracle"]


# ---------------------------------------------------------------------------
# Basic economics
# ---------------------------------------------------------------------------
def expected_value(p, v, save_rate, offer_cost):
    return np.asarray(p) * save_rate * np.asarray(v) - offer_cost


def breakeven_p(v, save_rate, offer_cost):
    """Churn probability above which an offer pays for itself: p > c / (s·V)."""
    return offer_cost / (save_rate * np.asarray(v, dtype=float))


def split_budget(biz, n):
    """Budget for a split of n customers: the campaign budget is set for the whole base."""
    return biz["campaign_budget"] * n / biz["base_size"]


def n_offers(budget, offer_cost):
    return int(np.floor(budget / offer_cost + 1e-9))


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    """Arrays a policy needs. `y` is only used by the oracle and for scoring outcomes."""
    p: np.ndarray
    v: np.ndarray
    y: np.ndarray
    m2m_fiber: np.ndarray
    tenure: np.ndarray
    monthly: np.ndarray

    @classmethod
    def from_df(cls, df, p, v, y=None):
        return cls(p=np.asarray(p, float), v=np.asarray(v, float),
                   y=np.zeros(len(df), int) if y is None else np.asarray(y, int),
                   m2m_fiber=df["m2m_fiber"].to_numpy(int), tenure=df["tenure"].to_numpy(float),
                   monthly=df["MonthlyCharges"].to_numpy(float))

    def take(self, idx):
        return Frame(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))

    def __len__(self):
        return len(self.p)


def priority(policy, f, save_rate, offer_cost):
    """(score, eligible): customers are contacted in descending score among eligible ones."""
    n = len(f)
    if policy == "ev":
        ev = expected_value(f.p, f.v, save_rate, offer_cost)
        return ev, ev > 0
    if policy == "risk":
        return f.p, np.ones(n, bool)
    if policy == "value":
        return f.v, np.ones(n, bool)
    if policy == "eda_rule":
        # Priority segment #1 of the EDA: Month-to-month × Fiber optic (55% churn, 72% of lost MRR),
        # newest customers first (churn is highest in the first year), higher bill breaks ties
        return -f.tenure + f.monthly / 1e4, f.m2m_fiber == 1
    if policy == "oracle":
        gain = f.y * save_rate * f.v - offer_cost
        return gain, gain > 0
    raise ValueError(f"Unknown policy: {policy}")


def select(score, eligible, k):
    """Boolean mask of the top-k eligible customers by score (stable order)."""
    order = np.argsort(-np.asarray(score), kind="stable")
    order = order[np.asarray(eligible)[order]][:max(k, 0)]
    mask = np.zeros(len(score), bool)
    mask[order] = True
    return mask


def run_policy(policy, f, save_rate, offer_cost, k):
    score, eligible = priority(policy, f, save_rate, offer_cost)
    return select(score, eligible, k)


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
def outcome(contact, f, save_rate, offer_cost, hv_threshold):
    """Campaign metrics for a contact mask on a labelled split."""
    c, y, v, p = contact, f.y, f.v, f.p
    n_c = int(c.sum())
    spend = n_c * offer_cost
    saved_value = save_rate * float((y * v)[c].sum())
    hv_churn = (y == 1) & (v >= hv_threshold)
    return {
        "contacted": n_c,
        "contact_share": n_c / len(f),
        "spend": spend,
        "churners_contacted": int(y[c].sum()),
        "expected_saves": save_rate * float(y[c].sum()),
        "saved_value": saved_value,
        "net_value": saved_value - spend,
        "expected_net": float((p * save_rate * v)[c].sum()) - spend,
        "roi": (saved_value - spend) / spend if spend else np.nan,
        "precision": float(y[c].mean()) if n_c else np.nan,
        "recall": float(y[c].sum() / max(y.sum(), 1)),
        "hv_churner_coverage": float(c[hv_churn].mean()) if hv_churn.any() else np.nan,
        "mean_p_contacted": float(p[c].mean()) if n_c else np.nan,
    }


def random_outcome(k, f, save_rate, offer_cost, hv_threshold):
    """Expected metrics of contacting k customers at random (exact expectation, no simulation noise)."""
    n, y, v, p = len(f), f.y, f.v, f.p
    frac = min(k, n) / n
    spend = min(k, n) * offer_cost
    saved_value = frac * save_rate * float((y * v).sum())
    return {
        "contacted": min(k, n), "contact_share": frac, "spend": spend,
        "churners_contacted": frac * y.sum(), "expected_saves": frac * save_rate * y.sum(),
        "saved_value": saved_value, "net_value": saved_value - spend,
        "expected_net": frac * save_rate * float((p * v).sum()) - spend,
        "roi": (saved_value - spend) / spend if spend else np.nan,
        "precision": float(y.mean()), "recall": frac, "hv_churner_coverage": frac,
        "mean_p_contacted": float(p.mean()),
    }


def evaluate_policy(policy, f, save_rate, offer_cost, k, hv_threshold):
    if policy == "random":
        return random_outcome(k, f, save_rate, offer_cost, hv_threshold)
    return outcome(run_policy(policy, f, save_rate, offer_cost, k), f, save_rate, offer_cost, hv_threshold)


def compare_policies(f, biz, hv_threshold, policies=DEFAULT_POLICIES, k=None):
    """One row per policy at the budget-implied number of offers `k` (default: from config, scaled to the split)."""
    s, c = biz["save_rate"], biz["offer_cost"]
    k = n_offers(split_budget(biz, len(f)), c) if k is None else k
    rows = {POLICY_LABELS[pol]: evaluate_policy(pol, f, s, c, k, hv_threshold) for pol in policies}
    return pd.DataFrame(rows).T.rename_axis("policy")


def bootstrap_policies(f, biz, hv_threshold, policies=DEFAULT_POLICIES, ref="risk", n_boot=1000, seed=42,
                       metrics=("net_value", "precision", "hv_churner_coverage")):
    """Paired bootstrap over customers: 95% CI of each metric, and of the net-value difference versus `ref`.

    Each resample re-runs the whole policy (ranking + cut-off) on the resampled customers with the same budget.
    """
    s, c = biz["save_rate"], biz["offer_cost"]
    k = n_offers(split_budget(biz, len(f)), c)
    rng = np.random.default_rng(seed)
    draws = {pol: {m: np.empty(n_boot) for m in metrics} for pol in policies}
    for b in range(n_boot):
        fb = f.take(rng.integers(0, len(f), len(f)))
        for pol in policies:
            o = evaluate_policy(pol, fb, s, c, k, hv_threshold)
            for m in metrics:
                draws[pol][m][b] = o[m]
    rows = []
    for pol in policies:
        row = {"policy": POLICY_LABELS[pol]}
        for m in metrics:
            row[f"{m}_lo"], row[f"{m}_hi"] = np.nanpercentile(draws[pol][m], [2.5, 97.5])
        diff = draws[pol]["net_value"] - draws[ref]["net_value"]
        row["net_diff_vs_ref"] = diff.mean()
        row["net_diff_lo"], row["net_diff_hi"] = np.percentile(diff, [2.5, 97.5])
        row["p_better_than_ref"] = (diff > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("policy")


# ---------------------------------------------------------------------------
# Profit curve and sensitivity
# ---------------------------------------------------------------------------
def profit_curve(score, f, save_rate, offer_cost, expected=False):
    """Cumulative net value when contacting customers in descending `score`, for k = 0..n."""
    order = np.argsort(-np.asarray(score), kind="stable")
    gain = (f.p if expected else f.y)[order] * save_rate * f.v[order] - offer_cost
    return np.concatenate([[0.0], np.cumsum(gain)])


def sensitivity_grid(f, biz, hv_threshold, save_rates, offer_costs, budget=None):
    """Net value of the EV and risk-only policies for every (save_rate, offer_cost) at a fixed budget in currency.

    A cheaper offer buys more contacts from the same budget. EV contacts only customers with EV > 0, so it may spend less.
    """
    budget = split_budget(biz, len(f)) if budget is None else budget
    rows = []
    for s in save_rates:
        for c in offer_costs:
            k = n_offers(budget, c)
            ev = evaluate_policy("ev", f, s, c, k, hv_threshold)
            risk = evaluate_policy("risk", f, s, c, k, hv_threshold)
            unconstrained = evaluate_policy("ev", f, s, c, len(f), hv_threshold)
            rows.append({"save_rate": s, "offer_cost": c, "max_offers": k,
                         "ev_contacted": ev["contacted"], "ev_net": ev["net_value"], "ev_roi": ev["roi"],
                         "risk_net": risk["net_value"], "ev_minus_risk": ev["net_value"] - risk["net_value"],
                         "best_possible_net": unconstrained["net_value"],
                         "best_possible_contacted": unconstrained["contacted"]})
    return pd.DataFrame(rows)


def with_params(biz, **changes):
    return {**biz, **changes}


def one_way_sensitivity(df, p, y, biz, ranges, hv_quantile, v_base_df):
    """Tornado table: net value of the EV policy when one assumption moves to its low / high value."""
    def net(b):
        v = vl.value_proxy(df, b)
        hv = np.quantile(vl.value_proxy(v_base_df, b), hv_quantile)
        f = Frame.from_df(df, p, v, y)
        k = n_offers(split_budget(b, len(f)), b["offer_cost"])
        return evaluate_policy("ev", f, b["save_rate"], b["offer_cost"], k, hv)["net_value"]

    base = net(biz)
    rows = []
    for param, (lo, hi) in ranges.items():
        rows.append({"parameter": param, "base": biz[param], "low": lo, "high": hi,
                     "net_at_low": net(with_params(biz, **{param: lo})),
                     "net_at_high": net(with_params(biz, **{param: hi}))})
    out = pd.DataFrame(rows)
    out["base_net"] = base
    out["swing"] = (out["net_at_high"] - out["net_at_low"]).abs()
    return out.sort_values("swing", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Segment-specific offers (stretch)
# ---------------------------------------------------------------------------
def assign_offers(df, offers):
    """First matching rule wins. Returns (offer name, cost, save_rate) arrays."""
    name = np.full(len(df), None, dtype=object)
    cost, save = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for o in offers:
        rule = o["rule"].strip()
        match = np.ones(len(df), bool) if rule.lower() in ("all", "true") else df.eval(rule).to_numpy(bool)
        new = match & (name == None)  # noqa: E711 – elementwise check on an object array
        name[new], cost[new], save[new] = o["name"], o["offer_cost"], o["save_rate"]
    if (name == None).any():  # noqa: E711
        raise ValueError("Some customers match no segment offer; add a catch-all rule ('all').")
    return name, cost, save


def segment_offer_campaign(p, v, y, cost, save, budget):
    """Contact customers with EV > 0 in descending EV per currency unit spent, until the budget runs out."""
    ev = np.asarray(p) * save * np.asarray(v) - cost
    order = np.argsort(-(ev / cost), kind="stable")
    order = order[ev[order] > 0]
    within = np.cumsum(cost[order]) <= budget + 1e-9
    contact = np.zeros(len(p), bool)
    contact[order[within]] = True
    spend = float(cost[contact].sum())
    saved = float((np.asarray(y) * save * v)[contact].sum())
    return contact, {"contacted": int(contact.sum()), "spend": spend, "saved_value": saved, "net_value": saved - spend,
                     "expected_net": float(ev[contact].sum()), "precision": float(np.asarray(y)[contact].mean())}


# ---------------------------------------------------------------------------
# Fairness / customer-experience checks
# ---------------------------------------------------------------------------
def group_table(contact, y, groups):
    """Per group: size, churn rate, share contacted, precision, and recall among churners (equal-opportunity view)."""
    d = pd.DataFrame({"group": np.asarray(groups), "contact": contact, "y": np.asarray(y)})
    g = d.groupby("group")
    out = pd.DataFrame({
        "n": g.size(),
        "churn_rate": g["y"].mean(),
        "contact_rate": g["contact"].mean(),
        "share_of_offers": g["contact"].sum() / max(d["contact"].sum(), 1),
        "precision": g.apply(lambda x: x.loc[x["contact"], "y"].mean() if x["contact"].any() else np.nan,
                             include_groups=False),
        "recall_churners": g.apply(lambda x: x.loc[x["y"] == 1, "contact"].mean(), include_groups=False),
    })
    return out
