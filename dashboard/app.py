"""Retention campaign decision dashboard.

Run from the repository root:  streamlit run dashboard/app.py
Needs the outputs of notebooks 02–04 (data/processed/, model_output/, targeting_output/).

Every number is recomputed from the sliders, which start at the base case in src/config.yaml.
- "Action list" figures use every customer in the base, scored by the selected model (train customers are in-sample).
- "Validated on test" figures use only the 1,409 held-out test customers and their actual churn, scaled to the base.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import preprocessing as pp  # noqa: E402
from src import targeting as tg  # noqa: E402
from src import value as vl  # noqa: E402

BLUE, ORANGE, AQUA, PURPLE, MUTED, LIGHT, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#898781", "#c3c2b7", "#0b0b0b"
POLICY_COLORS = {"ev": BLUE, "risk": ORANGE, "value": AQUA, "eda_rule": PURPLE, "random": MUTED, "oracle": LIGHT}
MODEL_LABELS = {"lr": "Logistic regression (main)", "rf": "Random forest", "lgbm": "LightGBM"}

st.set_page_config(page_title="Retention campaign planner", layout="wide")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load():
    config = pp.load_config()
    data = pp.load_processed(config)
    scores_path = ROOT / "targeting_output" / "tables" / "customer_scores.csv"
    table_path = ROOT / "targeting_output" / "tables" / "customer_targeting_table.csv"
    if not scores_path.exists():
        st.error("targeting_output/ not found. Run `python main.py` (or notebook 04) first.")
        st.stop()
    scores = pd.read_csv(scores_path).drop(columns=["split", pp.TARGET])
    extra = pd.read_csv(table_path, usecols=[pp.ID_COL, "top_reasons", "suggested_offer"])
    data = data.merge(scores, on=pp.ID_COL).merge(extra, on=pp.ID_COL)
    return config, data


config, data = load()
biz0, sens = config["business"], config["sensitivity"]

# ---------------------------------------------------------------------------
# Sidebar: assumptions
# ---------------------------------------------------------------------------
st.sidebar.header("Campaign assumptions")
st.sidebar.caption("Start values come from `src/config.yaml`. Offer cost and save rate are assumptions: "
                   "the data contains no retention campaign.")
budget = st.sidebar.slider("Campaign budget for the whole base ($)", 0, 200_000, int(biz0["campaign_budget"]), 5_000)
offer_cost = st.sidebar.slider("Offer cost per contacted customer ($)", 5, 200, int(biz0["offer_cost"]), 5)
save_rate = st.sidebar.slider("Save rate: share of would-be churners the offer keeps", 0.01, 0.60,
                              float(biz0["save_rate"]), 0.01, format="%.2f")
with st.sidebar.expander("Customer value assumptions"):
    margin = st.slider("Gross margin on the monthly bill", 0.10, 0.90, float(biz0["gross_margin"]), 0.05)
    horizon = st.slider("Months a saved customer stays", 1, 36, int(biz0["retention_horizon_months"]), 1)
    disc = st.slider("Annual discount rate", 0.0, 0.30, float(biz0["annual_discount_rate"]), 0.01)
model = st.sidebar.selectbox("Churn model", list(MODEL_LABELS), format_func=MODEL_LABELS.get)

biz = {**biz0, "campaign_budget": budget, "offer_cost": offer_cost, "save_rate": save_rate, "gross_margin": margin,
       "retention_horizon_months": horizon, "annual_discount_rate": disc}

# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
d = data.copy()
d["p"] = d[f"p_{model}"]
d["V"] = vl.value_proxy(d, biz)
v_edges = vl.value_tier_edges(d["V"], biz)
d["value_tier"] = vl.value_tier(d["V"], v_edges, biz)
d["risk_tier"] = vl.risk_tier(d["p"], biz)
d["EV"] = tg.expected_value(d["p"], d["V"], save_rate, offer_cost)
hv = float(np.quantile(d["V"], biz["high_value_quantile"]))

k_budget = tg.n_offers(budget, offer_cost)
f_all = tg.Frame.from_df(d, d["p"], d["V"])
d["contact"] = tg.run_policy("ev", f_all, save_rate, offer_cost, k_budget)
d["rank"] = d["EV"].rank(ascending=False, method="first").astype(int)
n_positive = int((d["EV"] > 0).sum())
n_contact = int(d["contact"].sum())
spend = n_contact * offer_cost
expected_net = float(d.loc[d["contact"], "EV"].sum())

test = d[d["split"] == "test"].reset_index(drop=True)
scale = biz["base_size"] / len(test)
f_test = tg.Frame.from_df(test, test["p"], test["V"], test[pp.TARGET])
k_test = tg.n_offers(tg.split_budget(biz, len(test)), offer_cost)


@st.cache_data
def test_ci(model, budget, offer_cost, save_rate, margin, horizon, disc, n_boot=300):
    b = tg.bootstrap_policies(f_test, biz, hv, policies=["ev", "risk"], ref="risk", n_boot=n_boot, seed=42)
    return b


policies = tg.compare_policies(f_test, biz, hv)
ci = test_ci(model, budget, offer_cost, save_rate, margin, horizon, disc)
ev_test = policies.loc[tg.POLICY_LABELS["ev"]]


def money(x):
    sign = "−" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x / 1e6:,.2f}M" if x >= 1e6 else f"{sign}${x / 1e3:,.1f}k" if x >= 1e4 else f"{sign}${x:,.0f}"


def md_money(x):
    """money() for Markdown text: a bare "$" pair would be rendered as LaTeX."""
    return money(x).replace("$", "\\$")


# ---------------------------------------------------------------------------
# Decision summary
# ---------------------------------------------------------------------------
st.title("Retention campaign planner")
st.caption("Decision: **which customers get a retention offer this month, and how many**, given the budget. "
           "Customers are ranked by expected value  EV = p(churn) × save rate × value − offer cost.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Customers to contact", f"{n_contact:,}", f"{n_contact / len(d):.1%} of base", delta_color="off", delta_arrow="off")
c2.metric("Spend", money(spend), f"of {md_money(budget)} budget", delta_color="off", delta_arrow="off")
c3.metric("Expected net value", money(expected_net), "model-predicted, whole base", delta_color="off", delta_arrow="off")
c4.metric("Validated on test (scaled)", money(ev_test["net_value"] * scale),
          f"95% CI {md_money(ci.loc[tg.POLICY_LABELS['ev'], 'net_value_lo'] * scale)} – "
          f"{md_money(ci.loc[tg.POLICY_LABELS['ev'], 'net_value_hi'] * scale)}", delta_color="off", delta_arrow="off")
c5.metric("Worth an offer (EV > 0)", f"{n_positive:,}", f"needs {md_money(n_positive * offer_cost)}", delta_color="off", delta_arrow="off")

if n_positive == 0:
    st.error(f"**Do not run this campaign.** At a {save_rate:.0%} save rate and \\${offer_cost} per offer, no customer's "
             "expected saving covers the offer cost.")
elif k_budget < n_positive:
    st.info(f"**Contact the top {n_contact:,} customers by EV.** The budget is binding: {n_positive - n_contact:,} more "
            f"customers would still be worth an offer. Raising the budget to about {md_money(n_positive * offer_cost)} "
            f"adds an expected {md_money(float(d.loc[(d['EV'] > 0) & ~d['contact'], 'EV'].sum()))}.")
else:
    st.success(f"**Contact all {n_positive:,} customers with EV > 0 and keep {md_money(budget - spend)} of the budget.** "
               "Beyond them, every extra offer is expected to lose money.")

risk_diff = ci.loc[tg.POLICY_LABELS["ev"]]
st.caption(f"On the held-out test set, EV targeting earns {md_money(risk_diff['net_diff_vs_ref'] * scale)} more than "
           f"contacting the highest-risk customers with the same budget (95% CI {md_money(risk_diff['net_diff_lo'] * scale)} "
           f"to {md_money(risk_diff['net_diff_hi'] * scale)}, scaled to the base). Money figures assume the save rate is "
           "right; it is not measured in this data, so treat them as scenarios, not forecasts.")

tabs = st.tabs(["Compare policies", "Campaign size", "Sensitivity", "Target segments", "Fairness check", "Action list"])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Same budget, different ways to choose customers (held-out test, scaled to the base)")
    order = ["ev", "risk", "eda_rule", "value", "random", "oracle"]
    p = policies.loc[[tg.POLICY_LABELS[o] for o in order]].astype(float)
    fig = go.Figure(go.Bar(x=p["net_value"] * scale, y=p.index, orientation="h",
                           marker_color=[POLICY_COLORS[o] for o in order],
                           text=[money(v) for v in p["net_value"] * scale], textposition="outside", cliponaxis=False))
    fig.update_layout(height=340, margin=dict(l=10, r=40, t=10, b=10), yaxis=dict(autorange="reversed"),
                      xaxis_title="Net value under the assumptions")
    st.plotly_chart(fig, width="stretch")
    view = pd.DataFrame({
        "Offers": (p["contacted"] * scale).round().astype(int),
        "Net value": [money(v) for v in p["net_value"] * scale],
        "ROI": [f"{v:.0%}" if pd.notna(v) else "—" for v in p["roi"]],
        "Churners among contacted": [f"{v:.0%}" for v in p["precision"]],
        "Churners reached": [f"{v:.0%}" for v in p["recall"]],
        "High-value churners reached": [f"{v:.0%}" for v in p["hv_churner_coverage"]],
    }, index=p.index)
    st.dataframe(view, width="stretch")
    st.caption("Perfect foresight knows who will churn: an upper bound, not an option.")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("How many customers to contact")
    share = np.arange(len(test) + 1) / len(test)
    fig = go.Figure()
    for pol in ["ev", "risk", "value"]:
        curve = tg.profit_curve(tg.priority(pol, f_test, save_rate, offer_cost)[0], f_test, save_rate, offer_cost)
        fig.add_trace(go.Scatter(x=share, y=curve * scale, name=tg.POLICY_LABELS[pol], line=dict(color=POLICY_COLORS[pol])))
    fig.add_vline(x=k_test / len(test), line_dash="dash", line_color=INK,
                  annotation_text=f"budget: {k_budget:,} offers", annotation_position="top")
    fig.add_vline(x=n_positive / len(d), line_dash="dot", line_color=BLUE,
                  annotation_text=f"EV > 0: {n_positive:,}", annotation_position="bottom right")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), xaxis_tickformat=".0%",
                      xaxis_title="Share of customers contacted, in each policy's order",
                      yaxis_title="Cumulative net value (test, scaled to base)", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    budgets = np.unique(np.r_[np.linspace(0, 200_000, 41), budget])
    rows = []
    for b in budgets:
        bb = {**biz, "campaign_budget": float(b)}
        k = tg.n_offers(tg.split_budget(bb, len(test)), offer_cost)
        for pol in ["ev", "risk"]:
            rows.append({"budget": b, "policy": tg.POLICY_LABELS[pol],
                         "net": tg.evaluate_policy(pol, f_test, save_rate, offer_cost, k, hv)["net_value"] * scale})
    bt = pd.DataFrame(rows)
    fig = go.Figure()
    for pol in ["ev", "risk"]:
        t = bt[bt["policy"] == tg.POLICY_LABELS[pol]]
        fig.add_trace(go.Scatter(x=t["budget"], y=t["net"], name=tg.POLICY_LABELS[pol], line=dict(color=POLICY_COLORS[pol])))
    fig.add_vline(x=budget, line_dash="dash", line_color=INK)
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Budget for the base ($)",
                      yaxis_title="Net value (test, scaled)", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    st.caption("The EV policy stops at EV = 0, so extra budget beyond that point is simply not spent. "
               "The risk-only policy keeps spending and its value falls.")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("What if the assumptions are wrong?")
    grid = tg.sensitivity_grid(f_test, biz, hv, sens["save_rate"], sens["offer_cost"])
    piv = grid.pivot(index="save_rate", columns="offer_cost", values="ev_net") * scale
    fig = go.Figure(go.Heatmap(z=piv.values, x=[f"${c}" for c in piv.columns], y=[f"{s:.0%}" for s in piv.index],
                               colorscale="Blues", text=[[money(v) for v in r] for r in piv.values],
                               texttemplate="%{text}", colorbar=dict(title="Net")))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Offer cost",
                      yaxis_title="Save rate")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Net value of EV targeting at the current budget ({md_money(budget)}), test set scaled to the base. "
               f"{int((grid['ev_net'] <= 0).sum())} of {len(grid)} scenarios do not pay; there the EV policy contacts "
               "nobody rather than losing money.")

    tor = tg.one_way_sensitivity(test, test["p"].to_numpy(), test[pp.TARGET].to_numpy(), biz, sens["one_way"],
                                 biz["high_value_quantile"], d)
    base_net = tor["base_net"].iloc[0] * scale
    fig = go.Figure()
    fig.add_trace(go.Bar(y=tor["parameter"], x=(tor["net_at_low"] * scale - base_net), base=base_net, orientation="h",
                         name="low value", marker_color=ORANGE, text=tor["low"].astype(str), textposition="outside"))
    fig.add_trace(go.Bar(y=tor["parameter"], x=(tor["net_at_high"] * scale - base_net), base=base_net, orientation="h",
                         name="high value", marker_color=BLUE, text=tor["high"].astype(str), textposition="outside"))
    fig.update_layout(barmode="overlay", height=330, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"), xaxis_title="Net value (test, scaled), one assumption changed")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Who is in the campaign (whole base)")
    tiers, vtiers = biz["risk_tiers"]["labels"][::-1], biz["value_tiers"]["labels"]
    m = d.groupby(["risk_tier", "value_tier"]).agg(n=("contact", "size"), contacted=("contact", "sum"),
                                                   mean_p=("p", "mean"), mean_V=("V", "mean"))
    m = m.reindex(pd.MultiIndex.from_product([tiers, vtiers]))
    z = (m["contacted"] / m["n"]).unstack().reindex(index=tiers, columns=vtiers)
    txt = [[f"{int(m.loc[(r, c), 'contacted'])} of {int(m.loc[(r, c), 'n'])}<br>p̄ {m.loc[(r, c), 'mean_p']:.0%} · "
            f"V̄ ${m.loc[(r, c), 'mean_V']:,.0f}" for c in vtiers] for r in tiers]
    fig = go.Figure(go.Heatmap(z=z.values, x=[f"{c} value" for c in vtiers], y=[f"{r} risk" for r in tiers],
                               colorscale=[[0, "#f5f8fc"], [1, "#7fb1ea"]], zmin=0, zmax=1, text=txt,
                               texttemplate="%{text}", textfont=dict(color=INK),
                               colorbar=dict(title="Contacted", tickformat=".0%")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

    d["tenure_seg"] = pd.cut(d["tenure"], [-1, 12, 24, 72], labels=["0–12 mo", "13–24 mo", "25–72 mo"]).astype(str)
    seg = d.groupby(["Contract", "InternetService", "tenure_seg"]).agg(
        customers=("contact", "size"), contacted=("contact", "sum"), avg_p_churn=("p", "mean"), avg_value=("V", "mean"),
        expected_net=("EV", lambda s: s[d.loc[s.index, "contact"]].sum()))
    seg = seg[seg["contacted"] > 0].sort_values("contacted", ascending=False)
    seg["share_of_offers"] = seg["contacted"] / max(n_contact, 1)
    st.dataframe(seg.style.format({"avg_p_churn": "{:.0%}", "avg_value": "${:,.0f}", "expected_net": "${:,.0f}",
                                   "share_of_offers": "{:.0%}"}), width="stretch")
    st.markdown("**Suggested offer by segment** (playbook; see `docs/retention_playbook.md`)")
    st.dataframe(d[d["contact"]].groupby("suggested_offer").agg(customers=("contact", "size"), avg_p=("p", "mean"))
                 .sort_values("customers", ascending=False).style.format({"avg_p": "{:.0%}"}), width="stretch")

# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Are offers spread fairly? (held-out test)")
    st.caption("A churner in each group should have a similar chance of being offered help (recall among churners). "
               "Differences that follow real differences in churn risk are expected; large gaps at similar risk are not. "
               "The model does not use gender.")
    contact_t = tg.run_policy("ev", f_test, save_rate, offer_cost, k_test)
    yn = {0: "No", 1: "Yes"}
    for col in ["SeniorCitizen", "gender", "Partner", "Dependents"]:
        g = test[col].map(yn) if col != "gender" else test[col]
        t = tg.group_table(contact_t, test[pp.TARGET].to_numpy(), g)
        st.markdown(f"**{col}**")
        st.dataframe(t.style.format({"churn_rate": "{:.0%}", "contact_rate": "{:.0%}", "share_of_offers": "{:.0%}",
                                     "precision": "{:.0%}", "recall_churners": "{:.0%}"}, na_rep="—"),
                     width="stretch")

# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader(f"Action list: {n_contact:,} customers to contact, in priority order")
    cols = ["rank", pp.ID_COL, "p", "risk_tier", "V", "value_tier", "EV", "top_reasons", "suggested_offer", "Contract",
            "InternetService", "PaymentMethod", "tenure", "MonthlyCharges"]
    out = d.loc[d["contact"], cols].sort_values("rank").rename(
        columns={"p": "p_churn", "V": "value_proxy", "EV": "expected_value"})
    st.dataframe(out.style.format({"p_churn": "{:.0%}", "value_proxy": "${:,.0f}", "expected_value": "${:,.0f}",
                                   "MonthlyCharges": "${:,.2f}"}), width="stretch", height=480, hide_index=True)
    st.download_button("Download action list (CSV)", out.to_csv(index=False).encode(), "retention_action_list.csv",
                       "text/csv")
    st.caption("Scores for customers in the training split are in-sample. In production the list would be built from "
               "active customers at the start of the month.")
