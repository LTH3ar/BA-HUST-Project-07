# Retention playbook

How to run the monthly retention campaign with the churn model and the expected-value rule. Numbers are from
`notebooks/04_targeting.ipynb` (test set, scaled to the 7,043-customer base) under the base-case assumptions in
`src/config.yaml`. Every assumption can be changed there or with the dashboard sliders.

## 1. Expected-value logic

For each customer *i*:

```
V_i  = MonthlyCharges_i × gross_margin × annuity(horizon, discount rate)      value at stake if a churn is prevented
EV_i = p_i × save_rate × V_i − offer_cost                                    expected net value of sending an offer
contact i  ⇔  EV_i > 0  ⇔  p_i > offer_cost / (save_rate × V_i)              break-even churn probability
```

- `p_i` comes from the logistic regression (calibrated: ECE 0.02 on test, so it can be used as a real probability).
- The offer cost is paid for every contacted customer, including those who would have stayed. The offer is assumed to change
  nothing for them, since there is no uplift data to say otherwise.
- **Order.** Sort by EV and go down the list. With one offer type, this order equals sorting by `p × MonthlyCharges`
  (expected monthly revenue at risk), whatever the assumptions. Assumptions decide where to stop, not who comes first.
- **Stop** when the budget runs out or EV ≤ 0, whichever comes first.

Base-case assumptions (all in `config.yaml → business`):

| Parameter | Base case | Sensitivity range used | Why this value |
|---|---|---|---|
| `offer_cost` | $40 per contact | $10 – $150 | About half a median monthly bill as credit, plus contact cost |
| `save_rate` | 25% | 5% – 50% | Mid-range planning assumption. **Not measured; the pilot replaces it** |
| `gross_margin` | 50% | 30% – 70% | Contribution margin on service revenue |
| `retention_horizon_months` | 12 | 6 – 24 | A saved customer is assumed to stay one more year |
| `annual_discount_rate` | 10% | 0% – 20% | Time value of money |
| `campaign_budget` | $40,000 | $10k – $200k | Planning figure for the whole base |

With these values, V = MonthlyCharges × 5.70 (median ≈ $400), and a typical customer needs a churn probability above about
40% to be worth a $40 offer.

## 2. Campaign size

| Budget for the base | Offers | Net value (EV rule) | Net value (risk-only rule) |
|---|---|---|---|
| $20,000 | 500 | ≈ $25k | ≈ $27k |
| **$40,000 (planned)** | **1,000** | **≈ $44k** | ≈ $41k |
| $60,000 | 1,500 | ≈ $55k | ≈ $51k |
| $80,000 (≈ EV > 0 cut-off of ~2,060 offers) | 2,000 | ≈ $56k | ≈ $52k |
| $120,000 | EV: ≈ 2,110 used, rest unspent · risk-only: 3,000 | ≈ $54k | ≈ $39k |

- The planned budget is **binding**: about 1,055 more customers would still be worth an offer.
- The value curve is flat between roughly $60k and $80k. Going past the EV > 0 cut-off adds nothing, because the EV rule
  stops spending, while the risk-only rule keeps spending and loses value.
- At small budgets ($10–20k) both rules pick nearly the same customers and perform the same.
- **Recommendation:** run the planned $40k campaign now. Move toward $60–80k only after the pilot confirms a save rate of
  about 20% or more.

## 3. Who gets an offer

Under the base case, 199 of every 200 offers go to **month-to-month fiber-optic customers**: 67% of offers go to those in
their first 12 months, 15% to those at 13–24 months and 18% to those at 25+ months. High-risk customers with low bills
(mostly DSL or no-internet month-to-month customers) get **no** offer, because their expected saving is below $40.

| Risk tier (p) × value tier (V) | Contacted (test) | Why |
|---|---|---|
| High risk × High value | 91% | Clear win: EV ≈ $50 per offer |
| High risk × Medium value | 54% | Positive EV, but competes for budget |
| High risk × Low value | 0% | Likely to leave, but a $40 offer costs more than the expected saving |
| Medium risk × High value | 5% | Only the riskiest |
| Everyone else | 0% | EV < 0 |

The customer-level list is `targeting_output/tables/customer_targeting_table.csv` (or the dashboard's *Action list*). Each
row has a priority rank, p_churn, value, EV, the break-even probability, the scorecard points, the **three main reasons**
from the logistic regression (for example "internet: Fiber optic; contract: Month-to-month; short tenure (3 mo)") and a
suggested offer.

## 4. Matching the offer to the reason (segment offers)

The model's main churn drivers suggest offers that remove the reason for leaving instead of only paying the customer to stay:

| Segment (first rule that matches) | Main driver (EDA, scorecard) | Suggested offer | Illustrative cost / save rate |
|---|---|---|---|
| Month-to-month **and** Fiber optic | No commitment + expensive fiber (+29 points for fiber) | Discount for moving to a 1-year contract | $60 / 30% |
| Month-to-month **and** Electronic check | Manual payment (+14 points) | One-month credit for switching to autopay | $20 / 15% |
| Internet, no Online Security and no Tech Support | Missing protection add-ons | Free 3-month security / tech-support trial | $15 / 15% |
| Everyone else | — | Generic loyalty discount | $40 / 25% |

Under these **invented** costs and save rates, segment offers did *not* beat the single $40 offer: $6.3k vs $8.9k on test at
the same budget. A cheap offer only pays if its save rate stays high enough relative to its cost. The case for segment offers
is customer experience and fixing root causes, and their save rates must be measured in the pilot (§6) before use.
A 1-year contract also removes the "month-to-month" risk factor itself (−22 points on the scorecard).

## 5. Monthly operating steps

1. **Day 1. Score.** Refresh features for active customers and score them with `model_output/models/lr.joblib`
   (`model.predict_churn(df)`).
2. **Rank.** Compute V and EV with the current `config.yaml`, rank by EV, apply the budget and the EV > 0 stop
   (`src/targeting.py`).
3. **Apply guardrails** (below), then export the action list (dashboard → *Action list* → download).
4. **Hold out a random 10–20% of the list** (no offer) to keep measuring the save rate. This is what turns the assumption
   into a measurement.
5. **Contact** within the first week, using the suggested offer and the top reasons as the conversation script.
6. **Day 30. Measure.** Compare churn in contacted vs held-out customers. Update `save_rate` (per offer type) and
   `offer_cost` (actual redemption cost) in `config.yaml`.
7. **Every quarter.** Check calibration (predicted vs actual churn by decile). Retrain (`python main.py`) if ECE rises above
   about 0.05 or ROC-AUC drops below about 0.80.

## 6. Pilot design (before scaling)

The data has no past campaign, so the first campaign should be an experiment:

- **Population:** the top of the EV list (for example the first 1,000).
- **Randomize** within that list: half get the offer, half do not. Randomize within each offer type if testing segment
  offers.
- **Outcome:** churn within 30 and 90 days. Save rate = 1 − churn(offer) / churn(no offer).
- **Size:** with about 68% expected churn in the top of the list, **≈ 125 customers per arm** detect a 25% save rate,
  **≈ 340** a 15% save rate and **≈ 760** a 10% save rate (80% power, 5% two-sided). A 500 + 500 pilot therefore resolves
  the question that matters most: whether the save rate is above or below the ~10–15% where the campaign stops paying.
- **Decision rule:** if the measured save rate (lower CI) is above about 15% at a $40 offer, scale up toward the EV > 0
  cut-off. If it is below about 10%, stop or redesign the offer (sensitivity grid, notebook 04 §6).
- Later, the pilot data allows real **uplift modelling**: who changes behaviour because of the offer. That replaces the
  "offer helps every churner equally" assumption.

## 7. Guardrails: fairness and customer experience

| Risk | Observation (test) | Guardrail |
|---|---|---|
| Age | Seniors are contacted at 29% vs 11%, in line with churn (44% vs 23%). Recall among churners 43% vs 34% | Keep `SeniorCitizen` only as a deliberate choice: without it the gap narrows to 39% vs 35% at about 4% lower campaign value. Review per campaign |
| Gender | Not a model input. Contact rate 13.8% vs 14.5%, recall among churners 36% vs 38% | Keep it out of the model; monitor |
| Families | Customers with partner or dependents get fewer offers (they churn half as often) | Expected; monitor that precision stays similar (61–71% across groups) |
| Loyal customers | Long-tenure and 1–2-year contract customers almost never qualify | Offer non-price loyalty benefits (service check, upgrade priority) outside the EV budget, so loyalty is not punished |
| Offer fatigue / gaming | Repeated discounts can teach customers to threaten churn | Cap at one retention offer per customer per 6 months; prefer contract or service offers over pure discounts |
| Explanations | Reasons are associations, not causes | Agents use them as conversation openers, not as statements about the customer |

## 8. What would change the plan

- **Measured save rate < 10%** at $40: the campaign does not pay. Stop, or test cheaper, targeted offers.
- **Save rate ≥ 30%:** expand toward the EV > 0 cut-off. The planned $40k campaign is worth ≈ $61k at a 30% save rate and ≈ $95k at 40%, and more customers qualify.
- **Real customer lifetime longer than 12 months:** value per saved customer rises. Raise `retention_horizon_months`, and
  more customers qualify.
- **New data with dates:** re-validate on the next month (temporal holdout) before trusting the probabilities.
