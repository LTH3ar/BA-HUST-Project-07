# Executive brief: who should get a retention offer?

*BA-HUST Capstone 2026, topic 7. Data: IBM Telco Customer Churn sample (fictional telco, 7,043 customers).*

## The decision

Each month the retention team has a fixed budget for retention offers. The question is **which customers get one, and how
many**. Sending offers to the wrong people wastes money: customers who would have stayed anyway still cost the offer, and
customers who pay little are not worth an expensive offer even if they are likely to leave.

## Recommendation

1. **Pick customers by expected value, not by risk alone.** For each customer, multiply the chance they leave (from the churn
   model) by what keeping them is worth and by the share of leavers an offer keeps, then subtract the offer cost. Contact
   customers in that order, and stop when the next one is no longer worth it or the budget runs out.
2. **At the planned $40k budget, contact the top 1,000 customers on the list.** Almost all of them are month-to-month fiber
   customers, two thirds in their first year with the company.
3. **Consider a budget of about $80k.** Around 2,060 customers are worth an offer under our assumptions. Going beyond that
   point loses money, and the list tells you where it is.
4. **Before rolling out, run a small randomized pilot** to measure how many at-risk customers an offer really keeps. That one
   number decides whether the campaign pays, and the data cannot tell us.

## Expected impact (and how sure we are)

Measured on 1,409 customers the analysis never saw while it was being built, then scaled to the whole base:

| Approach, same $40k budget | Expected net value | Plausible range | Share of valuable leavers reached |
|---|---|---|---|
| **Expected value (recommended)** | **≈ $44k** | $35k – $52k | **53%** |
| Highest churn risk only | ≈ $41k | $33k – $47k | 35% |
| Rule from the data exploration (month-to-month fiber, newest first) | ≈ $38.5k | $31k – $47k | 30% |
| Highest-paying customers only | ≈ $3k | −$7k – $12k | 56% |
| Random selection | ≈ −$12k (loss) | | 14% |

- The recommended approach earns somewhat more than "highest risk only" (about +$4k; we cannot rule out a tie at this budget).
  Its clearer advantage is that it **reaches about 50% more of the high-value customers who leave** and **knows when to stop
  spending**. At bigger budgets the gap widens: at $120k it earns about $54k against $39k.
- About 7 in 10 contacted customers would actually have left (versus 1 in 4 at random).
- The churn model is reliable for this purpose. It separates leavers from stayers well (ROC-AUC 0.85), and its probabilities
  are accurate (a predicted 60% means about 60% leave), which the value calculation depends on.

**These are scenarios, not forecasts.** They assume an offer costs $40, keeps 25% of the customers who would have left, and
that a kept customer stays 12 months at a 50% margin. Change those and the result changes a lot:

| If… | Net value of the $40k campaign |
|---|---|
| the offer keeps 10% of leavers instead of 25% | ≈ $1k (barely breaks even) |
| the offer keeps 40% | ≈ $95k |
| the offer costs $80 instead of $40 | ≈ $5k |
| the offer costs $20 | ≈ $96k |
| a kept customer stays 6 months instead of 12 | ≈ $4k |
| a kept customer stays 24 months | ≈ $121k |

The **order** of the list does not change with these assumptions; only how far down it is worth going. If offers turn out
to work rarely (for example 5% saved at $30 or more per offer), the method contacts nobody rather than losing money.

## Risks

- **The save rate is unknown.** The data has no past campaign. Everything hinges on an assumed 25%. → Pilot first (see the
  playbook): offer to a random half of the selected customers, hold back the other half, compare churn after a month.
- **Paying customers who would have stayed.** About 3 in 10 offers go to customers who were not going to leave. This cost is
  included in the figures, but a generous offer can also train customers to threaten to leave.
- **Fairness and customer experience.** Senior citizens receive offers more often (29% vs 11%) because they leave more often
  (44% vs 23%). This is consistent, not arbitrary, but it should be a deliberate choice; removing age from the model would cost
  about 4% of campaign value. Gender is not used. Loyal long-term customers rarely get offers, which can feel unfair if they
  notice. The playbook proposes non-price offers and a contact cap.
- **Offer mismatch.** A discount does not fix the reasons customers leave (no tech support, expensive fiber, manual
  payments). Matching the offer to the reason is likely to work better, but the save rates for such offers must also be
  measured.

## Limitations

- **Teaching data.** The telco is fictional and its monthly churn (26.5%) is far above real operators. The method transfers;
  the dollar figures do not.
- **One snapshot, no dates.** We could not test the model on a later month. A real deployment should be re-validated on the
  next month's data.
- **Simple value measure.** Value = monthly bill × margin over a 12-month horizon. It ignores cross-selling, price changes and
  the cost of winning a new customer, all of which would make retention look more valuable.
- **Correlation, not causation.** The model finds who is likely to leave, not why, and not who an offer would change. The
  reasons shown for each customer are statistical associations.

*Supporting material: README (method, reproduction), `notebooks/04_targeting.ipynb` (analysis), `dashboard/app.py` (try other
assumptions), `targeting_output/tables/customer_targeting_table.csv` (the list).*
