# Executive summary

**To:** Portfolio director, short-let operations
**Re:** Where the next improvement budget should go, and what it will and will not buy
**Evidence:** 84,849 guest reviews and 3,818 listings, Seattle, June 2009 – January 2016

---

## Recommendation

**Fund space and comfort — beds, bathrooms, heating, furnishing — and stop
funding cleanliness and host responsiveness, which are already working. Expect
the return in ranking and occupancy, not in nightly rate.**

Space and comfort carries the largest association with the rating in the
portfolio: **+0.92 rating points per standard deviation** of guest sentiment,
1.4× cleanliness and 3.4× location, holding property type, size, area, price,
review count and host tenure fixed. It is also one of only two attributes with
an above-median complaint rate. It is the one place where importance and
current weakness overlap.

---

## Why

**There are almost no failures left to remove.** The worst-performing attribute
in the whole portfolio draws a complaint in **1.6 of every 100 reviews**.
Cleanliness draws 0.5. Host communication draws 0.8. A programme built on
eliminating complaints is a programme built on an event that has already
stopped happening — and the reason is partly that guests who had a bad stay
mostly do not review at all, so even these rates are a floor.

**The scoreboard is nearly saturated.** The average listing scores **94.5 out
of 100**; **86.6%** are above 90 and a quarter score exactly 100. Every Airbnb
subscore has a median of 10/10. What separates a 98 from a 93 is not the
absence of problems — it is what guests find when nothing is wrong.

**And the text says what that is.** Reviews of top-quartile listings are
**1.69× more** about *feeling at home and being made welcome* — the vocabulary
is *cottage, beautiful, wonderful, loved, fresh, welcome, decorated, wine,
coffee, fruit, snacks*. Reviews of bottom-quartile listings run on *problem,
issues, noisy, loud, sleep, wasn't, didn't, price*. Bottom-quartile stays are
written up in objects and faults; top-quartile stays are written up in feelings
and gestures.

**Modelling both together, the ranking is stable.** Review text lifts
out-of-sample R² on the rating from **0.085 to 0.396** — gradient boosting
independently reaches 0.387 — and an unregularised refit agrees with the
regularised ranking at a rank correlation of **0.96**.

## The uncomfortable part

**None of this reaches price.** Against a comparable set — same neighbourhood
group, room type and capacity band, with the listing excluded from its own
benchmark — review text moves R² from **0.010 to 0.042**, and the gradient
boosted model gets *worse* with the text in, from 0.093 to 0.068. The largest
text association with price is **5.7% per standard deviation**, against a
comp-set price spread of **37%**.

So the business case cannot be "better reviews, higher rate". It has to be
"better reviews, better ranking, higher occupancy" — and this dataset cannot
measure occupancy. That is a gap in the case, and it should be closed before
capital is committed, not after.

**The whole prize is also modest.** Lifting a bottom-quartile listing to the
portfolio median on **every** attribute is worth **1.89 rating points** on a
base of 94.5. Real, worth having, and considerably smaller than most
review-optimisation pitches imply.

## What would change the answer

| If this is true | The recommendation moves |
|---|---|
| Better hosts simply do everything well, and furnishing is only what guests write down | The whole ranking is a proxy for host conscientiousness. Spend on host selection and training, not on furniture. **This is the confound I cannot rule out.** |
| Ranking and occupancy do not respond to a 1–2 point rating gain | The programme has no financial return at all. Test this first: it is the load-bearing assumption |
| Guests who had bad stays review at anything like normal rates | Complaint rates are real rather than a floor, and the "fix failures" strategy comes back into play |
| The portfolio is concentrated in the University District or Downtown | Attribute-level spending is the wrong lever. Both sit significantly below the city on 6 and 5 of 8 attributes respectively; that is an acquisition problem |
| The category is 2024 professionalised short-let, not 2016 Seattle | Every coefficient needs rebuilding. The method transfers; the numbers do not |

## Next three steps

1. **Randomise a furnishing upgrade across the portfolio and read the rating in
   six months.** Everything above is a cross-sectional association. This is the
   one action that converts the largest number in the study into something a
   board can act on, and it is cheap relative to the spend it would authorise.
2. **Join occupancy to the analysis.** `calendar.csv` ships with this dataset
   and carries booked-out dates. Revenue per available night — not asking price
   — is the outcome the recommendation actually rests on, and it is one join
   away.
3. **Hand-label 2,000 sentences.** The aspect extractor agrees with Airbnb's own
   subscores at a mean correlation of 0.27, which is real and weak, and the two
   attributes with no subscore — including the headline one — cannot be checked
   against anything at all. A labelled set turns "we think this measures space
   and comfort" into a precision and recall number.
