# Method notes

The places where a reasonable analyst could have done something different, and
why this version does what it does.

---

## The problem underneath everything: the outcome barely varies

The mean Seattle listing rating is **94.5 / 100**. The interquartile range is
**93 to 99**. **86.6%** of rated listings score 90 or better and **24.6%**
score exactly 100. Every one of Airbnb's six subscores has a median of 10 out
of 10, and between **46.5%** (value) and **70.7%** (communication) of listings
score a perfect 10 on each.

Three consequences run through the whole project.

1. **Ceiling effects flatten real differences.** A listing that is genuinely
   better than another may score identically because both are at 10. Any
   estimated coefficient is therefore an underestimate of the underlying
   quality difference, in an amount nothing here can quantify.
2. **The variance being explained is small in absolute terms.** The model
   reaches an out-of-sample R² of 0.40 on a variable with a standard deviation
   of 4.1 points. Explaining 40% of a four-point spread is worth doing and is
   not the same claim as "we can predict guest satisfaction".
3. **A second outcome was added because of it.** Price premium against a
   comparable set has a standard deviation of 0.32 in logs — roughly a **37%**
   spread — and no ceiling. It is the outcome that tells you whether the
   rating findings have commercial consequences. The answer turns out to be
   mostly no, which is more useful than a second confirmation would have been.

---

## Defining the eight aspects

Six of the eight are the constructs Airbnb itself measures: cleanliness,
location, communication, check-in, value, accuracy. That choice is not
convenience. It means six of the eight can be validated against an independent
measurement collected the same day from the same guests through a different
instrument.

Two more — **noise / disturbance** and **space / comfort** — were added because
they dominate the mined vocabulary and have no subscore. Nothing else in the
dataset can see them, and space and comfort turns out to be the single largest
driver in the study, so leaving them out would have removed the main finding.

**Where the surface forms came from.** A 6,000-review sample was dependency
parsed with spaCy `en_core_web_sm`; noun-chunk heads and the adjectives
attached to them were counted; heads appearing 40+ times were reviewed by hand
and assigned to aspects. `aspects.py` re-runs that mining pass on every
execution and writes the result to `outputs/tables/lexicon_discovery.csv`, so
the lexicon's recall is auditable rather than asserted. The audit is scored
only against **non-generic** heads: "place", "stay" and "apartment" are the
three commonest nouns in the corpus and none of them names an attribute, so
counting them as misses would understate recall by construction. Current
coverage is **53% of non-generic mined terms, 74% weighted by frequency**.

The audit fed back exactly once. The first draft missed *view, distance, walk,
shop, street, access, block, minute, town, city* (all location) and *privacy,
window, door, breakfast, garden, deck, patio* (all space and comfort). Those
were added; the file records why. The lexicon is otherwise frozen, because an
analysis whose feature definitions silently re-derive themselves on every run
is not reproducible.

**Two deliberate design choices.** The lexicon carries no polarity — "dirty"
and "spotless" are both cleanliness cues, and deciding which is a complaint is
VADER's job, not the lexicon's. And aspects are allowed to overlap: "quiet
neighbourhood" is a noise cue and a location cue and should count for both.

---

## Scoring: sentences, not reviews

Each review is split into sentences (newlines count as terminators — guests use
them as punctuation), each sentence is matched against the eight patterns, and
**only the matching sentences** are scored with VADER. This is the point of
aspect-based sentiment: *"The flat was spotless but the street was deafening"*
must come out positive on cleanliness and negative on noise, and a
whole-review score cannot express that.

VADER's published thresholds (±0.05) are used unchanged, so they cannot have
been tuned to flatter the result. 63.4% of sentences carry at least one aspect;
2.3% of scoreable reviews carry none.

**A known blind spot, pinned by a test.** VADER scores *"Exactly as described,
the photos were completely accurate"* at **0.0**: neither *described* nor
*accurate* is in its lexicon, because VADER was built for affective language
and accuracy is a factual claim. The negative direction still works
(*misleading* is scored). The accuracy aspect is therefore systematically
better at detecting complaints than compliments, and that asymmetry is the
most plausible reason its agreement with Airbnb's accuracy subscore is among
the weakest of the six. `tests/test_aspects.py` fails if this stops being true,
so the caveat cannot quietly go stale.

---

## Validating the extraction

Six aspects × two statistics, against Airbnb's own subscores, on listings with
10+ scoreable reviews and 2+ mentions of the aspect:

| Aspect | Pearson r | Spearman rho | n | Complaint-rate r |
|---|---|---|---|---|
| Location | 0.41 | 0.39 | 1,828 | −0.18 |
| Cleanliness | 0.40 | 0.26 | 1,791 | **−0.40** |
| Host communication | 0.24 | 0.22 | 1,826 | −0.19 |
| Accuracy | 0.19 | 0.19 | 1,600 | −0.18 |
| Value | 0.17 | 0.17 | 1,013 | −0.10 |
| Check-in | 0.17 | 0.16 | 1,690 | −0.18 |

Read honestly: **the extraction agrees with Airbnb's own instrument, weakly.**
Mean r = 0.27. The best pair leaves 83% of the subscore unexplained.

Three things worth separating out of that. First, the subscore is not ground
truth — it is another compressed, heaped measurement of the same latent thing,
and a perfect extractor could not correlate at 1.0 with it. Second, the
**complaint rate is a better signal than mean sentiment for cleanliness**
(−0.40 against +0.40, and negative complaints are the rarer, more informative
event). Third, the two aspects with no subscore — noise and space/comfort —
cannot be validated this way at all, and space/comfort is the headline result.
That is the weakest link in the chain and it should be named as such: **the
largest effect in this study rests on an extraction that could not be
externally checked.**

---

## Cleaning

Full log in `outputs/tables/cleaning_log.csv`; **2.0%** of raw review rows are
removed.

| Rule | Rows | Why |
|---|---|---|
| Automated cancellation notices | 805 | Airbnb writes these itself. They are fluent English, they name the host, and VADER reads them as mildly negative — leaving them in would attach a host-communication complaint to every listing that ever had a cancellation |
| Non-English | 839 | The lexicon and VADER are English-only; a Spanish review would score as neutral-with-no-mentions and dilute every mean |
| Empty comment | 18 | A review event with no opinion attached |
| Duplicate reviewer/listing/text | 19 | Double-posts |
| Very short (< 20 chars) | *kept, flagged* | "Great!" is a real review and belongs in the volume counts; it has no aspect content to extract, so it is excluded from scoring only |

**The cancellation filter is anchored to the start of the review on purpose.**
Unanchored, it would also delete the **91** surviving reviews that mention a
cancellation in passing — *"our original bnb was cancelled at the last minute, Karina took
us in"* — which is guest opinion and belongs in the analysis. A test caught two
surviving notices reading *"The host canceled **my** reservation…"* when the
first draft only matched *"this reservation"*; the pattern now covers
this/my/our/the.

### The language screen

A two-part heuristic, not a language classifier.

1. **Script.** Below 90% ASCII characters the review is not in the Latin
   alphabet — Chinese, Korean, Japanese, Russian, Greek, Hebrew, Arabic.
2. **Function words.** Latin-script languages pass the script test, so they are
   caught by counting function words on both sides. A review is rejected when
   it carries 3+ distinct non-English function words *and* more of them than
   English ones.

**What it deliberately does not do is require English function words to be
present.** That was the first design, and on top of the current screen it
would discard a further **1,367** surviving reviews — *"Great location, comfy
bed - easy check-in"* — which are exactly the reviews with the highest density
of aspect terms per word. Losing them would have biased the extraction towards
long, discursive reviewers. The counterfactual is recomputed on every run and
logged in `cleaning_log.csv`, so the rejected design stays costed rather than
remembered.

Recall is measured, not assumed: **51 surviving reviews (0.06%)** carry a
non-English function word and no English one. That is an upper bound on one
failure mode, not a full error rate — a short foreign review with no function
words at all is invisible to it.

---

## Why the outcome variables are what they are

**Outcome 1, `review_scores_rating`.** The thing the platform ranks on and the
thing a manager is judged by. Its problem is compression, discussed above.

**Outcome 2, price premium against a comparable set.** Comp set is same
`neighbourhood_group_cleansed` × `room_type` × capacity band (1-2, 3-4, 5-6,
7+). Premium is `log(price) − median(log(price) of the peers)`.

The **leave-one-out** part is the point, and it is what
`tests/test_drivers.py` is mostly about. With a plain group median, a comp set
of one listing produces a premium of exactly 0.0 — not "priced at the market"
but "there is no market here to compare to". Those rows would then be
indistinguishable from genuinely average listings and would pull every
coefficient towards zero. Here the listing is removed from its own benchmark,
and any comp set left with fewer than **8** peers returns NaN and drops out.
That costs **4.7%** of listings.

Price is what the host *asked* on the scrape date, not what they achieved.
Occupancy, discounting and length-of-stay effects are invisible.

---

## Modelling choices

**Features.** Two per aspect: mean sentiment across the listing's mentions
(undefined below 2 mentions, then mean-imputed — which after standardisation
is zero, so the coefficient is estimated off listings that actually have the
measurement), and mention rate, which is always defined and carries the
information the imputation discards. Coverage of the sentiment feature ranges
from 100% (space/comfort, location) to **55% (value)**; value is the one
aspect where imputation is doing heavy lifting, and it is also the one whose
importance interval crosses zero.

**Controls.** Room type, log capacity, bedrooms, neighbourhood group, log
review count, host tenure, amenity count, minimum nights, and log price for the
rating model. Neighbourhood dummies are *dropped* from the premium model
because the comp set already differences them out.

`host_is_superhost` is deliberately **not** a control. Superhost status is
awarded partly on the basis of ratings, so conditioning on it would be
conditioning on a consequence of the outcome.

**Two model families, both reported.** RidgeCV on standardised features with
95% bootstrap intervals over 400 resamples of the training rows (alpha held at
the cross-validated value inside the bootstrap, so the intervals carry sampling
noise rather than tuning noise), and LightGBM with early stopping. Both are
scored on the same held-out 30%.

**An unregularised cross-check.** Ridge shrinks coefficients towards zero, and
shrinkage is exactly the kind of thing that can manufacture a tidy ranking. The
same design is refit with OLS and HC3 robust standard errors: sign agreement on
the aspect block is **94%** and rank agreement is **0.96**. The ranking is in
the data, not in the penalty.

**Sensitivity to the review-count threshold.** `MIN_REVIEWS_FOR_MODEL = 10` is
a judgement call, so it is swept rather than defended:

| Threshold | Listings | R² (controls only) | R² (full) | Largest aspect effect |
|---|---|---|---|---|
| 5 | 2,315 | 0.153 | 0.416 | space/comfort, +0.90 |
| **10** | **1,828** | **0.085** | **0.396** | **space/comfort, +0.92** |
| 20 | 1,182 | 0.114 | 0.534 | space/comfort, +0.86 |

The headline attribute and its magnitude are stable across all three. R² rises
with the threshold because listing-level means get less noisy, which is what
should happen.

---

## The confound I cannot rule out

Stated once, properly, because it applies to every coefficient in the repo.

A host who furnishes a flat well is also, on average, a host who writes an
honest listing, replies within the hour, leaves the wifi password on the table
and prices sensibly. **Host conscientiousness is a common cause of the
predictor and the outcome**, and it is unobserved. "Space and comfort raises
the rating" and "conscientious hosts do everything well, and furnishing is the
part guests happen to write down" fit these data identically.

The controls help at the margin — property type, size, area, price and tenure
absorb some of it — and they cannot solve it. Neither can more features drawn
from the same reviews, since the reviews are downstream of the same host.

What would separate them: a manager who upgrades beds and bedding in a randomly
chosen half of a portfolio and reads the rating six months later. That is a
tractable experiment and it is the correct next step, not a better model.

The same caution applies to the priority matrix. It is a ranking device built
on associations. It says where to look first; it does not promise a return.

---

## The importance–performance matrix

**Importance** is the Ridge coefficient on the aspect's sentiment: rating
points per one standard deviation, holding the controls fixed.

**Performance** is **complaints per review**, not complaints per mention. The
per-mention rate ranks value for money as the worst attribute in the portfolio
(6.9% of value mentions are negative) purely because nobody mentions price
unless something is wrong with it — value is raised in only 7.4% of reviews and
is in fact one of the rarest complaints a guest actually makes (0.51 per 100
reviews). Per-review incidence is the operationally meaningful quantity and is
not confounded by salience.

Quadrants split at the **median** of each axis across the eight attributes. Any
split is arbitrary; a median split is the version that cannot be tuned to put a
favoured attribute in a favoured box. Attributes whose importance interval
crosses zero are drawn hollow and marked, and their quadrant is not treated as
a finding.

**Size of the prize** is the gap from the 25th percentile to the median of an
attribute's sentiment, in standard deviations, times that attribute's
coefficient. It is arithmetic on the coefficients and inherits every one of
their assumptions, including the causal one it cannot support.

---

## Geography

Reported at the 17-area `neighbourhood_group_cleansed` level rather than the
87-way `neighbourhood_cleansed`, and the reason is measured rather than
asserted: the **median neighbourhood standard error is 2.0× the area figure**,
and only **23 of 85** neighbourhoods in the modelling frame clear 25 listings
at all. A table of "best and worst neighbourhoods" at the fine level would
mostly be a table of which small areas got lucky.

Even at the area level, only **29% of the 128 area-by-attribute cells** differ
from the city by more than two standard errors, and only those are marked. One
area (Interbay, 5 listings) falls below the minimum and is dropped.

**"Other neighborhoods" is excluded from every best/worst claim.** It is Inside
Airbnb's residual bucket — a fifth of the portfolio and not a place. It is
shown on the map, flagged as a catch-all, and never named as an area to buy in.

---

## Topic modelling

Two passes. A **pooled NMF** across both rating bands so topics are comparable
and prevalence can be differenced, and a **separate NMF on the lower band
only**, because a pooled model is dominated by the praise that makes up most of
the corpus and can miss structure that exists only in the complaint tail.

Ten components, `nndsvd` initialisation for determinism, unigrams and bigrams
(*"walking distance"*, *"easy access"* and *"capitol hill"* are single concepts
that unigrams shred). **Personal names are stop-listed** — every reviewer and
host first name, 11,807 tokens — because a topic model left to itself will
happily produce a topic that is a list of Seattle hosts.

**Topics are named by rule, and the rules are checked against the fit.** Each
component is scored against a fixed anchor-term list and the best global
assignment wins; a component matching no rule on at least two anchors keeps its
own top three terms as its label. A topic model does not owe anyone an
interpretable topic, and papering over that with a confident name is the
standard way this technique gets oversold.

The term-level comparison uses the **weighted log-odds ratio with an
informative Dirichlet prior** (Monroe, Colaresi and Quinn, 2008) rather than a
frequency difference, which would return the words that are common in both
corpora.

One confound is visible in the output and stated on the chart: *apartment*,
*building* and *cottage* are property types, and property type is correlated
with rating. Part of the "objects versus feelings" contrast is a property-mix
effect.

---

## What a second pass should add

- **An intervention.** Everything above is cross-sectional. Randomising a
  furnishing upgrade across a portfolio would convert the largest coefficient
  in the study from an association into an effect. Nothing else on this list
  matters as much.
- **A supervised aspect classifier instead of a lexicon.** A few thousand
  hand-labelled sentences would give measurable precision and recall per
  aspect, replacing the current recall audit — which measures vocabulary
  coverage, not extraction accuracy — with a real error rate.
- **A domain sentiment model.** VADER's blindness to factual praise is a known,
  measurable weakness here. A model fine-tuned on hospitality review text would
  fix the accuracy aspect and probably several others.
- **Multi-city, multi-vintage.** One city and one scrape cannot separate what is
  true of guests from what was true of Seattle in 2016. The same pipeline
  across five cities and three years would.
- **Occupancy, not asking price.** `calendar.csv` ships with this scrape and
  carries booked-out dates. Revenue per available night is the outcome a
  manager actually optimises, and it is one join away.
