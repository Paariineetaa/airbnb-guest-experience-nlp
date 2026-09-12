"""
Step 5. What is said, and what is said differently when the rating is low.

Two passes over the same corpus:

1. **Pooled NMF.** One topic model on reviews from both rating bands, so
   prevalence can be differenced. Behind the prevalence chart.
2. **Per-band NMF.** A separate model on low-rated listings' reviews only. A
   pooled model is dominated by the 87% of the corpus that is praise and
   misses structure in the complaint tail.

Then a term-level comparison independent of either model: the weighted
log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi and
Quinn, 2008), which finds the words distinguishing the two corpora rather than
the words common to both.

Personal names are stripped. Reviews are full of "Kelly was wonderful", and an
unconstrained topic model produces a topic that is a list of Seattle hosts.

Outputs
  figures/09_topic_prevalence.png
  figures/10_complaint_language.png
  tables/topics_pooled.csv
  tables/topics_low_band.csv
  tables/distinguishing_terms.csv
  topic_metrics.json
"""
from __future__ import annotations

import json
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

import config as C
import viz

viz.use_style()

# Words that are everywhere in this corpus and carry no information about what
# a stay was like. "seattle" is in a third of reviews. "airbnb" and "host" are
# structural. Removing them stops every topic from being about Seattle.
EXTRA_STOPWORDS = {
    "seattle", "airbnb", "air", "bnb", "place", "stay", "stayed", "staying",
    "host", "hosts", "hosted", "guest", "guests", "definitely", "really",
    "just", "great", "good", "nice", "did", "didn", "don", "ve", "ll",
    "does", "got", "went", "come", "came", "make", "made", "want", "wanted",
    "thank", "thanks", "recommend", "recommended", "highly", "time", "times",
    "day", "days", "night", "nights", "week", "weekend", "trip", "visit",
    "lot", "lots", "bit", "little", "way", "thing", "things", "like",
    "would", "could", "also", "us", "we", "our", "my", "very", "quite",
}

# Topic naming. Each rule is (name, anchor terms). A fitted topic takes the
# name of the rule matching the most anchors in its top terms. If nothing
# matches, it is named from its own top three terms. A hand-written label
# cannot then survive a change in the model it describes.
TOPIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Neighbourhood, transit and errands",
     ("downtown", "bus", "parking", "close", "area", "walk", "grocery",
      "transit", "car", "light rail")),
    ("Walking distance to bars and restaurants",
     ("walking", "distance", "walking distance", "restaurants", "bars",
      "shops", "coffee", "market")),
    ("Capitol Hill and the nightlife",
     ("capitol", "capitol hill", "hill", "capital", "capital hill",
      "pike", "belltown")),
    ("A quiet, safe street",
     ("quiet", "quiet neighborhood", "neighborhood quiet", "safe",
      "peaceful", "residential")),
    ("The apartment and its condition",
     ("apartment", "apartment clean", "apartment location", "located",
      "building", "view", "condo", "unit")),
    ("The house, the garden and the space",
     ("house", "house clean", "beautiful house", "house beautiful",
      "garden", "yard", "backyard", "porch")),
    ("The room, the bed and the bathroom",
     ("room", "room clean", "bed", "bathroom", "private", "bed comfortable",
      "shower", "towels")),
    ("Easy check-in and access",
     ("easy", "check", "access", "easy access", "easy check", "check check",
      "communicate", "described", "instructions", "key")),
    ("Feeling at home and made welcome",
     ("home", "feel", "felt", "welcome", "feel welcome", "lovely",
      "wonderful", "welcoming", "hospitality")),
    ("Perfect - nothing to add",
     ("perfect", "perfect location", "location perfect", "loved", "spot",
      "absolutely", "needed", "flawless")),
    ("Host responsiveness and helpfulness",
     ("helpful", "friendly", "accommodating", "responsive", "questions",
      "friendly helpful", "extremely", "kind", "communication")),
]

_TOKEN = re.compile(r"[a-z][a-z']+")


def name_stopwords(reviews: pd.DataFrame, listings: pd.DataFrame) -> set[str]:
    """
    Every personal name in the data, lower-cased.

    Both sides are needed. Reviewer names cover most common first names, but
    hosts are the ones written *about*, and an unusual host name in 200
    reviews of one listing is a term a topic model latches on to.
    """
    names: set[str] = set()
    for s in (reviews["reviewer_name"], listings["host_name"]):
        names |= {w for v in s.dropna().astype(str).unique()
                  for w in _TOKEN.findall(v.lower()) if len(w) > 2}
    return names


TOKEN_PATTERN = r"(?u)\b[a-z][a-z']{2,}\b"


_TOKENISE = re.compile(r"\b[a-z][a-z']{2,}\b")


def stop_list(extra_names: set[str]) -> list[str]:
    """
    Stop words, passed through the same tokeniser the vectoriser uses.

    An entry like "persons'" (a reviewer name in the data) goes into the list
    intact but comes out of the analyser as "persons", which is then not
    stopped. sklearn warns about the inconsistency.
    """
    raw = set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS | extra_names
    return sorted({t for w in raw for t in _TOKENISE.findall(w)})


def vectoriser(stopwords: set[str]) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=C.TOPIC_MAX_FEATURES,
        stop_words=stop_list(stopwords),
        token_pattern=TOKEN_PATTERN, lowercase=True,
        min_df=C.TOPIC_MIN_DF, max_df=C.TOPIC_MAX_DF,
        # Bigrams: "walking distance", "easy access" and "capitol hill" are
        # single concepts that unigrams shred.
        ngram_range=(1, 2), sublinear_tf=True,
    )


def top_terms(model: NMF, vocab: np.ndarray, k: int = 12) -> list[list[str]]:
    return [list(vocab[np.argsort(comp)[::-1][:k]]) for comp in model.components_]


def label_topics(terms: list[list[str]]) -> list[str]:
    """
    Give each fitted topic a readable name, or leave it without one.

    Names are matched globally, not topic-by-topic. Every (topic, rule) pair
    is scored by how many of the rule's anchor terms appear in the topic's top
    terms, and the strongest pairs are assigned first. A topic matching no
    rule on two anchors keeps its own top three terms as its label.
    """
    pairs = sorted(
        ((sum(a in tt for a in anchors), ti, name)
         for ti, tt in enumerate(terms)
         for name, anchors in TOPIC_RULES),
        reverse=True)
    labels: list[str | None] = [None] * len(terms)
    used: set[str] = set()
    for score, ti, name in pairs:
        if score < 2 or labels[ti] is not None or name in used:
            continue
        labels[ti] = name
        used.add(name)
    return [lab if lab else ", ".join(terms[i][:3])
            for i, lab in enumerate(labels)]


def log_odds(low_counts: np.ndarray, high_counts: np.ndarray,
             prior: np.ndarray) -> np.ndarray:
    """
    Weighted log-odds ratio, informative Dirichlet prior (Monroe et al. 2008).

    A raw frequency difference between the bands returns "the", "room" and
    "nice". This measure divides by the standard error of the log-odds, so a
    term has to be differentially used and used often enough to count.
    """
    a0 = prior.sum()
    n_l, n_h = low_counts.sum(), high_counts.sum()
    d_l = np.log((low_counts + prior) / (n_l + a0 - low_counts - prior))
    d_h = np.log((high_counts + prior) / (n_h + a0 - high_counts - prior))
    var = 1.0 / (low_counts + prior) + 1.0 / (high_counts + prior)
    return (d_l - d_h) / np.sqrt(var)


def main() -> None:
    reviews = pd.read_parquet(C.REVIEWS_PARQUET)
    listings = pd.read_parquet(C.LISTINGS_PARQUET)
    frame = pd.read_parquet(C.MODEL_FRAME_PARQUET)

    rated = frame[frame["review_scores_rating"].notna()]
    lo_cut = float(np.percentile(rated["review_scores_rating"], C.LOW_BAND_PCTL))
    hi_cut = float(np.percentile(rated["review_scores_rating"], C.HIGH_BAND_PCTL))
    low_ids = set(rated.loc[rated["review_scores_rating"] <= lo_cut, "id"])
    high_ids = set(rated.loc[rated["review_scores_rating"] >= hi_cut, "id"])
    print(f"Rating bands: lower <= {lo_cut:.0f} ({len(low_ids):,} listings), "
          f"upper >= {hi_cut:.0f} ({len(high_ids):,} listings)")

    rv = reviews[~reviews["is_short"]]
    rng = np.random.default_rng(C.RANDOM_STATE)

    def band_docs(ids: set[int]) -> pd.Series:
        d = rv.loc[rv["listing_id"].isin(ids), "comments"]
        if len(d) > C.TOPIC_MAX_DOCS_PER_BAND:
            d = d.iloc[rng.choice(len(d), C.TOPIC_MAX_DOCS_PER_BAND, replace=False)]
        return d

    low_docs, high_docs = band_docs(low_ids), band_docs(high_ids)
    print(f"  {len(low_docs):,} lower-band reviews, {len(high_docs):,} upper-band")

    stop_names = name_stopwords(reviews, listings)
    vec = vectoriser(stop_names)
    docs = pd.concat([low_docs, high_docs])
    band = np.r_[np.zeros(len(low_docs)), np.ones(len(high_docs))]
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()

    # nndsvd init makes the factorisation deterministic, which matters when
    # topic labels are matched to components by rule.
    nmf = NMF(n_components=C.N_TOPICS, init="nndsvd", max_iter=900, tol=1e-5,
              random_state=C.RANDOM_STATE, l1_ratio=0.5)
    W = nmf.fit_transform(X)
    terms = top_terms(nmf, vocab)
    labels = label_topics(terms)

    # Prevalence = share of a review's topic weight, averaged over reviews.
    share = W / np.clip(W.sum(axis=1, keepdims=True), 1e-9, None)
    prev = pd.DataFrame({
        "topic": labels,
        "terms": [", ".join(t[:8]) for t in terms],
        "prev_low": share[band == 0].mean(axis=0),
        "prev_high": share[band == 1].mean(axis=0),
    })
    prev["lift_low_vs_high"] = prev["prev_low"] / prev["prev_high"]
    prev.sort_values("lift_low_vs_high", ascending=False).to_csv(
        C.TAB / "topics_pooled.csv", index=False)
    print(prev.sort_values("lift_low_vs_high", ascending=False)
          [["topic", "prev_low", "prev_high", "lift_low_vs_high"]].to_string(index=False))

    # separate model on the complaint tail
    vec_lo = vectoriser(stop_names)
    X_lo = vec_lo.fit_transform(low_docs)
    nmf_lo = NMF(n_components=C.N_TOPICS, init="nndsvd", max_iter=900, tol=1e-5,
                 random_state=C.RANDOM_STATE, l1_ratio=0.5)
    W_lo = nmf_lo.fit_transform(X_lo)
    terms_lo = top_terms(nmf_lo, vec_lo.get_feature_names_out())
    labels_lo = label_topics(terms_lo)
    share_lo = W_lo / np.clip(W_lo.sum(axis=1, keepdims=True), 1e-9, None)
    pd.DataFrame({"topic": labels_lo,
                  "terms": [", ".join(t) for t in terms_lo],
                  "prevalence": share_lo.mean(axis=0)}).to_csv(
        C.TAB / "topics_low_band.csv", index=False)

    # distinguishing terms
    cnt = TfidfVectorizer(
        stop_words=stop_list(stop_names), token_pattern=TOKEN_PATTERN,
        min_df=C.TOPIC_MIN_DF, max_df=0.5,
        use_idf=False, norm=None, binary=False, sublinear_tf=False)
    Xc = cnt.fit_transform(docs)
    vocab_c = cnt.get_feature_names_out()
    lo_c = np.asarray(Xc[band == 0].sum(axis=0)).ravel()
    hi_c = np.asarray(Xc[band == 1].sum(axis=0)).ravel()
    total = lo_c + hi_c
    z = log_odds(lo_c, hi_c, total / total.sum() * 500)
    dt = (pd.DataFrame({"term": vocab_c, "z": z, "n_lower": lo_c, "n_upper": hi_c})
          .sort_values("z"))
    dt.to_csv(C.TAB / "distinguishing_terms.csv", index=False)

    # fig 9
    p = prev.sort_values("lift_low_vs_high")
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    y = np.arange(len(p))
    ax.barh(y - 0.185, p["prev_high"], height=0.33, color=viz.BLUE)
    ax.barh(y + 0.185, p["prev_low"], height=0.33, color=viz.ORANGE)
    ax.set_yticks(y, p["topic"])
    for i, (lo_v, hi_v, lift) in enumerate(zip(p["prev_low"], p["prev_high"],
                                               p["lift_low_vs_high"], strict=True)):
        ax.text(max(lo_v, hi_v) + 0.004, i, f"x{lift:.2f}", va="center",
                fontsize=8.8, color=viz.INK_2)
    ax.set_xlim(0, float(p[["prev_low", "prev_high"]].to_numpy().max()) * 1.22)
    ax.grid(axis="y", visible=False)
    viz.pct(ax, "x")
    ax.set_xlabel("Share of review topic weight")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=viz.BLUE),
               plt.Rectangle((0, 0), 1, 1, color=viz.ORANGE)],
              [f"Upper band (rating >= {hi_cut:.0f})",
               f"Lower band (rating <= {lo_cut:.0f})"], loc="upper right")
    top_lift = p.iloc[-1]
    warmest = p.iloc[0]
    head = (f"'{warmest['topic']}' is "
            f"{1 / warmest['lift_low_vs_high']:.2f}x more of the text in the "
            "highest-rated listings")
    viz.title(ax, head,
              f"Pooled NMF, {C.N_TOPICS} topics, {len(docs):,} reviews. "
              "Multipliers are lower band / upper band.")
    # ax.get_title() reads the centre title, which is empty when viz.title
    # writes a left-aligned one, so the headline is passed through explicitly.
    ax.set_title(head, loc="left", pad=22)
    viz.source(fig, textwrap.fill(
        "Topic names are assigned by matching each fitted component against "
        "a fixed rule list; the terms behind every one are in "
        "outputs/tables/topics_pooled.csv.", 108))
    viz.save(fig, C.FIG / "09_topic_prevalence.png")

    # fig 10
    # z = log-odds(lower band) - log-odds(upper band), so the largest positive
    # values are the terms the lower band over-uses and the largest negative
    # ones are the terms the upper band over-uses.
    k = 14
    lower_terms = dt.tail(k).iloc[::-1]
    upper_terms = dt.head(k)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))

    ax = axes[0]
    ax.barh(range(k), lower_terms["z"].to_numpy()[::-1], color=viz.ORANGE, height=0.66)
    viz.bar_ends(ax, horizontal=True)
    ax.set_yticks(range(k), lower_terms["term"].to_numpy()[::-1])
    ax.set_xlabel("Weighted log-odds z (lower band over upper)")
    ax.grid(axis="y", visible=False)
    head_l = "Lower-rated reviews name objects and problems"
    viz.title(ax, head_l,
              f"Terms over-used in reviews of listings rated <= {lo_cut:.0f}")
    ax.set_title(head_l, loc="left", pad=20)

    ax = axes[1]
    ax.barh(range(k), -upper_terms["z"].to_numpy()[::-1], color=viz.BLUE, height=0.66)
    viz.bar_ends(ax, horizontal=True)
    ax.set_yticks(range(k), upper_terms["term"].to_numpy()[::-1])
    ax.set_xlabel("Weighted log-odds z (upper band over lower)")
    ax.grid(axis="y", visible=False)
    head_r = "Higher-rated reviews name feelings and touches"
    viz.title(ax, head_r,
              f"Terms over-used in reviews of listings rated >= {hi_cut:.0f}")
    ax.set_title(head_r, loc="left", pad=20)
    viz.source(fig, textwrap.fill(
        "Monroe, Colaresi and Quinn (2008) weighted log-odds with an "
        "informative Dirichlet prior, so words common in both corpora do "
        f"not dominate. {len(vocab_c):,} terms with {C.TOPIC_MIN_DF}+ "
        "occurrences; personal names removed. Note the confound: "
        "'apartment', 'building' and 'cottage' are property types, and "
        "property type is correlated with rating.", 108))
    fig.subplots_adjust(wspace=0.34)
    viz.save(fig, C.FIG / "10_complaint_language.png")

    # metrics
    metrics = {
        "n_topics": C.N_TOPICS,
        "low_band_cut": lo_cut,
        "high_band_cut": hi_cut,
        "low_band_listings": len(low_ids),
        "high_band_listings": len(high_ids),
        "low_band_reviews": int(len(low_docs)),
        "high_band_reviews": int(len(high_docs)),
        "vocabulary": int(len(vocab)),
        "names_stoplisted": int(len(stop_names)),
        "topics": [
            {"topic": r["topic"], "terms": r["terms"],
             "prevalence_lower_band": float(r["prev_low"]),
             "prevalence_upper_band": float(r["prev_high"]),
             "lift_lower_vs_upper": float(r["lift_low_vs_high"])}
            for _, r in prev.sort_values("lift_low_vs_high",
                                         ascending=False).iterrows()],
        "topics_low_band_only": [
            {"topic": t, "terms": ", ".join(tt), "prevalence": float(pv)}
            for t, tt, pv in zip(labels_lo, terms_lo, share_lo.mean(axis=0),
                                 strict=True)],
        "most_over_indexed_topic_lower_band": str(top_lift["topic"]),
        "most_over_indexed_lift_lower_band": float(top_lift["lift_low_vs_high"]),
        "most_over_indexed_topic_upper_band": str(warmest["topic"]),
        "most_over_indexed_lift_upper_band": float(1 / warmest["lift_low_vs_high"]),
        "terms_over_used_lower_band": dt.tail(20)["term"].tolist()[::-1],
        "terms_over_used_upper_band": dt.head(20)["term"].tolist(),
        "max_z_lower_band": float(dt["z"].max()),
        "max_z_upper_band": float(-dt["z"].min()),
    }
    (C.OUT / "topic_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k_: v for k_, v in metrics.items()
                      if k_ not in ("topics", "topics_low_band_only")}, indent=2))


if __name__ == "__main__":
    main()
