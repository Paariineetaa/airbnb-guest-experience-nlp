"""
The aspect lexicon: eight guest-experience attributes and the surface forms
that signal them.

Where it comes from
-------------------
Six of the eight aspects are the subscores Airbnb itself collects
(cleanliness, location, value, communication, check-in, accuracy), so the
extraction can be validated against an independent measurement of the same
construct. Noise/disturbance and space/comfort were added because they
dominate the mined vocabulary and have no subscore of their own.

The surface forms were seeded by dependency parsing a 6,000-review sample with
spaCy `en_core_web_sm`, taking noun-chunk heads and the adjectives attached to
them, and keeping heads that appeared 40+ times. `aspects.py` re-runs that
mining pass into `outputs/tables/lexicon_discovery.csv`, so coverage against
the corpus vocabulary is auditable. The list is committed and frozen: the
analysis has to be deterministic, and a lexicon that re-derives itself on
every run is not.

Two choices
-----------
- **Terms, not sentiment.** Nothing here carries polarity. "dirty" and
  "spotless" are both cleanliness cues. Whether the sentence is a complaint is
  VADER's job. Polarity in the lexicon would hard-code the answer.
- **Overlap is allowed.** "quiet neighborhood" is a noise cue and a location
  cue, and counts for both. Aspects are not a partition of the text.
"""
from __future__ import annotations

import re

# Ordered. Everything downstream - chart series, coefficient tables, the
# priority matrix - reads this order, so colour assignment stays stable.
ASPECTS: tuple[str, ...] = (
    "cleanliness",
    "location",
    "host_communication",
    "checkin",
    "value",
    "accuracy",
    "noise",
    "space_comfort",
)

LABELS: dict[str, str] = {
    "cleanliness": "Cleanliness",
    "location": "Location",
    "host_communication": "Host communication",
    "checkin": "Check-in / arrival",
    "value": "Value for money",
    "accuracy": "Accuracy vs listing",
    "noise": "Noise / disturbance",
    "space_comfort": "Space / comfort",
}

# Airbnb's own subscore for the same construct, where one exists. These pairs
# are the convergent-validity check in aspects.py. None means the dataset has
# no independent measurement.
SUBSCORE: dict[str, str | None] = {
    "cleanliness": "review_scores_cleanliness",
    "location": "review_scores_location",
    "host_communication": "review_scores_communication",
    "checkin": "review_scores_checkin",
    "value": "review_scores_value",
    "accuracy": "review_scores_accuracy",
    "noise": None,
    "space_comfort": None,
}

# Single tokens or multi-word phrases. Matching is case-insensitive on word
# boundaries. A trailing "*" means "this stem plus any suffix", which collapses
# clean/cleaner/cleanest/cleaning/cleanliness to one entry without a stemmer
# that would also collapse "cleaner" the noun.
TERMS: dict[str, tuple[str, ...]] = {
    "cleanliness": (
        "clean*", "spotless", "immaculate", "tidy", "untidy", "pristine",
        "dirty", "filthy", "grubby", "grime", "grimy", "dust", "dusty",
        "stain*", "smelly", "smell", "smelled", "odor", "odour", "musty",
        "mold", "mould", "mildew", "hygien*", "sanitary", "unsanitary",
        "housekeep*", "messy", "mess", "cobweb*", "cockroach*", "roach*",
        "bed bug*", "bedbug*", "crumbs", "sticky", "garbage", "trash",
        "well kept", "well-kept", "spotlessly",
    ),
    "location": (
        "location", "located", "neighborhood", "neighbourhood", "neighborhoods",
        "neighbourhoods", "downtown", "walking distance", "walkable",
        "close to", "near by", "nearby", "minutes from", "minutes away",
        "blocks from", "block from", "steps from", "convenient*",
        "central*", "restaurant*", "bar", "bars", "cafe*", "coffee shop*",
        "shops", "shopping", "grocery", "market", "pike place",
        "bus", "buses", "light rail", "transit", "transportation",
        "public transport*", "airport", "parking", "park nearby",
        "safe area", "sketchy", "unsafe", "commute", "uber", "taxi",
        "space needle", "waterfront", "easy access", "getting around",
        # Added after the first recall audit (see aspects.audit_lexicon).
        # The highest-frequency non-generic head nouns the first draft
        # missed, each a location cue.
        "view", "views", "distance", "walk", "walked", "walking",
        "shop", "street", "streets", "access", "block", "blocks",
        "minute", "minutes", "town", "city", "stroll",
    ),
    "host_communication": (
        "host", "hosts", "hostess", "communicat*", "responsive",
        "respond*", "replied", "reply", "answered", "answering",
        "in touch", "get in touch", "got back to", "text", "texted",
        "email*", "message*", "phone", "prompt*", "accommodating",
        "helpful", "attentive", "gracious", "hospitalit*", "welcoming",
        "recommendation*", "suggestion*", "tips", "flexible", "unresponsive",
        "never responded", "slow to respond", "checked in on us",
    ),
    "checkin": (
        "check in", "check-in", "checkin", "checking in", "checked in",
        "check out", "check-out", "checkout", "checked out",
        "arrival", "arrived", "arriving", "self check*", "lock box",
        "lockbox", "key", "keys", "keypad", "door code", "entry code",
        "access code", "combination", "instruction*", "direction*",
        "greeted", "greeting", "met us", "met me", "showed us around",
        "early check", "late check", "drop our bags", "drop off our bags",
        "luggage", "smooth process", "seamless",
    ),
    "value": (
        "value", "worth", "worth it", "worth every", "price", "priced",
        "pricing", "cost", "costs", "affordable", "cheap", "cheaper",
        "inexpensive", "expensive", "overpriced", "pricey", "bargain",
        "deal", "great deal", "for the money", "money", "rate", "rates",
        "fee", "fees", "cleaning fee", "charge", "charged", "budget",
        "reasonable", "unreasonable",
    ),
    "accuracy": (
        "as described", "as advertised", "description", "described",
        "listing", "advertis*", "accurate*", "inaccurate", "exactly as",
        "just as", "just like the", "photo*", "picture*", "pictured",
        "misleading", "as expected", "what we expected", "what i expected",
        "not what we", "not what i", "matched", "matches the", "true to",
        "as promised", "as shown",
    ),
    "noise": (
        "noise", "noisy", "noises", "loud", "loudly", "quiet", "quieter",
        "quietly", "silent", "peaceful", "sleep", "slept", "sleeping",
        "wake", "woke", "awoke", "traffic", "street noise", "thin wall*",
        "walls are thin", "upstairs", "footsteps", "barking", "dog barked",
        "music", "party", "partying", "construction", "siren*", "train",
        "airplane*", "snoring", "disturb*", "earplug*", "soundproof*",
        "white noise", "air conditioner noise",
    ),
    "space_comfort": (
        "space", "spacious", "roomy", "cozy", "cosy", "comfortable",
        "comfy", "comfort", "uncomfortable", "bed", "beds", "mattress",
        "pillow*", "linens", "sheets", "towel*", "bathroom", "shower",
        "kitchen", "fridge", "refrigerator", "living room", "bedroom*",
        "layout", "cramped", "tiny", "small for", "too small", "roomier",
        "plenty of room", "plenty of space", "amenit*", "wifi", "wi-fi",
        "internet", "heat", "heating", "heater", "air conditioning",
        "cold", "chilly", "warm", "stuffy", "hot water", "well appointed",
        "well-appointed", "furnish*", "decor*", "homey", "homely",
        # Added after the first recall audit, same rule as above.
        "privacy", "window", "windows", "door", "closet", "washer",
        "dryer", "porch", "deck", "patio", "yard", "garden", "breakfast",
    ),
}


# Head nouns that name the accommodation or the visit rather than an attribute
# of it. They dominate any frequency count of noun chunks ("place" is the
# commonest noun by a factor of two) and an aspect lexicon has nothing to do
# with them. The recall audit reports coverage with and without them.
GENERIC_HEADS = frozenset(
    """
place stay apartment room house home time area experience lot day night thing
trip week weekend way part person people family friend couple husband wife
guest host_name unit studio cottage condo suite property accommodation
loft flat visit need question thank thanks morning evening afternoon
""".split()  # noqa: SIM905 - a wrapped word block beats a long list literal
)


def _to_pattern(term: str) -> str:
    """One lexicon entry -> one regex alternative."""
    stem = term[:-1] if term.endswith("*") else term
    body = re.escape(stem).replace(r"\ ", r"\s+")
    # A "*" entry may run on into further word characters. A plain entry may
    # not, so "bar" never fires on "barbecue".
    return body + (r"\w*" if term.endswith("*") else "")


def compile_patterns() -> dict[str, re.Pattern[str]]:
    """One compiled alternation per aspect, longest-first so phrases win."""
    out: dict[str, re.Pattern[str]] = {}
    for aspect in ASPECTS:
        terms = sorted(TERMS[aspect], key=len, reverse=True)
        alts = "|".join(_to_pattern(t) for t in terms)
        out[aspect] = re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.IGNORECASE)
    return out


def n_terms() -> dict[str, int]:
    return {a: len(TERMS[a]) for a in ASPECTS}
