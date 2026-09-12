"""
Text handling shared by the cleaning step and the aspect scorer.

Three judgement calls, made once here rather than five times inline: what
counts as an automated message, what counts as English, and where a sentence
ends.
"""
from __future__ import annotations

import re
import unicodedata

# automated platform messages
# Airbnb writes a review row itself when a booking is cancelled. The text is
# templated and normally ends "This is an automated posting." A platform
# event, not a guest opinion. Left in, it attaches "canceled" to the
# host-communication aspect of every listing that ever had a cancellation.
#
# The first two alternatives are anchored to the start of the review.
# Unanchored, they also delete 91 genuine reviews that mention a cancellation
# in passing - "our original bnb was cancelled at the last minute, Karina took
# us in". The third is unanchored: the automated sentence ends the message.
#
# The "my/our/the reservation" variants were added after a test caught two
# notices reading "The host canceled my reservation 4 days before arrival."
# The first draft matched only "this reservation".
CANCELLATION_RE = re.compile(
    r"(^the\s+host\s+cancell?ed\s+(?:this|my|our|the)\s+reservation"
    r"|^the\s+reservation\s+was\s+cancell?ed"
    r"|this\s+is\s+an\s+automated\s+posting)",
    re.IGNORECASE,
)


# The 40 commonest English function words. Closed-class: content words like
# "hotel" or "restaurant" appear in Spanish and German reviews too.
ENGLISH_STOPWORDS = frozenset(
    """
a an the and or but if then than that this these those of in on at to for
with from by is was were are be been we i you he she it they our my your
their us not no very had have has did do does
""".split()  # noqa: SIM905 - a wrapped word block beats a long list literal
)

# Function words from the languages that appear in this corpus - Spanish,
# French, German, Italian, Portuguese, Dutch. Anything that is also an English
# word ("a", "no", "in", "die", "so") is stripped out below, so a token counts
# for one side only.
FOREIGN_STOPWORDS = frozenset(
    """
el la los las un una unos unas y pero porque muy con para por como este esta
estos estas nos nuestro fue era estaba estuvo tiene tienen todo todos toda
le les des du une deux dans avec pour sur nous vous ils elles tres etait
etaient ete beaucoup bien aussi mais chez tout toute tous plus
der das den dem ein eine einen einem und oder aber weil sehr mit fuer auf
ist sind war waren wir ihr sie nicht auch sehr alles alle schoen gut
il lo gli una del della dei delle che non piu molto anche siamo erano
os as uma umas obrigado muito nao com para mas tambem
het een van zijn niet ook maar zeer heel goed
""".split()  # noqa: SIM905 - a wrapped word block beats a long list literal
) - ENGLISH_STOPWORDS

_WORD_RE = re.compile(r"[a-z']+")

# Sentence boundary: terminal punctuation, or a newline. Guests use newlines
# as punctuation more often than full stops.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_WS_RE = re.compile(r"[ \t\r\f\v]+")


def normalise(text: str) -> str:
    """NFKC-fold, collapse horizontal whitespace, keep newlines as breaks."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return _WS_RE.sub(" ", t).strip()


def is_cancellation(text: str) -> bool:
    return bool(CANCELLATION_RE.search((text or "").strip()))


def ascii_ratio(text: str) -> float:
    """Share of characters in the ASCII range. 1.0 for an empty string."""
    if not text:
        return 1.0
    return sum(1 for ch in text if ord(ch) < 128) / len(text)


def stopword_hits(text: str) -> tuple[int, int]:
    """(distinct English function words, distinct non-English function words).

    Distinct, not total. A review repeating "the" ten times does not qualify.
    """
    words = set(_WORD_RE.findall((text or "").lower()))
    return len(words & ENGLISH_STOPWORDS), len(words & FOREIGN_STOPWORDS)


def looks_english(text: str, min_ascii: float, min_foreign: int) -> bool:
    """
    A two-part heuristic, not a language classifier.

    1. **Script.** Below `min_ascii` ASCII characters the review is not in the
       Latin alphabet - Chinese, Korean, Japanese, Russian, Greek, Hebrew,
       Arabic.
    2. **Function words.** Latin-script languages pass the script test and are
       caught by counting function words on both sides. A review is rejected
       when it carries at least `min_foreign` distinct non-English function
       words *and* more of them than English ones.

    English function words are not required to be present. A quarter of this
    corpus is telegraphic - "Great location, comfy bed - easy check-in" - and
    requiring them threw away 1,720 such reviews, the ones with the highest
    density of aspect terms.

    Misses: a short Latin-script review with too few function words to trip
    the count ("Perfecto!", "Sehr gut"). `aspects.py` reports how many reviews
    match no aspect at all, which bounds the damage.
    """
    if ascii_ratio(text) < min_ascii:
        return False
    en, foreign = stopword_hits(text)
    return not (foreign >= min_foreign and foreign > en)


def sentences(text: str) -> list[str]:
    """Split into sentences. Cheap: this runs 85k times."""
    if not text:
        return []
    parts = (p.strip() for p in _SENT_SPLIT_RE.split(text))
    return [p for p in parts if p]
