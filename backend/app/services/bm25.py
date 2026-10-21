import math
import re
from collections import Counter


# Function words carry no topic. Without this list, questions like "Can I
# ... on a trip?" keyword-match random chunks on "can", "a" and "on", and
# rank fusion counts those matches as much as a real keyword hit.
STOPWORDS = frozenset(
    "a about above after again all an and any are as at be because been "
    "before being below between both but by can could did do does doing "
    "down during each few for from further had has have having he her here "
    "hers him his how i if in into is it its just me more most my no nor "
    "not of off on once only or other our ours out over own same she should "
    "so some such than that the their them then there these they this those "
    "through to too under until up very was we were what when where which "
    "while who whom why will with would you your yours".split()
)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"\w+", text.lower())
        if token not in STOPWORDS
    ]


class BM25:
    """Okapi BM25 keyword scoring over a fixed list of texts."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / len(texts) if texts else 0

        document_frequency = Counter(
            term for counts in self.term_counts for term in counts
        )
        # Lucene-style IDF: always positive, even for very common terms.
        self.idf = {
            term: math.log(1 + (len(texts) - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        """Return one BM25 score per text; 0 means no query term matched."""

        # ponytail: scans every text per query, fine up to ~100k chunks;
        # switch to an inverted index (or a search engine) beyond that.
        terms = tokenize(query)
        results = []

        for counts, length in zip(self.term_counts, self.lengths):
            score = 0.0
            normalizer = self.k1 * (
                1 - self.b + self.b * length / self.average_length
            )

            for term in terms:
                frequency = counts.get(term, 0)

                if frequency:
                    score += (
                        self.idf[term]
                        * frequency
                        * (self.k1 + 1)
                        / (frequency + normalizer)
                    )

            results.append(score)

        return results
