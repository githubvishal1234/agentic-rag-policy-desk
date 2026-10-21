from app.services.bm25 import BM25, tokenize

TEXTS = [
    "Employees must use the approved VPN for remote access.",
    "Employees must use strong passwords.",
    "Employees get 20 days of annual leave.",
]


def test_tokenize():
    assert tokenize("Use the VPN, at 9:00 AM!") == [
        "use", "vpn", "9", "00", "am",
    ]


def test_exact_keyword_ranks_first():
    scores = BM25(TEXTS).scores("vpn")

    assert scores[0] > 0
    assert scores[1] == scores[2] == 0


def test_rare_terms_outweigh_common_terms():
    bm25 = BM25(TEXTS)

    # "employees" is in every text, "passwords" in one.
    assert bm25.idf["passwords"] > bm25.idf["employees"] > 0


def test_no_match_and_empty_corpus():
    assert BM25(TEXTS).scores("holiday bonus") == [0, 0, 0]
    assert BM25([]).scores("anything") == []
