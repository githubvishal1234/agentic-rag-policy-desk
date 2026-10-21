Hybrid search with an English stopword list in the BM25 tokenizer, so function words (can, a, on...) no longer create keyword matches that rank fusion over-rewards.

Diagnosed from v1 transcripts paraphrase-06 and paraphrase-07: the correct chunk was vector rank 1 but was pushed out of the top 4 by chunks that matched only on stopwords.

Caveat: the diagnosis used these same 26 cases (no held-out set), so this score is optimistic; the change itself is generic, not case-specific.

Command: python evaluation/run_retrieval_eval.py --variant v2 --method hybrid
