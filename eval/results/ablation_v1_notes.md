# ablation v1 notes

Ran 20 questions x 4 configs. Used local embeddings (all-MiniLM-L6-v2) since I didn't have an OpenAI key, and Claude Sonnet for generation + judging. No Cohere key so reranker used the local cross-encoder (bge-reranker-base).

Hybrid won on retrieval recall (0.775) — 20 points over dense-only (0.575). Sparse was close at 0.750, which makes sense because BM25 is good at exact financial terms like "non-accrual" and "leverage ratio." Dense by itself missed a lot of section types.

Hybrid also won on overall average (0.567 vs 0.481 for dense_only). The combination of dense + sparse via RRF clearly helps for these kinds of questions.

The big surprise: hybrid_reranked (0.480 avg) did WORSE than plain hybrid (0.567). Reranking improved citation precision a bit (0.464 vs 0.425) but tanked retrieval recall (0.625 vs 0.775) and fact recall (0.350 vs 0.500). I think the problem is that the reranker trims from 20 to 10 chunks, and the local cross-encoder doesn't know enough about financial text to keep the right ones. It's probably dropping relevant chunks that had exact BM25 matches in favor of chunks that are semantically similar but don't contain the actual numbers.

8 out of 20 questions got 0 fact recall across ALL configs. Most of these are medium/hard questions asking for cross-period trends or cross-issuer comparisons — things like "rank all 8 BDCs by NII yield" or "how has leverage trended over 4 quarters." The system retrieves some relevant chunks but the LLM can't synthesize a complete answer from partial data, or the judge is strict about matching expected facts.

Dense beat hybrid on 2 questions (both cross-period medium difficulty). Probably the BM25 component brought in noisy boilerplate chunks that diluted the context.

Next steps: try Cohere reranker instead of local, bump rerank_top_k to 15, and hand-correct the eval expected_facts since some of them are probably wrong.
