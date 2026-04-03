# ablation v3 notes

Switched from local embeddings (all-MiniLM-L6-v2, 384-dim) to OpenAI text-embedding-3-small (1536-dim). Everything else the same as v2 — fixed tender_offer classifier, Claude Sonnet for generation/judging, local cross-encoder reranker.

31,280 chunks indexed (about 1,500 lost to OpenAI rate limit failures during indexing — 95% survived).

Overall config ranking shifted. In v2 hybrid was clearly ahead (0.582 avg). In v3 the gap narrowed — hybrid still leads (0.504) but hybrid_reranked is right behind (0.496). The big change is that hybrid_reranked's citation precision jumped to 0.539, far ahead of any other config. So the reranker IS helping now — just on citation quality rather than retrieval recall.

Dense_only got a 5-point bump on retrieval recall (0.575→0.625), which makes sense since that's the component most affected by better embeddings. Sparse_only barely changed since BM25 doesn't use embeddings.

Investigated the reranker: bumping top_k from 20 to 40 and rerank_top_k from 10 to 20 improved retrieval recall (0.550→0.650) and fact recall (0.400→0.500) but tanked citation precision (0.539→0.304). It's a genuine precision-recall tradeoff. The tight 20/10 setting finds precise citations but drops relevant chunks. The loose 40/20 setting retrieves better but cites worse. Neither makes hybrid_reranked clearly beat hybrid — they trade off different metrics.

Surprise: retrieval recall for hybrid actually dropped v2→v3 (0.775→0.675). I think this is partly LLM non-determinism — sparse_only also dropped (0.750→0.700) even though BM25 doesn't touch embeddings. The noise floor on 20 questions is about ±5%.

v1→v2→v3 comparison for hybrid (best config):
- v1 (local embed, broken classifier): 0.567 avg
- v2 (local embed, fixed classifier): 0.582 avg
- v3 (OpenAI embed, fixed classifier): 0.504 avg

The v3 average is lower mostly because citation precision and fact recall fluctuated. Retrieval recall is roughly stable. Honestly, with 20 questions the numbers bounce around enough between runs that I can't claim the OpenAI embeddings are clearly better or worse. I'd need a larger eval set to see a real signal.
