# ablation v2 notes

Re-ran with the tender_offer classifier fix (50 sections now classified as tender_offer vs zero in v1). Still using local embeddings — OpenAI key has insufficient quota so couldn't test cloud embeddings this round.

Config ranking stayed the same: hybrid (0.582) > dense (0.498) > hybrid_reranked (0.495) > sparse (0.455). Hybrid still wins by a wide margin on retrieval recall (0.775 vs 0.575 for dense). The RRF merge is clearly pulling its weight.

Fact recall improved across the board compared to v1 — dense went from 0.475 to 0.600, hybrid from 0.500 to 0.525, hybrid_reranked from 0.350 to 0.450. The improvements are probably from the classifier fix giving the retriever better-labeled chunks for sections that discuss repurchase programs. Or just LLM non-determinism honestly — hard to tell with only 20 questions.

The tender_offer classifier fix didn't visibly help the three repurchase-related eval questions (Q8, Q12, Q15). They were all 0.0 in v1 and stayed 0.0 or 0.5 in v2. The fix changes the section metadata but the actual chunk text is the same, so unless the query router explicitly filters for tender_offer sections, the retriever won't behave differently. The fix's real value is for users who want to filter by section type manually.

Hybrid_reranked did worse than v1 on retrieval recall (0.575 vs 0.625). Still worse than plain hybrid. I'm more and more convinced the local cross-encoder is the problem — it's not great on financial text and trimming to 10 chunks is too aggressive.

No question went from passing in v1 to failing in v2. A few easy questions (Q1, Q4) moved around but the overall pattern held — easy questions mostly work, medium/hard questions that need cross-period or cross-issuer data mostly don't.
