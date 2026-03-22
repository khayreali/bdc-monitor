import logging
import re

import yaml

from bdc_monitor.domain import Answer, EvalQuestion, EvalResult
from bdc_monitor.generation.llm_client import LLMClient
from bdc_monitor.generation.rag_pipeline import RAGPipeline

log = logging.getLogger(__name__)


def load_eval_questions(path: str) -> list[EvalQuestion]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [EvalQuestion(**q) for q in data["questions"]]


class Evaluator:

    def __init__(self, pipeline: RAGPipeline, judge_llm: LLMClient):
        self.pipeline = pipeline
        self.judge = judge_llm

    def run(self, questions: list[EvalQuestion], config_name: str = "") -> list[EvalResult]:
        results = []
        for i, eq in enumerate(questions):
            log.info(f"eval [{i+1}/{len(questions)}] {eq.question[:60]}...")
            answer = self.pipeline.ask(eq.question)
            result = self._score(eq, answer, config_name)
            results.append(result)
            log.info(
                f"  retrieval_recall={result.retrieval_recall:.2f} "
                f"citation_precision={result.citation_precision:.2f} "
                f"fact_recall={result.fact_recall:.2f}"
            )
        return results

    def _score(self, eq: EvalQuestion, answer: Answer, config_name: str) -> EvalResult:
        return EvalResult(
            question=eq,
            answer=answer,
            retrieval_recall=self._retrieval_recall(eq, answer),
            citation_precision=self._citation_precision(eq, answer),
            fact_recall=self._fact_recall(eq, answer),
            config_name=config_name,
        )

    def _retrieval_recall(self, eq: EvalQuestion, answer: Answer) -> float:
        if eq.relevant_filings:
            found = {rc.chunk.filing_accession for rc in answer.chunks_used}
            hits = sum(1 for f in eq.relevant_filings if f in found)
            return hits / len(eq.relevant_filings)

        if eq.relevant_section_types:
            found = {rc.chunk.section_type for rc in answer.chunks_used}
            hits = sum(1 for st in eq.relevant_section_types if st in found)
            return hits / len(eq.relevant_section_types)

        return 1.0

    def _citation_precision(self, eq: EvalQuestion, answer: Answer) -> float:
        if not answer.citations:
            return 0.0

        # build a map from chunk_id to chunk text
        chunk_texts = {rc.chunk.chunk_id: rc.chunk.text for rc in answer.chunks_used}

        supported = 0
        for cit in answer.citations:
            chunk_text = chunk_texts.get(cit.chunk_id, "")
            if not chunk_text:
                continue

            # extract the claim context around the citation
            claim = self._extract_claim(answer.answer_text, cit.chunk_id)

            prompt = (
                f"Does this filing excerpt support the claim made about it?\n\n"
                f"Claim from answer: {claim}\n\n"
                f"Filing excerpt: {chunk_text[:1500]}\n\n"
                f"Reply with only YES or NO."
            )
            try:
                resp = self.judge.generate(
                    "You are evaluating citation accuracy. Be strict.", prompt, max_tokens=10
                )
                if "YES" in resp.upper():
                    supported += 1
            except Exception:
                log.exception("citation judge call failed")

        return supported / len(answer.citations)

    def _fact_recall(self, eq: EvalQuestion, answer: Answer) -> float:
        if not eq.expected_facts:
            return 1.0

        facts_list = "\n".join(f"{i+1}. {f}" for i, f in enumerate(eq.expected_facts))
        prompt = (
            f"Given this answer, determine which expected facts are present.\n\n"
            f"Answer:\n{answer.answer_text[:3000]}\n\n"
            f"Expected facts:\n{facts_list}\n\n"
            f"For each numbered fact, reply with the number and YES or NO. "
            f"Example: 1. YES\\n2. NO"
        )
        try:
            resp = self.judge.generate(
                "You are evaluating answer completeness. Be strict.", prompt, max_tokens=200
            )
            # parse "1. YES\n2. NO" format
            found = 0
            for line in resp.strip().split("\n"):
                if "YES" in line.upper():
                    found += 1
            return found / len(eq.expected_facts)
        except Exception:
            log.exception("fact recall judge call failed")
            return 0.0

    def _extract_claim(self, answer_text: str, chunk_id: str) -> str:
        marker = f"[CITE: {chunk_id}]"
        idx = answer_text.find(marker)
        if idx == -1:
            # try without exact whitespace
            pattern = re.escape(chunk_id)
            m = re.search(rf"\[CITE:\s*{pattern}\s*\]", answer_text)
            idx = m.start() if m else -1

        if idx == -1:
            return answer_text[:300]

        start = max(0, idx - 150)
        end = min(len(answer_text), idx + len(marker) + 150)
        return answer_text[start:end]
