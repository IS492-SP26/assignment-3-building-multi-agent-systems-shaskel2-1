"""
LLM-as-a-Judge

Two independent judging prompts (Rubric + Holistic) using the vLLM endpoint.
Includes retry logic for transient 502/503 errors.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


class LLMJudge:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")
        self.model_cfg = config.get("models", {}).get("judge", {})
        self.criteria: List[Dict[str, Any]] = config.get("evaluation", {}).get("criteria", [])

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        self.model_name = os.getenv("OPENAI_MODEL") or self.model_cfg.get("name", "Qwen/Qwen3-8B")

        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.logger.info("LLMJudge initialized with %d criteria", len(self.criteria))

    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        rubric_scores = await self._run_rubric_judge(query, response, sources, ground_truth)
        holistic = await self._run_holistic_judge(query, response, ground_truth)

        total_weight = sum(c.get("weight", 1.0) for c in self.criteria)
        weighted = sum(
            rubric_scores.get(c["name"], {}).get("score", 0.0) * c.get("weight", 1.0)
            for c in self.criteria
        )
        rubric_overall = weighted / total_weight if total_weight > 0 else 0.0
        holistic_score = holistic.get("score", 0.0)
        combined = round(0.6 * rubric_overall + 0.4 * holistic_score, 4)

        return {
            "query": query,
            "overall_score": combined,
            "rubric_score": round(rubric_overall, 4),
            "holistic_score": round(holistic_score, 4),
            "criterion_scores": rubric_scores,
            "holistic_feedback": holistic.get("feedback", ""),
        }

    async def _run_rubric_judge(self, query, response, sources, ground_truth):
        criteria_text = "\n".join(
            f"- {c['name']} (weight {c['weight']}): {c['description']}"
            for c in self.criteria
        )
        prompt = f"""You are an expert evaluator assessing a research assistant's response.

ORIGINAL QUERY:
{query}

SYSTEM RESPONSE:
{response[:2000]}

{f"EXPECTED ANSWER: {ground_truth}" if ground_truth else ""}

EVALUATION CRITERIA:
{criteria_text}

Score each criterion on a 0-10 integer scale (0=completely fails, 10=excellent).
Return ONLY valid JSON:
{{
  "criterion_scores": {{
    "<criterion_name>": {{"score": <0-10>, "reasoning": "<1-2 sentences>"}},
    ...
  }}
}}"""

        raw = self._call_llm_with_retry(prompt)
        try:
            data = self._parse_json(raw)
            raw_scores = data.get("criterion_scores", {})
            result = {}
            for c in self.criteria:
                name = c["name"]
                entry = raw_scores.get(name, {})
                raw_score = float(entry.get("score", 5))
                result[name] = {
                    "score": round(max(0.0, min(raw_score / 10.0, 1.0)), 4),
                    "reasoning": entry.get("reasoning", ""),
                    "criterion": name,
                }
            return result
        except Exception as exc:
            self.logger.error("Rubric judge parse error: %s", exc)
            return {c["name"]: {"score": 0.5, "reasoning": "Parse error", "criterion": c["name"]} for c in self.criteria}

    async def _run_holistic_judge(self, query, response, ground_truth):
        prompt = f"""You are an HCI domain expert reviewing an AI research assistant's response.

QUERY:
{query}

RESPONSE:
{response[:2000]}

{f"REFERENCE ANSWER: {ground_truth}" if ground_truth else ""}

Evaluate holistically: accuracy, coverage, depth, and absence of hallucinations.
Score 0-10 and provide 2-3 sentence feedback.
Return ONLY valid JSON:
{{
  "score": <0-10>,
  "feedback": "<narrative critique>"
}}"""

        raw = self._call_llm_with_retry(prompt)
        try:
            data = self._parse_json(raw)
            raw_score = float(data.get("score", 5))
            return {
                "score": round(max(0.0, min(raw_score / 10.0, 1.0)), 4),
                "feedback": data.get("feedback", ""),
            }
        except Exception as exc:
            self.logger.error("Holistic judge parse error: %s", exc)
            return {"score": 0.5, "feedback": "Unable to parse holistic evaluation."}

    def _call_llm_with_retry(self, prompt: str, retries: int = 3, delay: float = 4.0) -> str:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Check OPENAI_API_KEY.")
        temperature = self.model_cfg.get("temperature", 0.2)
        max_tokens = self.model_cfg.get("max_tokens", 1024)
        last_exc = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Always respond with valid JSON only. Do not include <think> tags."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                self.logger.warning("Judge LLM attempt %d failed: %s", attempt + 1, exc)
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
        raise RuntimeError(f"Judge LLM failed after {retries} attempts: {last_exc}")

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        import re
        # Strip Qwen3 thinking tags
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # Strip markdown code fences
        if clean.startswith("```"):
            clean = re.sub(r'^```[a-z]*\n?', '', clean)
            clean = re.sub(r'\n?```$', '', clean)
            clean = clean.strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            clean = clean[start:end]
        return json.loads(clean)
