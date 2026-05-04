"""
LLM-as-a-Judge

Uses the vLLM/OpenAI-compatible endpoint to score system responses on
multiple criteria using two independent judging prompts:

  Prompt A (Rubric Judge):  scores relevance, evidence quality, factual accuracy,
                             safety compliance, and clarity on a 0-10 scale.
  Prompt B (Holistic Judge): gives an overall quality score and written feedback
                              from the perspective of an HCI domain expert.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


class LLMJudge:
    """
    LLM-as-a-Judge for evaluating multi-agent research outputs.

    Uses two independent judging prompts for each response:
      - Prompt A: criterion-level rubric scoring (0-10 per criterion)
      - Prompt B: holistic domain-expert evaluation (0-10 overall + narrative)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")

        self.model_cfg = config.get("models", {}).get("judge", {})
        self.criteria: List[Dict[str, Any]] = config.get("evaluation", {}).get("criteria", [])

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        self.model_name = os.getenv("OPENAI_MODEL") or self.model_cfg.get("name", "Qwen/Qwen3-8B")

        if not api_key:
            self.logger.warning("OPENAI_API_KEY not found; judge will not function.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.logger.info("LLMJudge initialized with %d criteria", len(self.criteria))

    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run both judge prompts and return combined results.

        Returns:
          - overall_score (float, 0-1)
          - criterion_scores (dict of criterion -> {score, reasoning})
          - holistic_score (float, 0-1)
          - holistic_feedback (str)
          - combined_score (float): weighted average of rubric + holistic
        """
        # --- Prompt A: criterion-level rubric ---
        rubric_scores = await self._run_rubric_judge(query, response, sources, ground_truth)

        # --- Prompt B: holistic domain-expert ---
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

    async def _run_rubric_judge(
        self,
        query: str,
        response: str,
        sources: Optional[List],
        ground_truth: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Prompt A: score each criterion individually on a 0-10 scale."""
        criteria_text = "\n".join(
            f"- {c['name']} (weight {c['weight']}): {c['description']}"
            for c in self.criteria
        )

        prompt = f"""You are an expert evaluator assessing a research assistant's response.

ORIGINAL QUERY:
{query}

SYSTEM RESPONSE:
{response}

{f"EXPECTED ANSWER: {ground_truth}" if ground_truth else ""}
{f"NUMBER OF SOURCES CITED: {len(sources)}" if sources else ""}

EVALUATION CRITERIA:
{criteria_text}

Score each criterion on a 0-10 integer scale (0=completely fails, 10=excellent).
Return ONLY valid JSON in this exact format:
{{
  "criterion_scores": {{
    "<criterion_name>": {{"score": <0-10>, "reasoning": "<1-2 sentences>"}},
    ...
  }}
}}"""

        raw = self._call_llm(prompt)
        try:
            data = self._parse_json(raw)
            raw_scores = data.get("criterion_scores", {})
            normalized = {}
            for c in self.criteria:
                name = c["name"]
                entry = raw_scores.get(name, {})
                raw_score = float(entry.get("score", 5))
                normalized[name] = {
                    "score": round(max(0.0, min(raw_score / 10.0, 1.0)), 4),
                    "reasoning": entry.get("reasoning", ""),
                    "criterion": name,
                }
            return normalized
        except Exception as exc:
            self.logger.error("Rubric judge parse error: %s | raw: %s", exc, raw[:200])
            return {c["name"]: {"score": 0.5, "reasoning": "Parse error", "criterion": c["name"]} for c in self.criteria}

    async def _run_holistic_judge(
        self,
        query: str,
        response: str,
        ground_truth: Optional[str],
    ) -> Dict[str, Any]:
        """Prompt B: holistic expert evaluation returning an overall score and narrative."""
        prompt = f"""You are an HCI domain expert reviewing a research report generated by an AI assistant.

QUERY:
{query}

RESPONSE:
{response}

{f"REFERENCE ANSWER: {ground_truth}" if ground_truth else ""}

As an HCI domain expert, evaluate this response holistically. Consider:
  - Does it demonstrate accurate knowledge of the field?
  - Does it cover the most important aspects of the topic?
  - Is the depth appropriate for an HCI researcher or advanced student?
  - Does it avoid errors, hallucinations, or unsupported claims?

Provide an overall quality score (0-10) and a short narrative critique (2-4 sentences).
Return ONLY valid JSON:
{{
  "score": <0-10>,
  "feedback": "<narrative critique>"
}}"""

        raw = self._call_llm(prompt)
        try:
            data = self._parse_json(raw)
            raw_score = float(data.get("score", 5))
            return {
                "score": round(max(0.0, min(raw_score / 10.0, 1.0)), 4),
                "feedback": data.get("feedback", ""),
            }
        except Exception as exc:
            self.logger.error("Holistic judge parse error: %s | raw: %s", exc, raw[:200])
            return {"score": 0.5, "feedback": "Unable to parse holistic evaluation."}

    def _call_llm(self, prompt: str) -> str:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized.")
        temperature = self.model_cfg.get("temperature", 0.2)
        max_tokens = self.model_cfg.get("max_tokens", 1024)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("```", 2)[-1] if "```" in clean[3:] else clean[3:]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()
        # Find the first '{' and last '}'
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            clean = clean[start:end]
        return json.loads(clean)
