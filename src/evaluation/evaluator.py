"""
System Evaluator
Runs batch evaluation across test queries and generates a full report.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .judge import LLMJudge


class SystemEvaluator:
    """
    Evaluates the multi-agent system using test queries and LLM-as-a-Judge.

    Usage:
        evaluator = SystemEvaluator(config, orchestrator=orch)
        report = asyncio.run(evaluator.evaluate_system("data/example_queries.json"))
    """

    def __init__(self, config: Dict[str, Any], orchestrator=None):
        self.config = config
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("evaluation.evaluator")

        eval_cfg = config.get("evaluation", {})
        self.enabled: bool = eval_cfg.get("enabled", True)
        self.max_queries: Optional[int] = eval_cfg.get("num_test_queries", None)

        self.judge = LLMJudge(config)
        self.results: List[Dict[str, Any]] = []

    async def evaluate_system(
        self,
        test_queries_path: str = "data/example_queries.json",
    ) -> Dict[str, Any]:
        """Run full batch evaluation and return a consolidated report."""
        if not self.enabled:
            return {"error": "Evaluation disabled in config."}

        test_cases = self._load_queries(test_queries_path)
        if self.max_queries:
            test_cases = test_cases[: self.max_queries]

        self.logger.info("Evaluating %d queries...", len(test_cases))

        for i, case in enumerate(test_cases, 1):
            self.logger.info("Query %d/%d: %s", i, len(test_cases), case.get("query", "")[:60])
            try:
                result = await self._evaluate_query(case)
            except Exception as exc:
                self.logger.error("Error on query %d: %s", i, exc)
                result = {"query": case.get("query", ""), "error": str(exc)}
            self.results.append(result)

        report = self._generate_report()
        self._save_results(report)
        return report

    async def _evaluate_query(self, case: Dict[str, Any]) -> Dict[str, Any]:
        query = case.get("query", "")
        ground_truth = case.get("ground_truth")

        if self.orchestrator:
            try:
                response_data = self.orchestrator.process_query(query)
            except Exception as exc:
                response_data = {"query": query, "response": f"Error: {exc}", "metadata": {}}
        else:
            response_data = {
                "query": query,
                "response": "No orchestrator connected for this evaluation run.",
                "metadata": {},
            }

        evaluation = await self.judge.evaluate(
            query=query,
            response=response_data.get("response", ""),
            sources=response_data.get("metadata", {}).get("citations", []),
            ground_truth=ground_truth,
        )

        return {
            "query": query,
            "category": case.get("category", "general"),
            "response": response_data.get("response", ""),
            "evaluation": evaluation,
            "metadata": response_data.get("metadata", {}),
            "ground_truth": ground_truth,
        }

    def _load_queries(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            self.logger.warning("Test queries file not found: %s", path)
            return []
        with open(p) as f:
            return json.load(f)

    def _generate_report(self) -> Dict[str, Any]:
        if not self.results:
            return {"error": "No results."}

        successful = [r for r in self.results if "error" not in r]
        failed = [r for r in self.results if "error" in r]

        overall_scores = [r["evaluation"].get("overall_score", 0.0) for r in successful]
        rubric_scores_map: Dict[str, List[float]] = {}

        for r in successful:
            for crit, data in r["evaluation"].get("criterion_scores", {}).items():
                rubric_scores_map.setdefault(crit, []).append(data.get("score", 0.0))

        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        avg_criteria = {
            crit: round(sum(vals) / len(vals), 4)
            for crit, vals in rubric_scores_map.items()
        }

        best = max(successful, key=lambda r: r["evaluation"].get("overall_score", 0.0), default=None)
        worst = min(successful, key=lambda r: r["evaluation"].get("overall_score", 0.0), default=None)

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_queries": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(self.results) if self.results else 0.0,
            },
            "scores": {
                "overall_average": round(avg_overall, 4),
                "by_criterion": avg_criteria,
            },
            "best_result": {
                "query": best["query"],
                "score": best["evaluation"].get("overall_score", 0.0),
            } if best else None,
            "worst_result": {
                "query": worst["query"],
                "score": worst["evaluation"].get("overall_score", 0.0),
            } if worst else None,
            "detailed_results": self.results,
        }

    def _save_results(self, report: Dict[str, Any]):
        out = Path("outputs")
        out.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_file = out / f"evaluation_{ts}.json"
        with open(results_file, "w") as f:
            json.dump(report, f, indent=2)
        self.logger.info("Saved evaluation report to %s", results_file)

        summary_file = out / f"evaluation_summary_{ts}.txt"
        with open(summary_file, "w") as f:
            s = report.get("summary", {})
            sc = report.get("scores", {})
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp       : {report.get('timestamp', '')}\n")
            f.write(f"Total queries   : {s.get('total_queries', 0)}\n")
            f.write(f"Successful      : {s.get('successful', 0)}\n")
            f.write(f"Failed          : {s.get('failed', 0)}\n")
            f.write(f"Success rate    : {s.get('success_rate', 0.0):.1%}\n\n")
            f.write(f"Overall average : {sc.get('overall_average', 0.0):.4f}\n\n")
            f.write("By criterion:\n")
            for crit, score in sc.get("by_criterion", {}).items():
                f.write(f"  {crit:<25} {score:.4f}\n")
            if report.get("best_result"):
                f.write(f"\nBest query  ({report['best_result']['score']:.4f}): {report['best_result']['query'][:80]}\n")
            if report.get("worst_result"):
                f.write(f"Worst query ({report['worst_result']['score']:.4f}): {report['worst_result']['query'][:80]}\n")
        self.logger.info("Saved summary to %s", summary_file)
