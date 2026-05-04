"""
Main Entry Point

Usage:
  python main.py                  # AutoGen example mode
  python main.py --mode cli       # Interactive CLI
  python main.py --mode web       # Streamlit web UI
  python main.py --mode evaluate  # Batch evaluation with LLM-as-a-Judge
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


def _load_config(path: str = "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def _setup_logging(config: dict):
    log_cfg = config.get("logging", {})
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_cfg.get("file", "logs/system.log")),
        ],
    )


def run_cli():
    """Launch the interactive CLI."""
    from src.ui.cli import main as cli_main
    cli_main()


def run_web():
    """Launch the Streamlit web UI."""
    import subprocess
    print("Starting Streamlit web UI at http://localhost:8501 ...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"], check=False)


async def run_evaluation(config_path: str = "config.yaml"):
    """
    Run batch evaluation with LLM-as-a-Judge on all example queries.
    Saves detailed results and a text summary to outputs/.
    """
    config = _load_config(config_path)
    _setup_logging(config)
    logger = logging.getLogger("main.evaluate")

    from src.autogen_orchestrator import AutoGenOrchestrator
    from src.evaluation.evaluator import SystemEvaluator

    print("\n" + "=" * 70)
    print("MULTI-AGENT RESEARCH SYSTEM - BATCH EVALUATION")
    print("=" * 70)

    print("\nInitializing orchestrator...")
    try:
        orchestrator = AutoGenOrchestrator(config)
    except Exception as exc:
        logger.error("Failed to initialize orchestrator: %s", exc)
        print(f"ERROR: {exc}")
        return

    evaluator = SystemEvaluator(config, orchestrator=orchestrator)

    query_file = "data/example_queries.json"
    print(f"Running evaluation on: {query_file}\n")

    report = await evaluator.evaluate_system(query_file)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    s = report.get("summary", {})
    sc = report.get("scores", {})
    print(f"\nTotal queries   : {s.get('total_queries', 0)}")
    print(f"Successful      : {s.get('successful', 0)}")
    print(f"Failed          : {s.get('failed', 0)}")
    print(f"Overall score   : {sc.get('overall_average', 0.0):.4f} (0-1 scale)")

    print("\nScores by criterion:")
    for crit, score in sc.get("by_criterion", {}).items():
        bar = "#" * int(score * 20)
        print(f"  {crit:<25} {score:.4f}  [{bar:<20}]")

    if report.get("best_result"):
        print(f"\nBest  query ({report['best_result']['score']:.4f}): {report['best_result']['query'][:70]}")
    if report.get("worst_result"):
        print(f"Worst query ({report['worst_result']['score']:.4f}): {report['worst_result']['query'][:70]}")

    print("\nDetailed results saved to outputs/")
    print("=" * 70 + "\n")


def run_autogen_demo(config_path: str = "config.yaml"):
    """Run a single demo query end-to-end (default mode)."""
    config = _load_config(config_path)
    _setup_logging(config)

    from src.autogen_orchestrator import AutoGenOrchestrator

    orchestrator = AutoGenOrchestrator(config)

    print(orchestrator.visualize_workflow())

    query = "What are the key principles of explainable AI for novice users in HCI?"
    print(f"\nDemo query: {query}\n")
    print("=" * 70)

    result = orchestrator.process_query(query)

    print("\n" + "=" * 70)
    print("RESPONSE")
    print("=" * 70)
    print(result.get("response", "No response generated."))

    meta = result.get("metadata", {})
    print(f"\nMessages exchanged : {meta.get('num_messages', 0)}")
    print(f"Sources gathered   : {meta.get('num_sources', 0)}")
    print(f"Agents involved    : {', '.join(meta.get('agents_involved', []))}")

    safety_events = result.get("safety_events", [])
    if safety_events:
        print(f"\nSafety events      : {len(safety_events)}")

    # Save the demo session
    import json
    from datetime import datetime
    Path("outputs").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = f"outputs/demo_session_{ts}.json"
    with open(session_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSession saved to {session_file}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent HCI Research Assistant")
    parser.add_argument(
        "--mode",
        choices=["cli", "web", "evaluate", "autogen"],
        default="autogen",
        help="Mode: cli | web | evaluate | autogen (default)",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    elif args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        asyncio.run(run_evaluation(args.config))
    else:
        run_autogen_demo(args.config)


if __name__ == "__main__":
    main()
