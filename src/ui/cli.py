"""
Command Line Interface
Interactive CLI for the multi-agent research system.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

from src.autogen_orchestrator import AutoGenOrchestrator


class CLI:
    """Interactive CLI for the HCI Multi-Agent Research Assistant."""

    DIVIDER = "=" * 70
    THIN = "-" * 70

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self._setup_logging()
        self.logger = logging.getLogger("cli")
        self.orchestrator = AutoGenOrchestrator(self.config)
        self.query_count = 0
        self.safety_event_count = 0

    def _setup_logging(self):
        log_cfg = self.config.get("logging", {})
        Path("logs").mkdir(exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, log_cfg.get("level", "INFO")),
            format=log_cfg.get("format", "%(asctime)s %(name)s %(levelname)s %(message)s"),
            handlers=[
                logging.FileHandler(log_cfg.get("file", "logs/system.log")),
            ],
        )

    async def run(self):
        self._print_welcome()
        while True:
            try:
                query = input("\nResearch query (or 'help'): ").strip()
            except (KeyboardInterrupt, EOFError):
                self._print_goodbye()
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                self._print_goodbye()
                break
            if query.lower() == "help":
                self._print_help()
                continue
            if query.lower() == "clear":
                os.system("clear" if os.name == "posix" else "cls")
                continue
            if query.lower() == "stats":
                self._print_stats()
                continue

            print(f"\n{self.DIVIDER}")
            print("Processing your query through the agent pipeline...")
            print(f"{self.DIVIDER}\n")

            try:
                result = self.orchestrator.process_query(query)
                self.query_count += 1
                self._display_result(result)
            except Exception as exc:
                print(f"\nERROR: {exc}")
                self.logger.exception("Error processing query")

    def _print_welcome(self):
        print(f"\n{self.DIVIDER}")
        print(f"  {self.config['system']['name']}")
        print(f"  Topic: {self.config['system']['topic']}")
        print(f"{self.DIVIDER}")
        print("\nType 'help' for commands, or enter a research query.\n")

    def _print_help(self):
        print(f"\n{self.THIN}")
        print("  help    Show this message")
        print("  stats   Show session statistics")
        print("  clear   Clear the terminal")
        print("  quit    Exit")
        print(f"{self.THIN}")

    def _print_goodbye(self):
        print(f"\n{self.DIVIDER}")
        print("Thank you for using the Multi-Agent HCI Research Assistant. Goodbye!")
        print(f"{self.DIVIDER}\n")

    def _print_stats(self):
        print(f"\n{self.THIN}")
        print(f"  Queries processed : {self.query_count}")
        print(f"  Safety events     : {self.safety_event_count}")
        safety_stats = self.orchestrator.safety_manager.get_safety_stats()
        print(f"  Violation rate    : {safety_stats.get('violation_rate', 0.0):.1%}")
        print(f"{self.THIN}")

    def _display_result(self, result: Dict[str, Any]):
        # --- Safety events ---
        safety_events = result.get("safety_events", [])
        new_events = [e for e in safety_events if not e.get("shown")]
        if new_events:
            self.safety_event_count += len(new_events)
            print(f"{'!' * 70}")
            print(f"SAFETY EVENTS DETECTED ({len(new_events)})")
            print(f"{'!' * 70}")
            for event in new_events:
                print(f"  Type   : {event.get('type', '?').upper()}")
                print(f"  Action : {event.get('action', '?').upper()}")
                for v in event.get("violations", []):
                    print(f"  [{v.get('category','?')}] ({v.get('severity','?')}) {v.get('reason','')}")
            print()

        if result.get("metadata", {}).get("input_blocked"):
            print(f"INPUT BLOCKED: {result.get('response', '')}\n")
            return

        # --- Response ---
        print(f"\n{self.DIVIDER}")
        print("RESPONSE")
        print(f"{self.DIVIDER}\n")
        response = result.get("response", "No response generated.")
        print(response)

        # --- Citations ---
        citations = result.get("metadata", {}).get("citations", [])
        if citations:
            print(f"\n{self.THIN}")
            print("CITATIONS")
            print(f"{self.THIN}")
            for i, url in enumerate(citations, 1):
                print(f"  [{i}] {url}")

        # --- Agent traces (verbose) ---
        if self.config.get("ui", {}).get("verbose", False):
            history = result.get("conversation_history", [])
            if history:
                print(f"\n{self.THIN}")
                print("AGENT TRACES (summary)")
                print(f"{self.THIN}")
                for msg in history:
                    src = msg.get("source", "?")
                    snippet = msg.get("content", "")[:120].replace("\n", " ")
                    print(f"  [{src}] {snippet}...")

        # --- Metadata ---
        meta = result.get("metadata", {})
        print(f"\n{self.THIN}")
        print(
            f"  Messages: {meta.get('num_messages', 0)} | "
            f"Sources: {meta.get('num_sources', 0)} | "
            f"Agents: {', '.join(meta.get('agents_involved', []))}"
        )
        output_action = meta.get("output_action", "allow")
        if output_action != "allow":
            print(f"  Output safety action: {output_action.upper()}")
        print(f"{self.DIVIDER}\n")

        # --- Save session ---
        Path("outputs").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"outputs/cli_session_{ts}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent HCI Research CLI")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cli = CLI(config_path=args.config)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
