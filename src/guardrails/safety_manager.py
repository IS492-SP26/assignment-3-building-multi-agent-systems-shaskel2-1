"""
Safety Manager
Coordinates input and output guardrails and maintains a log of safety events.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .input_guardrail import InputGuardrail
from .output_guardrail import OutputGuardrail


class SafetyManager:
    """
    Central coordinator for all safety guardrails.

    On each query:
      1. check_input_safety()  is called before agents run.
      2. check_output_safety() is called on the final response.

    Policy categories and response strategies are read from config.yaml
    under the 'safety' key.

    Safety events are logged to logs/safety_events.log and kept
    in-memory for display in the UI.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        safety_cfg = config.get("safety", {})

        self.enabled: bool = safety_cfg.get("enabled", True)
        self.log_events: bool = safety_cfg.get("log_events", True)

        self.prohibited_categories: List[str] = safety_cfg.get(
            "prohibited_categories",
            ["harmful_content", "prompt_injection", "misinformation", "pii_exposure"],
        )
        self.on_violation: Dict[str, Any] = safety_cfg.get("on_violation", {})
        self.log_file: Optional[str] = safety_cfg.get("safety_log_file")

        self.input_guardrail = InputGuardrail(config)
        self.output_guardrail = OutputGuardrail(config)

        self.safety_events: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("safety")

        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Run input guardrail on a user query.

        Returns:
          - safe (bool)
          - query (str): original or sanitized query
          - violations (list)
          - action (str): "allow", "warn", or "refuse"
          - message (str): user-facing explanation if refused
        """
        if not self.enabled:
            return {"safe": True, "query": query, "violations": [], "action": "allow"}

        result = self.input_guardrail.validate(query)
        violations = result.get("violations", [])
        action = result.get("action", "allow")

        is_safe = action == "allow"

        if not is_safe and self.log_events:
            self._log_event("input", query, violations, is_safe, action)

        out = {
            "safe": is_safe,
            "query": result.get("sanitized_input", query),
            "violations": violations,
            "action": action,
        }

        if not is_safe:
            out["message"] = self.on_violation.get(
                "message",
                "This query cannot be processed due to safety policies."
            )

        return out

    def check_output_safety(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run output guardrail on a generated response.

        Returns:
          - safe (bool)
          - response (str): original, sanitized, or refusal message
          - violations (list)
          - action (str): "allow", "sanitize", or "refuse"
        """
        if not self.enabled:
            return {"safe": True, "response": response, "violations": [], "action": "allow"}

        result = self.output_guardrail.validate(response, sources)
        violations = result.get("violations", [])
        action = result.get("action", "allow")

        is_safe = action == "allow"

        if not is_safe and self.log_events:
            self._log_event("output", response, violations, is_safe, action)

        if action == "refuse":
            final_response = self.on_violation.get(
                "message",
                "The generated response was blocked by safety policies."
            )
        elif action == "sanitize":
            final_response = result.get("sanitized_output", response)
        else:
            final_response = response

        return {
            "safe": is_safe,
            "response": final_response,
            "violations": violations,
            "action": action,
        }

    def _log_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool,
        action: str,
    ):
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "action": action,
            "violations": violations,
            "content_preview": (content[:100] + "...") if len(content) > 100 else content,
        }
        self.safety_events.append(event)
        self.logger.warning(
            "Safety event: type=%s action=%s violations=%d",
            event_type,
            action,
            len(violations),
        )
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as exc:
                self.logger.error("Failed to write safety log: %s", exc)

    def get_safety_events(self) -> List[Dict[str, Any]]:
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])
        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0.0,
        }

    def clear_events(self):
        self.safety_events = []
