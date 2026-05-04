"""
Input Guardrail
Checks user inputs for safety violations before they reach the agent pipeline.

Policy categories enforced:
  1. harmful_content    - requests for violence, weapons, hacking, etc.
  2. prompt_injection   - attempts to override system instructions
  3. off_topic_queries  - requests unrelated to HCI / AI research
  4. length_violations  - inputs that are too short or too long
"""

import re
from typing import Dict, Any, List


HARMFUL_KEYWORDS = [
    "how to hack", "how to attack", "how to exploit", "how to make a bomb",
    "how to make a weapon", "malware", "ransomware", "phishing", "ddos",
    "doxxing", "how to hurt", "how to kill", "suicide method", "self-harm method",
]

INJECTION_PATTERNS = [
    r"ignore (all |previous )?instructions",
    r"disregard (your |all |previous )?instructions",
    r"forget everything",
    r"you are now",
    r"act as if you (are|were)",
    r"pretend (you are|to be)",
    r"new (system )?prompt",
    r"override (safety|guidelines|policy)",
    r"bypass (safety|filter|guardrail)",
    r"jailbreak",
    r"developer mode",
    r"\bsystem:\s",
    r"\[system\]",
    r"ignore the above",
    r"your real instructions",
]

OFF_TOPIC_BLOCK = [
    "write me a poem about",
    "tell me a joke",
    "give me a recipe",
    "what is the weather",
    "help me with my taxes",
    "write my essay",
    "do my homework",
]

HCI_KEYWORDS = [
    "hci", "human-computer interaction", "user interface", "ux", "ui design",
    "usability", "accessibility", "interaction design", "user experience",
    "explainable ai", "xai", "transparency", "fairness", "ethics",
    "machine learning", "artificial intelligence", "deep learning",
    "natural language", "chatbot", "visualization", "augmented reality",
    "virtual reality", "mobile", "design", "prototype", "evaluation",
    "cognitive", "perception", "gesture", "voice", "touch", "research",
    "study", "survey", "review", "literature", "paper", "framework",
    "method", "approach", "system", "model", "algorithm", "neural",
    "computer", "technology", "data", "analysis", "user", "people",
    "interface", "interaction", "experience", "testing", "performance",
    "latest", "recent", "trend", "development", "what", "how", "why",
    "compare", "explain", "describe", "overview", "best practices",
]


class InputGuardrail:
    """
    Guardrail for validating user input before it reaches the agent pipeline.

    Checks for:
      - Prohibited harmful content (category: harmful_content)
      - Prompt injection attempts (category: prompt_injection)
      - Off-topic queries not related to HCI/AI (category: off_topic_queries)
      - Length violations (category: length_violation)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        safety_cfg = config.get("safety", {})
        input_rules = safety_cfg.get("input_rules", {})

        self.min_length: int = input_rules.get("min_length", 5)
        self.max_length: int = input_rules.get("max_length", 2000)

        self.blocked_keywords: List[str] = [
            kw.lower() for kw in input_rules.get("blocked_keywords", [])
        ]
        self.harmful_keywords: List[str] = [
            kw.lower() for kw in input_rules.get("harmful_keywords", [])
        ] + HARMFUL_KEYWORDS

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate a user query for safety.

        Returns a dict with:
          - valid (bool): whether the query is safe to process
          - violations (list): list of violation dicts
          - sanitized_input (str): query after sanitization (same as input if valid)
          - action (str): "allow", "sanitize", or "refuse"
        """
        violations: List[Dict[str, Any]] = []
        text_lower = query.lower().strip()

        violations.extend(self._check_length(query))
        violations.extend(self._check_harmful_content(text_lower))
        violations.extend(self._check_prompt_injection(text_lower))
        violations.extend(self._check_off_topic(text_lower))

        high_severity = any(v["severity"] == "high" for v in violations)
        any_violation = len(violations) > 0

        if high_severity:
            action = "refuse"
        elif any_violation:
            action = "warn"
        else:
            action = "allow"

        return {
            "valid": not any_violation,
            "violations": violations,
            "sanitized_input": query.strip(),
            "action": action,
        }

    def _check_length(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        if len(text) < self.min_length:
            violations.append({
                "validator": "length",
                "category": "length_violation",
                "reason": "Query is too short to be meaningful.",
                "severity": "low",
            })
        if len(text) > self.max_length:
            violations.append({
                "validator": "length",
                "category": "length_violation",
                "reason": f"Query exceeds maximum length of {self.max_length} characters.",
                "severity": "medium",
            })
        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for keyword in self.harmful_keywords:
            if keyword in text:
                violations.append({
                    "validator": "harmful_content",
                    "category": "harmful_content",
                    "reason": f"Query contains a prohibited term related to harmful activity.",
                    "severity": "high",
                    "matched": keyword,
                })
                break
        for keyword in self.blocked_keywords:
            if keyword in text and not any(v["category"] == "harmful_content" for v in violations):
                violations.append({
                    "validator": "blocked_keyword",
                    "category": "harmful_content",
                    "reason": "Query contains a system-blocked phrase.",
                    "severity": "high",
                    "matched": keyword,
                })
                break
        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({
                    "validator": "prompt_injection",
                    "category": "prompt_injection",
                    "reason": "Query appears to contain a prompt injection attempt.",
                    "severity": "high",
                    "matched_pattern": pattern,
                })
                break
        return violations

    def _check_off_topic(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for phrase in OFF_TOPIC_BLOCK:
            if phrase in text:
                violations.append({
                    "validator": "relevance",
                    "category": "off_topic_queries",
                    "reason": "Query appears unrelated to HCI or AI research.",
                    "severity": "medium",
                })
                return violations

        has_hci_term = any(kw in text for kw in HCI_KEYWORDS)
        if not has_hci_term and len(text.split()) > 4:
            violations.append({
                "validator": "relevance",
                "category": "off_topic_queries",
                "reason": (
                    "Query does not appear to relate to HCI, AI, or research topics. "
                    "This system is designed for HCI research assistance."
                ),
                "severity": "low",
            })
        return violations
