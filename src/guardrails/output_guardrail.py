"""
Output Guardrail
Inspects system outputs for safety violations before they are shown to the user.

Policy categories enforced:
  1. pii_exposure      - personal identifiable information in output
  2. harmful_content   - unsafe instructions, violent/hateful language
  3. misinformation    - unverified or clearly false factual claims (keyword heuristic)
"""

import re
from typing import Dict, Any, List, Optional


PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
    "phone_us": r'\b(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card": r'\b(?:\d{4}[\s\-]?){3}\d{4}\b',
    "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
}

HARMFUL_PHRASES = [
    r"step[s]? to (make|build|create|synthesize) (a |an )?(bomb|weapon|virus|malware|exploit)",
    r"how to (hack|attack|break into|compromise) (a |an )?",
    r"instruc(tion)?s? (for|on) (making|building|creating) (a |an )?(bomb|weapon)",
    r"(kill|murder|harm|hurt) (yourself|someone|people)",
]

UNSUPPORTED_CLAIM_PATTERNS = [
    r"it is (a fact|proven|confirmed) that .{0,80} will (cure|solve|eliminate)",
    r"100% (effective|guaranteed|accurate|safe)",
    r"scientists (have proven|confirmed) that .{0,80} (is dangerous|causes cancer|causes death)",
]


class OutputGuardrail:
    """
    Guardrail for validating system output before it reaches the user.

    Checks for:
      - PII leakage (category: pii_exposure)
      - Harmful or dangerous content (category: harmful_content)
      - Misinformation signals (category: misinformation)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def validate(self, response: str, sources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Validate an agent output for safety.

        Returns:
          - valid (bool)
          - violations (list)
          - sanitized_output (str): output with PII redacted if safe to show
          - action (str): "allow", "sanitize", or "refuse"
        """
        violations: List[Dict[str, Any]] = []

        violations.extend(self._check_pii(response))
        violations.extend(self._check_harmful_content(response))
        violations.extend(self._check_misinformation(response))

        high_severity = any(v["severity"] == "high" for v in violations)
        any_violation = len(violations) > 0

        if high_severity:
            action = "refuse"
        elif any_violation:
            action = "sanitize"
        else:
            action = "allow"

        sanitized = self._sanitize(response, violations) if any_violation else response

        return {
            "valid": not any_violation,
            "violations": violations,
            "sanitized_output": sanitized,
            "action": action,
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append({
                    "validator": "pii",
                    "category": "pii_exposure",
                    "pii_type": pii_type,
                    "reason": f"Output contains {pii_type.replace('_', ' ')} information.",
                    "severity": "high",
                    "matches": matches,
                    "pattern": pattern,
                })
        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        text_lower = text.lower()
        for pattern in HARMFUL_PHRASES:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violations.append({
                    "validator": "harmful_content",
                    "category": "harmful_content",
                    "reason": "Output contains potentially harmful instructions or content.",
                    "severity": "high",
                })
                break
        return violations

    def _check_misinformation(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for pattern in UNSUPPORTED_CLAIM_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({
                    "validator": "misinformation",
                    "category": "misinformation",
                    "reason": "Output contains phrasing that may suggest unsupported absolute claims.",
                    "severity": "medium",
                })
                break
        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        sanitized = text
        for v in violations:
            if v.get("validator") == "pii":
                pattern = v.get("pattern", "")
                if pattern:
                    sanitized = re.sub(pattern, "[REDACTED]", sanitized)
        return sanitized
