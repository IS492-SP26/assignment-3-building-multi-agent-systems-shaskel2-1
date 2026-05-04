"""
AutoGen-Based Orchestrator
"""

import asyncio
import concurrent.futures
import logging
import re
from typing import Any, Dict, List

from src.agents.autogen_agents import create_research_team
from src.guardrails.safety_manager import SafetyManager


def _run_async(coro):
    """Run a coroutine safely from any thread (including Streamlit's ScriptRunner)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from model output."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()


class AutoGenOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")
        self.safety_manager = SafetyManager(config)
        self.logger.info("Creating research team...")
        self.team = create_research_team(config)
        self.logger.info("Research team ready.")

    def process_query(self, query: str) -> Dict[str, Any]:
        self.logger.info("Processing query: %s", query[:80])

        # 1. Input safety
        input_safety = self.safety_manager.check_input_safety(query)
        if not input_safety["safe"] and input_safety["action"] == "refuse":
            return {
                "query": query,
                "response": input_safety.get("message", "Request refused by safety policy."),
                "conversation_history": [],
                "metadata": {
                    "num_messages": 0, "num_sources": 0,
                    "agents_involved": [], "input_blocked": True,
                },
                "safety_events": self.safety_manager.get_safety_events(),
            }

        safe_query = input_safety.get("query", query)

        # 2. Run agents in a dedicated thread with its own event loop
        try:
            result = _run_async(self._process_query_async(safe_query))
        except Exception as exc:
            self.logger.error("Agent pipeline error: %s", exc, exc_info=True)
            return {
                "query": query,
                "error": str(exc),
                "response": (
                    f"The agent pipeline encountered an error: {exc}\n\n"
                    "This is often caused by a temporary issue with the model endpoint (502/503). "
                    "Please wait a moment and try again."
                ),
                "conversation_history": [],
                "metadata": {"error": True},
                "safety_events": self.safety_manager.get_safety_events(),
            }

        # 3. Output safety
        output_safety = self.safety_manager.check_output_safety(result.get("response", ""))
        result["response"] = output_safety["response"]
        result["metadata"]["safety_events"] = self.safety_manager.get_safety_events()
        result["metadata"]["output_action"] = output_safety["action"]
        result["safety_events"] = self.safety_manager.get_safety_events()
        return result

    async def _process_query_async(self, query: str) -> Dict[str, Any]:
        task_message = (
            f"Research Query: {query}\n\n"
            "Please work together to answer this query:\n"
            "1. Planner: Create a research plan with specific search terms.\n"
            "2. Researcher: Use web_search and paper_search tools to gather evidence.\n"
            "3. Writer: Synthesize findings into a well-cited response with a References section.\n"
            "4. Critic: Evaluate quality and say TERMINATE to approve, or request revision."
        )
        task_result = await self.team.run(task=task_message)
        messages = self._extract_messages(task_result)
        final_response = self._find_final_response(messages, query)
        return self._build_result(query, messages, final_response)

    def _extract_messages(self, task_result) -> List[Dict[str, Any]]:
        messages = []
        for msg in task_result.messages:
            source = getattr(msg, "source", "unknown")
            raw = getattr(msg, "content", "")
            if isinstance(raw, str):
                content = _strip_thinking(raw)
            elif isinstance(raw, list):
                parts = []
                for item in raw:
                    if isinstance(item, str):
                        parts.append(_strip_thinking(item))
                    elif hasattr(item, "content"):
                        parts.append(_strip_thinking(str(item.content)))
                    elif hasattr(item, "name") and hasattr(item, "arguments"):
                        parts.append(f"[Tool call: {item.name}({item.arguments})]")
                    else:
                        parts.append(str(item))
                content = "\n".join(parts)
            else:
                content = _strip_thinking(str(raw))
            messages.append({"source": source, "content": content})
        return messages

    def _find_final_response(self, messages: List[Dict[str, Any]], original_query: str) -> str:
        # Look for Writer output first
        for msg in reversed(messages):
            if msg["source"] == "Writer" and len(msg["content"]) > 50:
                return msg["content"].replace("DRAFT COMPLETE", "").strip()
        # Fall back to Critic
        for msg in reversed(messages):
            if msg["source"] == "Critic" and len(msg["content"]) > 20:
                return msg["content"].replace("TERMINATE", "").strip()
        # Fall back to any non-task agent message
        agent_sources = {"Planner", "Researcher", "Writer", "Critic"}
        for msg in reversed(messages):
            if msg["source"] in agent_sources and len(msg["content"]) > 20:
                return msg["content"].replace("TERMINATE", "").strip()
        # No agent output at all — give a clear error rather than echoing the task
        return (
            "The research agents did not produce a response. This is usually caused by a "
            "temporary issue with the model endpoint (e.g., 502 Bad Gateway). "
            "Please try again in a few seconds."
        )

    def _build_result(self, query: str, messages: List[Dict[str, Any]], final_response: str) -> Dict[str, Any]:
        plan, critique = "", ""
        research_findings: List[str] = []
        agent_sources = {"Planner", "Researcher", "Writer", "Critic"}

        for msg in messages:
            src = msg["source"]
            if src == "Planner" and not plan:
                plan = msg["content"]
            elif src == "Researcher":
                research_findings.append(msg["content"])
            elif src == "Critic":
                critique = msg["content"]

        all_text = " ".join(m["content"] for m in messages)
        urls = list(dict.fromkeys(re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', all_text)))
        agents_involved = list(dict.fromkeys(
            m["source"] for m in messages if m["source"] in agent_sources
        ))

        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": len(urls),
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": agents_involved,
                "citations": urls,
            },
        }

    def visualize_workflow(self) -> str:
        return (
            "\nAutoGen Research Workflow:\n"
            "  User Query -> Safety Input Check\n"
            "      -> Planner -> Researcher (web+paper search)\n"
            "      -> Writer -> Critic -> Safety Output Check\n"
            "      -> Final Response\n"
        )
