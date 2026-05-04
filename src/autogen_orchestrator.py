"""
AutoGen-Based Orchestrator

Workflow:
  1. Safety input check
  2. Planner   -> creates a research plan
  3. Researcher -> gathers evidence using web/paper search tools
  4. Writer    -> synthesizes findings into a cited response
  5. Critic    -> evaluates quality and terminates or requests revision
  6. Safety output check
"""

import asyncio
import concurrent.futures
import logging
from typing import Any, Dict, List, Optional

from autogen_agentchat.messages import TextMessage, ToolCallMessage, ToolCallResultMessage

from src.agents.autogen_agents import create_research_team
from src.guardrails.safety_manager import SafetyManager


class AutoGenOrchestrator:
    """
    Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat.
    Wraps the research team with input and output safety guardrails.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")

        self.logger.info("Initializing safety manager...")
        self.safety_manager = SafetyManager(config)

        self.logger.info("Creating research team...")
        self.team = create_research_team(config)
        self.logger.info("Research team ready.")

        self.workflow_trace: List[Dict[str, Any]] = []

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent system.

        Steps:
          1. Input safety check
          2. Multi-agent research pipeline
          3. Output safety check

        Returns a dict with:
          - query, response, conversation_history, metadata, safety_events
        """
        self.logger.info("Processing query: %s", query[:80])

        # --- 1. Input safety ---
        input_safety = self.safety_manager.check_input_safety(query)
        if not input_safety["safe"] and input_safety["action"] == "refuse":
            return {
                "query": query,
                "response": input_safety.get("message", "Request refused by safety policy."),
                "conversation_history": [],
                "metadata": {
                    "num_messages": 0,
                    "num_sources": 0,
                    "agents_involved": [],
                    "safety_events": self.safety_manager.get_safety_events(),
                    "input_blocked": True,
                },
                "safety_events": self.safety_manager.get_safety_events(),
            }

        safe_query = input_safety.get("query", query)

        # --- 2. Run agents ---
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        self._process_query_async(safe_query, max_rounds),
                    ).result()
            else:
                result = loop.run_until_complete(
                    self._process_query_async(safe_query, max_rounds)
                )
        except Exception as exc:
            self.logger.error("Error in agent pipeline: %s", exc, exc_info=True)
            return {
                "query": query,
                "error": str(exc),
                "response": f"An error occurred while processing your query: {exc}",
                "conversation_history": [],
                "metadata": {"error": True},
                "safety_events": self.safety_manager.get_safety_events(),
            }

        # --- 3. Output safety ---
        output_safety = self.safety_manager.check_output_safety(
            result.get("response", ""),
        )
        result["response"] = output_safety["response"]
        result["metadata"]["safety_events"] = self.safety_manager.get_safety_events()
        result["metadata"]["output_action"] = output_safety["action"]
        result["safety_events"] = self.safety_manager.get_safety_events()

        return result

    async def _process_query_async(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        task_message = (
            f"Research Query: {query}\n\n"
            "Please work together to answer this query comprehensively:\n"
            "1. Planner: Create a structured research plan with specific search terms.\n"
            "2. Researcher: Use web_search and paper_search tools to gather evidence.\n"
            "3. Writer: Synthesize findings into a well-cited response with a References section.\n"
            "4. Critic: Evaluate quality and either approve (TERMINATE) or request revision."
        )

        task_result = await self.team.run(task=task_message)

        messages = self._extract_messages(task_result)
        final_response = self._find_final_response(messages)

        return self._build_result(query, messages, final_response)

    def _extract_messages(self, task_result) -> List[Dict[str, Any]]:
        """Convert AutoGen TaskResult messages to plain dicts."""
        messages = []
        for msg in task_result.messages:
            source = getattr(msg, "source", "unknown")
            if isinstance(msg, (ToolCallMessage,)):
                content_parts = []
                for call in (msg.content if isinstance(msg.content, list) else []):
                    name = getattr(call, "name", "tool")
                    args = getattr(call, "arguments", "")
                    content_parts.append(f"[Tool call: {name}({args})]")
                content = "\n".join(content_parts) if content_parts else str(msg.content)
            elif isinstance(msg, (ToolCallResultMessage,)):
                content_parts = []
                for res in (msg.content if isinstance(msg.content, list) else []):
                    content_parts.append(str(getattr(res, "content", res)))
                content = "\n".join(content_parts) if content_parts else str(msg.content)
            else:
                raw = getattr(msg, "content", "")
                content = raw if isinstance(raw, str) else str(raw)

            messages.append({"source": source, "content": content})
        return messages

    def _find_final_response(self, messages: List[Dict[str, Any]]) -> str:
        """Return the last Writer message, or the last Critic message, or the last message."""
        for msg in reversed(messages):
            if msg["source"] == "Writer" and len(msg["content"]) > 50:
                return msg["content"].replace("DRAFT COMPLETE", "").strip()
        for msg in reversed(messages):
            if msg["source"] == "Critic" and len(msg["content"]) > 20:
                return msg["content"].replace("TERMINATE", "").strip()
        if messages:
            return messages[-1]["content"].replace("TERMINATE", "").strip()
        return ""

    def _build_result(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        final_response: str,
    ) -> Dict[str, Any]:
        plan = ""
        research_findings: List[str] = []
        critique = ""

        for msg in messages:
            src = msg["source"]
            content = msg["content"]
            if src == "Planner" and not plan:
                plan = content
            elif src == "Researcher":
                research_findings.append(content)
            elif src == "Critic":
                critique = content

        import re
        all_text = " ".join(m["content"] for m in messages)
        urls = list(dict.fromkeys(re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', all_text)))

        agents_involved = list(dict.fromkeys(m["source"] for m in messages if m["source"] != "unknown"))

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

    def get_agent_descriptions(self) -> Dict[str, str]:
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources using search tools",
            "Writer": "Synthesizes findings into coherent, well-cited responses",
            "Critic": "Evaluates quality and provides feedback or approves the response",
        }

    def visualize_workflow(self) -> str:
        return (
            "\nAutoGen Research Workflow:\n"
            "  User Query\n"
            "      -> Safety Input Check\n"
            "      -> Planner  (creates research plan)\n"
            "      -> Researcher (web_search + paper_search tools)\n"
            "      -> Writer  (synthesizes + cites sources)\n"
            "      -> Critic  (approves or requests revision)\n"
            "      -> Safety Output Check\n"
            "      -> Final Response to User\n"
        )
