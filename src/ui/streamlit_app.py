"""
Streamlit Web Interface
Run with: streamlit run src/ui/streamlit_app.py
      or: python main.py --mode web
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import asyncio
import yaml
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from src.autogen_orchestrator import AutoGenOrchestrator

load_dotenv()


def load_config() -> Dict[str, Any]:
    p = Path("config.yaml")
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f)
    return {}


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "orchestrator" not in st.session_state:
        config = load_config()
        try:
            st.session_state.orchestrator = AutoGenOrchestrator(config)
            st.session_state.config = config
        except Exception as exc:
            st.error(f"Failed to initialize orchestrator: {exc}")
            st.session_state.orchestrator = None
            st.session_state.config = config
    if "show_traces" not in st.session_state:
        st.session_state.show_traces = True
    if "show_safety_log" not in st.session_state:
        st.session_state.show_safety_log = False
    if "total_safety_events" not in st.session_state:
        st.session_state.total_safety_events = 0


def process_query(query: str) -> Dict[str, Any]:
    orch = st.session_state.orchestrator
    if orch is None:
        return {
            "query": query,
            "error": "Orchestrator not initialized.",
            "response": "System not ready. Check your .env configuration.",
            "metadata": {},
        }
    try:
        result = orch.process_query(query)
        return result
    except Exception as exc:
        return {
            "query": query,
            "error": str(exc),
            "response": f"An error occurred: {exc}",
            "metadata": {},
        }


def display_safety_banner(result: Dict[str, Any]):
    """Show a prominent warning if any safety action was triggered."""
    safety_events = result.get("safety_events", [])
    blocked = result.get("metadata", {}).get("input_blocked", False)
    output_action = result.get("metadata", {}).get("output_action", "allow")

    if blocked:
        st.error(
            "**Safety Policy Violation (INPUT BLOCKED)**  \n"
            "Your query was blocked before reaching the agents.  \n"
            f"Reason: {safety_events[-1]['violations'][0]['reason'] if safety_events and safety_events[-1]['violations'] else 'Policy violation'}"
        )
        return

    if output_action == "refuse":
        st.error(
            "**Safety Policy Violation (OUTPUT REFUSED)**  \n"
            "The generated response was blocked by output safety guardrails.  \n"
            "The system refused to return content that may violate safety policies."
        )
    elif output_action == "sanitize":
        st.warning(
            "**Safety Notice (OUTPUT SANITIZED)**  \n"
            "Some content in the response was redacted to comply with safety policies."
        )

    for event in safety_events:
        violations = event.get("violations", [])
        for v in violations:
            category = v.get("category", "unknown")
            reason = v.get("reason", "Unknown reason")
            severity = v.get("severity", "medium")
            color = "error" if severity == "high" else "warning"
            getattr(st, color)(
                f"**Safety Event** | Category: `{category}` | Severity: `{severity}`  \n{reason}"
            )


def display_agent_traces(result: Dict[str, Any]):
    """Render per-agent conversation traces in expandable panels."""
    history = result.get("conversation_history", [])
    if not history:
        st.info("No agent trace available.")
        return

    agent_colors = {
        "Planner": "blue",
        "Researcher": "green",
        "Writer": "orange",
        "Critic": "red",
    }
    agent_icons = {
        "Planner": "📋",
        "Researcher": "🔍",
        "Writer": "✍️",
        "Critic": "🔬",
    }

    for msg in history:
        source = msg.get("source", "Unknown")
        content = msg.get("content", "")
        if not content.strip():
            continue
        icon = agent_icons.get(source, "🤖")
        with st.expander(f"{icon} **{source}**", expanded=(source == "Writer")):
            st.markdown(content)


def display_citations(result: Dict[str, Any]):
    """Display deduplicated URLs gathered by the Researcher."""
    citations = result.get("metadata", {}).get("citations", [])
    if not citations:
        st.info("No citations extracted.")
        return
    for i, url in enumerate(citations, 1):
        st.markdown(f"**[{i}]** [{url}]({url})")


def display_response(result: Dict[str, Any]):
    """Main response display with tabs for response, traces, citations."""
    if "error" in result and not result.get("response"):
        st.error(f"Error: {result['error']}")
        return

    # Safety banners first
    display_safety_banner(result)

    tab_resp, tab_traces, tab_citations, tab_meta = st.tabs(
        ["📄 Response", "🔍 Agent Traces", "📚 Citations", "📊 Metadata"]
    )

    with tab_resp:
        response = result.get("response", "")
        if response:
            st.markdown(response)
        else:
            st.warning("No response generated.")

    with tab_traces:
        if st.session_state.show_traces:
            display_agent_traces(result)
        else:
            st.info("Enable 'Show Agent Traces' in the sidebar to view this panel.")

    with tab_citations:
        display_citations(result)

    with tab_meta:
        meta = result.get("metadata", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Messages Exchanged", meta.get("num_messages", 0))
        c2.metric("Sources Gathered", meta.get("num_sources", 0))
        c3.metric("Agents Involved", len(meta.get("agents_involved", [])))

        if meta.get("agents_involved"):
            st.markdown("**Agents:** " + " → ".join(meta["agents_involved"]))

        output_action = meta.get("output_action", "allow")
        action_label = {"allow": "Allowed", "sanitize": "Sanitized", "refuse": "Refused"}.get(
            output_action, output_action
        )
        st.markdown(f"**Output Safety Action:** `{action_label}`")


def display_evaluation_panel():
    """Inline LLM-as-a-Judge panel for the last query."""
    if not st.session_state.history:
        st.info("Run a query first to see evaluation results.")
        return

    last = st.session_state.history[-1]
    result = last.get("result", {})
    response = result.get("response", "")
    query = last.get("query", "")

    if not response or len(response) < 20:
        st.warning("Response too short to evaluate.")
        return

    if st.button("Run LLM-as-a-Judge on last response", type="primary"):
        with st.spinner("Running judge evaluation..."):
            try:
                config = st.session_state.config
                from src.evaluation.judge import LLMJudge
                judge = LLMJudge(config)
                eval_result = asyncio.run(judge.evaluate(query=query, response=response))

                st.success(f"Overall Score: **{eval_result['overall_score']:.4f}** (0-1 scale)")
                st.markdown(f"Rubric Score: `{eval_result['rubric_score']:.4f}` | "
                            f"Holistic Score: `{eval_result['holistic_score']:.4f}`")

                st.markdown("**Criterion Scores:**")
                for crit, data in eval_result.get("criterion_scores", {}).items():
                    score = data.get("score", 0.0)
                    bar = int(score * 20)
                    st.markdown(f"- **{crit}**: `{score:.4f}` {'▓' * bar}{'░' * (20 - bar)}")
                    if data.get("reasoning"):
                        st.caption(data["reasoning"])

                if eval_result.get("holistic_feedback"):
                    st.markdown("**Holistic Feedback (Expert View):**")
                    st.info(eval_result["holistic_feedback"])

                # Save judge output
                Path("outputs").mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"outputs/judge_output_{ts}.json", "w") as f:
                    json.dump({"query": query, "evaluation": eval_result}, f, indent=2)
            except Exception as exc:
                st.error(f"Judge error: {exc}")


def display_sidebar():
    with st.sidebar:
        st.title("Settings")
        st.session_state.show_traces = st.checkbox("Show Agent Traces", value=st.session_state.show_traces)
        st.session_state.show_safety_log = st.checkbox("Show Safety Event Log", value=st.session_state.show_safety_log)

        st.divider()
        st.title("Statistics")
        orch = st.session_state.orchestrator
        safety_events = 0
        if orch:
            try:
                safety_events = len(orch.safety_manager.get_safety_events())
            except Exception:
                pass
        st.metric("Queries Run", len(st.session_state.history))
        st.metric("Safety Events", safety_events)

        st.divider()
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.markdown("### System Info")
        config = st.session_state.get("config", {})
        st.markdown(f"**Topic:** {config.get('system', {}).get('topic', 'HCI Research')}")
        st.markdown(f"**Model:** {config.get('models', {}).get('default', {}).get('name', 'Qwen/Qwen3-8B')}")

        safety_cfg = config.get("safety", {})
        cats = safety_cfg.get("prohibited_categories", [])
        if cats:
            st.markdown("**Safety Categories:**")
            for cat in cats:
                st.markdown(f"  - `{cat}`")


def display_safety_log():
    """Show all safety events recorded this session."""
    orch = st.session_state.orchestrator
    if not orch:
        return
    try:
        events = orch.safety_manager.get_safety_events()
    except Exception:
        events = []

    if not events:
        st.info("No safety events recorded this session.")
        return

    for i, event in enumerate(reversed(events), 1):
        evt_type = event.get("type", "unknown")
        action = event.get("action", "allow")
        ts = event.get("timestamp", "")
        violations = event.get("violations", [])
        with st.expander(f"Event {i}: {evt_type.upper()} | {action.upper()} | {ts[:19]}"):
            st.markdown(f"**Preview:** {event.get('content_preview', '')}")
            for v in violations:
                st.markdown(f"- **{v.get('category', 'unknown')}** ({v.get('severity', '?')}): {v.get('reason', '')}")


def main():
    st.set_page_config(
        page_title="Multi-Agent HCI Research Assistant",
        page_icon="🤖",
        layout="wide",
    )
    initialize_session_state()

    st.title("Multi-Agent HCI Research Assistant")
    st.markdown(
        "Powered by AutoGen + Qwen3-8B. Ask about HCI, Explainable AI, UX, or related research topics."
    )

    display_sidebar()

    col_main, col_side = st.columns([2, 1])

    with col_main:
        query = st.text_area(
            "Research query:",
            height=100,
            placeholder="e.g., What are the key principles of explainable AI for novice users?",
        )

        if st.button("Search", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Agents working on your query..."):
                    result = process_query(query.strip())
                    st.session_state.history.append(
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "query": query.strip(),
                            "result": result,
                        }
                    )
                st.divider()
                display_response(result)
            else:
                st.warning("Please enter a query.")

        if st.session_state.history:
            st.divider()
            st.markdown("### LLM-as-a-Judge Evaluation")
            display_evaluation_panel()

        if st.session_state.history:
            with st.expander("Query History"):
                for item in reversed(st.session_state.history):
                    st.markdown(f"**[{item['timestamp']}]** {item['query']}")

        if st.session_state.show_safety_log:
            st.divider()
            st.markdown("### Safety Event Log")
            display_safety_log()

    with col_side:
        st.markdown("### Example Queries")
        examples = [
            "What are the key principles of explainable AI for novice users?",
            "How has AR usability evolved in the last 5 years?",
            "What are ethical considerations in using AI for education?",
            "Compare methods for measuring user experience in mobile apps",
            "What are best practices for voice interfaces for elderly users?",
            "How do cultural factors influence mobile app design?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                with st.spinner("Agents working on your query..."):
                    result = process_query(ex)
                    st.session_state.history.append(
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "query": ex,
                            "result": result,
                        }
                    )
                st.rerun()

        st.divider()
        st.markdown("### How It Works")
        st.markdown(
            "1. **Planner** breaks down the query\n"
            "2. **Researcher** uses web + paper search tools\n"
            "3. **Writer** synthesizes with citations\n"
            "4. **Critic** approves or requests revision\n"
            "5. **Safety guardrails** check input and output"
        )

        st.divider()
        st.markdown("### Safety Policies")
        st.markdown(
            "Blocked categories:\n"
            "- Harmful content\n"
            "- Prompt injection\n"
            "- Off-topic queries\n"
            "- PII exposure\n"
            "- Misinformation"
        )

        if st.session_state.history:
            st.divider()
            st.markdown("### Export Last Session")
            last = st.session_state.history[-1]
            export_data = json.dumps(last, indent=2, default=str)
            st.download_button(
                "Download Session JSON",
                data=export_data,
                file_name=f"session_{last['timestamp'].replace(':', '-')}.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
