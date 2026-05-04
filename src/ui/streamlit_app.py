"""
Streamlit Web Interface
Run with: streamlit run src/ui/streamlit_app.py
      or: python main.py --mode web
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import concurrent.futures
import json
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
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "orchestrator" not in st.session_state:
        config = load_config()
        st.session_state.config = config
        try:
            st.session_state.orchestrator = AutoGenOrchestrator(config)
        except Exception as exc:
            st.error(f"Failed to initialize orchestrator: {exc}")
            st.session_state.orchestrator = None
    if "show_traces" not in st.session_state:
        st.session_state.show_traces = True
    if "show_safety_log" not in st.session_state:
        st.session_state.show_safety_log = False
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


def run_query(query: str):
    """Process a query and store result in session state."""
    orch = st.session_state.orchestrator
    if orch is None:
        result = {
            "query": query,
            "error": "Orchestrator not initialized.",
            "response": "System not ready. Check your .env configuration.",
            "metadata": {},
        }
    else:
        result = orch.process_query(query.strip())

    st.session_state.latest_result = result
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query.strip(),
        "result": result,
    })


def display_safety_banner(result: Dict[str, Any]):
    blocked = result.get("metadata", {}).get("input_blocked", False)
    output_action = result.get("metadata", {}).get("output_action", "allow")
    safety_events = result.get("safety_events", [])

    if blocked:
        st.error(
            "**Safety Policy Violation — INPUT BLOCKED**  \n"
            "Your query was stopped before reaching the agents.  \n"
            + (safety_events[-1]["violations"][0]["reason"] if safety_events and safety_events[-1].get("violations") else "")
        )
        return

    if output_action == "refuse":
        st.error("**Safety Policy Violation — OUTPUT REFUSED**  \nThe generated response was blocked by output guardrails.")
    elif output_action == "sanitize":
        st.warning("**Safety Notice — OUTPUT SANITIZED**  \nSome content was redacted to comply with safety policies.")

    for event in safety_events:
        for v in event.get("violations", []):
            fn = st.error if v.get("severity") == "high" else st.warning
            fn(f"**Safety Event** | `{v.get('category','?')}` | `{v.get('severity','?')}` — {v.get('reason','')}")


def display_agent_traces(result: Dict[str, Any]):
    history = result.get("conversation_history", [])
    agent_icons = {"Planner": "📋", "Researcher": "🔍", "Writer": "✍️", "Critic": "🔬"}
    agent_sources = {"Planner", "Researcher", "Writer", "Critic"}

    shown = [m for m in history if m.get("source") in agent_sources and m.get("content", "").strip()]
    if not shown:
        st.info("No agent messages to display. The model endpoint may have been temporarily unavailable.")
        return

    for msg in shown:
        source = msg["source"]
        icon = agent_icons.get(source, "🤖")
        with st.expander(f"{icon} **{source}**", expanded=(source == "Writer")):
            st.markdown(msg["content"])


def display_citations(result: Dict[str, Any]):
    citations = result.get("metadata", {}).get("citations", [])
    if not citations:
        st.info("No URLs were extracted from agent outputs.")
        return
    for i, url in enumerate(citations, 1):
        st.markdown(f"**[{i}]** [{url}]({url})")


def display_result(result: Dict[str, Any]):
    if not result:
        return

    display_safety_banner(result)

    if result.get("metadata", {}).get("input_blocked"):
        return

    tab_resp, tab_traces, tab_cites, tab_meta = st.tabs(
        ["📄 Response", "🔍 Agent Traces", "📚 Citations", "📊 Metadata"]
    )

    with tab_resp:
        response = result.get("response", "")
        if response:
            st.markdown(response)
        else:
            st.warning("No response was generated.")

    with tab_traces:
        if st.session_state.show_traces:
            display_agent_traces(result)
        else:
            st.info("Enable 'Show Agent Traces' in the sidebar.")

    with tab_cites:
        display_citations(result)

    with tab_meta:
        meta = result.get("metadata", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Messages", meta.get("num_messages", 0))
        c2.metric("Sources Found", meta.get("num_sources", 0))
        c3.metric("Agents", len(meta.get("agents_involved", [])))
        if meta.get("agents_involved"):
            st.markdown("**Pipeline:** " + " → ".join(meta["agents_involved"]))
        action = meta.get("output_action", "allow")
        st.markdown(f"**Output safety action:** `{action}`")


def display_evaluation_panel():
    if not st.session_state.latest_result:
        st.info("Run a query first.")
        return

    result = st.session_state.latest_result
    query = result.get("query", "")
    response = result.get("response", "")

    if not response or len(response) < 30:
        st.warning("Response is too short to evaluate meaningfully.")
        return

    if st.button("Run LLM-as-a-Judge on last response", type="primary"):
        with st.spinner("Running judge (may take 15-30 seconds)..."):
            try:
                from src.evaluation.judge import LLMJudge
                judge = LLMJudge(st.session_state.config)

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    eval_result = pool.submit(
                        asyncio.run,
                        judge.evaluate(query=query, response=response)
                    ).result()

                st.success(f"Overall Score: **{eval_result['overall_score']:.4f}** (0–1 scale)")
                col1, col2 = st.columns(2)
                col1.metric("Rubric Score", f"{eval_result['rubric_score']:.4f}")
                col2.metric("Holistic Score", f"{eval_result['holistic_score']:.4f}")

                st.markdown("**Criterion Scores (Prompt A — Rubric):**")
                for crit, data in eval_result.get("criterion_scores", {}).items():
                    score = data.get("score", 0.0)
                    bar = "▓" * int(score * 20) + "░" * (20 - int(score * 20))
                    st.markdown(f"- **{crit}**: `{score:.4f}` {bar}")
                    if data.get("reasoning"):
                        st.caption(data["reasoning"])

                if eval_result.get("holistic_feedback"):
                    st.markdown("**Holistic Feedback (Prompt B — Domain Expert):**")
                    st.info(eval_result["holistic_feedback"])

                Path("outputs").mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"outputs/judge_output_{ts}.json", "w") as f:
                    json.dump({"query": query, "evaluation": eval_result}, f, indent=2)

            except Exception as exc:
                st.error(f"Judge error: {exc}")
                st.info("The model endpoint may be temporarily overloaded (502). Wait a few seconds and try again.")


def display_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        st.session_state.show_traces = st.checkbox("Show Agent Traces", value=st.session_state.show_traces)
        st.session_state.show_safety_log = st.checkbox("Show Safety Event Log", value=st.session_state.show_safety_log)

        st.divider()
        st.title("📊 Statistics")
        orch = st.session_state.orchestrator
        safety_count = 0
        if orch:
            try:
                safety_count = len(orch.safety_manager.get_safety_events())
            except Exception:
                pass
        st.metric("Queries Run", len(st.session_state.history))
        st.metric("Safety Events", safety_count)

        st.divider()
        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.latest_result = None
            st.rerun()

        st.divider()
        st.markdown("### System")
        config = st.session_state.get("config", {})
        st.markdown(f"**Topic:** {config.get('system', {}).get('topic', 'HCI')}")
        st.markdown(f"**Model:** {config.get('models', {}).get('default', {}).get('name', 'Qwen3-8B')}")

        st.divider()
        st.markdown("### Safety Policies")
        st.markdown(
            "- Harmful content\n"
            "- Prompt injection\n"
            "- Off-topic queries\n"
            "- PII exposure\n"
            "- Misinformation"
        )

        if st.session_state.latest_result:
            st.divider()
            st.markdown("### Export")
            data = json.dumps(st.session_state.latest_result, indent=2, default=str)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("Download Session JSON", data=data,
                               file_name=f"session_{ts}.json", mime="application/json")


def display_safety_log():
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
        with st.expander(f"Event {i}: {event.get('type','?').upper()} | {event.get('action','?').upper()} | {event.get('timestamp','')[:19]}"):
            st.markdown(f"**Preview:** {event.get('content_preview', '')}")
            for v in event.get("violations", []):
                st.markdown(f"- **{v.get('category','?')}** ({v.get('severity','?')}): {v.get('reason','')}")


def main():
    st.set_page_config(
        page_title="Multi-Agent HCI Research Assistant",
        page_icon="🤖",
        layout="wide",
    )
    initialize_session_state()

    # Process any pending query (from example buttons) before rendering
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        with st.spinner(f"Researching: {query[:60]}..."):
            run_query(query)

    st.title("🤖 Multi-Agent HCI Research Assistant")
    st.markdown("Powered by AutoGen + Qwen3-8B. Ask about HCI, Explainable AI, UX, or related research.")

    display_sidebar()

    col_main, col_side = st.columns([2, 1])

    with col_main:
        query_input = st.text_area(
            "Research query:",
            height=100,
            placeholder="e.g., What are the key principles of explainable AI for novice users?",
        )

        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query_input.strip():
                with st.spinner("Agents working on your query..."):
                    run_query(query_input)
            else:
                st.warning("Please enter a query.")

        # Always display the latest result
        if st.session_state.latest_result:
            st.divider()
            st.markdown(f"**Query:** *{st.session_state.latest_result.get('query', '')}*")
            display_result(st.session_state.latest_result)

        # LLM Judge panel
        if st.session_state.latest_result:
            st.divider()
            st.markdown("### 🧑‍⚖️ LLM-as-a-Judge Evaluation")
            display_evaluation_panel()

        # History
        if len(st.session_state.history) > 1:
            with st.expander("📜 Query History"):
                for item in reversed(st.session_state.history):
                    st.markdown(f"**[{item['timestamp']}]** {item['query']}")

        # Safety log
        if st.session_state.show_safety_log:
            st.divider()
            st.markdown("### 🛡️ Safety Event Log")
            display_safety_log()

    with col_side:
        st.markdown("### 💡 Example Queries")
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
                st.session_state.pending_query = ex
                st.rerun()

        st.divider()
        st.markdown("### ℹ️ How It Works")
        st.markdown(
            "1. **Planner** breaks down the query\n"
            "2. **Researcher** uses web + paper search\n"
            "3. **Writer** synthesizes with citations\n"
            "4. **Critic** approves or requests revision\n"
            "5. **Safety guardrails** check input and output"
        )

        st.divider()
        st.markdown("### 🛡️ Safety Policies")
        st.markdown(
            "Blocked: harmful content, prompt injection, "
            "off-topic queries, PII exposure, misinformation"
        )


if __name__ == "__main__":
    main()
