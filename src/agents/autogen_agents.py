"""
AutoGen Agent Implementations

Creates the four research agents (Planner, Researcher, Writer, Critic)
and assembles them into a RoundRobinGroupChat team.
"""

import os
from typing import Any, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.tools import FunctionTool
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.tools.web_search import web_search
from src.tools.paper_search import paper_search


def create_model_client(config: Dict[str, Any]) -> OpenAIChatCompletionClient:
    """Create a model client from config (supports vllm and openai providers)."""
    model_cfg = config.get("models", {}).get("default", {})
    provider = model_cfg.get("provider", "vllm")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL") or model_cfg.get("name", "Qwen/Qwen3-8B")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")

    return OpenAIChatCompletionClient(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
    )


def create_planner_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    system_message = config.get("agents", {}).get("planner", {}).get("system_prompt") or (
        "You are a Research Planner specializing in Human-Computer Interaction (HCI) and Explainable AI.\n"
        "When given a research query:\n"
        "1. Identify 3-5 key concepts to investigate.\n"
        "2. Suggest specific search queries for the Researcher.\n"
        "3. Outline the structure of the expected synthesis.\n"
        "Be concise and actionable. End your message with PLAN COMPLETE."
    )
    return AssistantAgent(
        name="Planner",
        model_client=model_client,
        description="Breaks down research queries into actionable search steps.",
        system_message=system_message,
    )


def create_researcher_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    system_message = config.get("agents", {}).get("researcher", {}).get("system_prompt") or (
        "You are a Research Specialist in HCI and Explainable AI.\n"
        "Use the web_search and paper_search tools to gather evidence following the Planner's plan.\n"
        "For each result, record the title, URL, and key finding.\n"
        "Aim for 5-8 diverse sources (mix of academic papers and web sources).\n"
        "Always cite sources with URLs. End your message with RESEARCH COMPLETE."
    )
    web_tool = FunctionTool(
        web_search,
        description=(
            "Search the web for recent articles, blog posts, and general information. "
            "Returns formatted results with titles, URLs, and snippets."
        ),
    )
    paper_tool = FunctionTool(
        paper_search,
        description=(
            "Search Semantic Scholar for academic papers. "
            "Returns papers with authors, years, abstracts, citation counts, and URLs. "
            "Use year_from to filter recent papers (e.g., year_from=2020)."
        ),
    )
    return AssistantAgent(
        name="Researcher",
        model_client=model_client,
        tools=[web_tool, paper_tool],
        description="Gathers evidence from web and academic sources using search tools.",
        system_message=system_message,
    )


def create_writer_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    system_message = config.get("agents", {}).get("writer", {}).get("system_prompt") or (
        "You are a Research Writer specializing in HCI and AI.\n"
        "Synthesize the Researcher's findings into a well-organized, cited response.\n"
        "Structure:\n"
        "  1. Brief overview paragraph\n"
        "  2. Thematic sections with ### headers\n"
        "  3. Inline citations: [Author, Year] or [Source: Title]\n"
        "  4. References section at the end listing all sources with URLs\n"
        "Write at an advanced undergraduate level. End your message with DRAFT COMPLETE."
    )
    return AssistantAgent(
        name="Writer",
        model_client=model_client,
        description="Synthesizes research findings into coherent, well-cited responses.",
        system_message=system_message,
    )


def create_critic_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    system_message = config.get("agents", {}).get("critic", {}).get("system_prompt") or (
        "You are a Research Critic and Quality Reviewer.\n"
        "Evaluate the Writer's response on:\n"
        "  1. Relevance: Does it answer the original query?\n"
        "  2. Evidence: Are claims backed by cited sources?\n"
        "  3. Completeness: Are important aspects covered?\n"
        "  4. Clarity: Is it well-organized and readable?\n"
        "If the response meets quality standards, say TERMINATE.\n"
        "Otherwise, list 1-3 specific improvements needed."
    )
    return AssistantAgent(
        name="Critic",
        model_client=model_client,
        description="Evaluates research quality and either approves or requests revision.",
        system_message=system_message,
    )


def create_research_team(config: Dict[str, Any]) -> RoundRobinGroupChat:
    """Assemble the four-agent research team as a RoundRobinGroupChat."""
    model_client = create_model_client(config)

    planner = create_planner_agent(config, model_client)
    researcher = create_researcher_agent(config, model_client)
    writer = create_writer_agent(config, model_client)
    critic = create_critic_agent(config, model_client)

    termination = TextMentionTermination("TERMINATE")

    return RoundRobinGroupChat(
        participants=[planner, researcher, writer, critic],
        termination_condition=termination,
        max_turns=config.get("system", {}).get("max_iterations", 12),
    )
