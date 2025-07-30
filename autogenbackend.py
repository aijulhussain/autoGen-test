'''
    core logic for the multi agent literature review assistant built with the
    autogen, agentchat stack(>0.4). it exposes a single public coroutine run_litrev() that drives a 
    two-agent team

    search_agent- crafts an arxiv query and fetches paper via the provided
                    arxiv_search tool
    
    summarizer- writes a short markdown  abstruct from the selected paper


    the module is deliberately self contained so it can reused in CLI apps, streamlit, FastAPI, gradio, etc. 
'''

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Dict, List

import arxiv 
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import (
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
)

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.ollama import OllamaChatCompletionClient


def arxiv_search(query:str, max_result: int=10)->List[Dict]:
    """
        Return a compact list of arxic papers matching *query*.
        Each elemnt contains: ``title``,``authors``, ``published``, ``summary``, 
        and ``pdf_url``. The helper is wraped as an autogen *FunctionTool* below so it
        can be invoked by agents through the normal tool-use machanism.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_result,
        sort_by=arxiv.SortCriterion.Relevance,
        
    )

    papers: List[Dict]=[]
    for result in client.results(search):
        papers.append(
            {
                "title":result.title,
                "authors":[a.name for a in result.authors],
                "published": result.published.strftime("%y-%m-%d"),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
            }
        )
    return papers


arxiv_tool = FunctionTool(
    arxiv_search,
    description=(
        "Searches arxiv and returns up to *max_results* papers, each containg"
        "title, authors, publication date, abstruct, and pdf_url."
    ),
)


def build_team(model: str="llama3")->RoundRobinGroupChat:
    """ Create and return tow agent *RoundRobinGroupChat"""

    llm_client = OllamaChatCompletionClient(model=model)

    #agent that only calls the arxiv tool and forwards top n papers

    search_agent = AssistantAgent(
        name = "search_agent",
        description="Crafts arXiv queries retrieves candidate papers.",
        system_message=(
            "Given a user topic, think of the best arXiv query and call the "
            "provided tool. Always fetch five times the papers requested so"
            "that you can down-select the most relevent ones. When the tool "
            "returns, choose exactly the number of papers requested and pass"
            "them as concise JSON to the summarizer."
        ),
        tools=[arxiv_tool],
        model_client=llm_client,
        reflect_on_tool_use=True,
    )