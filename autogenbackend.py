'''
    core logic for the multi agent literature review assistant built with the
    autogen, agentchat stack(>0.4). it exposes a single public coroutine run_litrev() that drives a 
    two-agent team

    search_agent- crafts an arxiv query and fetches paper via the provided
                    arxiv_search tool
    
    summarizer- writes a short markdown literature review from the selected paper


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