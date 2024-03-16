
from typing import Dict, Sequence
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.agents import Tool
from langchain import hub
from langchain.agents import (create_json_chat_agent,
                              create_react_agent, \
                              create_xml_agent, \
                              create_structured_chat_agent,
                              create_openai_tools_agent, \
                              create_openai_functions_agent,
                              create_xml_agent,
                              create_self_ask_with_search_agent)

#tools
def create_tools(llm):
    
    # ddg_search = DuckDuckGoSearchResults()
    _search = TavilySearchResults()
    web_fetch_tool = Tool.from_function(
        func=fetch_web_page,
        name="WebFetcher",
        description="Fetches the content of a web page"
        )
    
    prompt_template = "Summarize the following content: {content}"
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template))

    summarize_tool = Tool.from_function(
        func=llm_chain.run,
        name="Summarizer",
        description="Summarizes a web page")
    
    # tools = [ddg_search, web_fetch_tool, summarize_tool]
    tools = [_search, web_fetch_tool, summarize_tool] 
    return tools
    

def create_agent(
    agent_type: str,
    llm: BaseLanguageModel, 
    tools: Sequence[BaseTool],
    prompt: PromptTemplate = None
) -> AgentExecutor:
    """Create a LangChain agent executor.
    
    Args:
        agent_type: The type of agent to create. Options are:
            - "openai_funcs": OpenAI Functions agent
            - "openai_tools": OpenAI Tools agent 
            - "xml": XML agent
            - "json_chat": JSON Chat agent 
            - "react": React agent
            - "self_ask_search": Self Ask agent with search
        
        llm: The language model to use. Defaults to OpenAI.
        tools: The tools to equip the agent with. 
        prompt: A prompt template to initialize the agent with.
            If none provided, will pull a template from Hub.
            
    Returns:  
        An executor for the created agent.
    """
    # Map agent types to creation functions
    CREATE_AGENT_FUNCS = {"openai_funcs": create_openai_functions_agent, 
                       "openai_tools": create_openai_tools_agent, 
                       "xml": create_xml_agent, 
                       "json_chat":create_json_chat_agent ,
                       "structured": create_structured_chat_agent,
                       "react": create_react_agent,
                       "self_ask_search": create_self_ask_with_search_agent
                      }

    # Get the agent creation function
    create_agent_func = CREATE_AGENT_FUNCS[agent_type]
    
    if not prompt:
        if agent_type=='openai_funcs':
            prompt = hub.pull("hwchase17/openai-functions-agent")
        elif agent_type=='openai_tools':
            prompt = hub.pull("hwchase17/openai-tools-agent")
        elif agent_type=='xml':
            prompt = hub.pull("hwchase17/xml-agent-convo")
        elif agent_type=='json_chat':
            prompt = hub.pull("hwchase17/react-chat-json")
        elif agent_type=='structured':
            prompt = hub.pull("hwchase17/structured-chat-agent")
        elif agent_type=='react':
            prompt = hub.pull("hwchase17/react")
        elif agent_type=='self_ask_search':
            prompt = hub.pull("hwchase17/self-ask-with-search")

    # Return an agent executor by passing in the agent and tools
    if agent_type=='self_ask_search':
    #   ddg_search_ = [DuckDuckGoSearchResults(name="Intermediate Answer")]
      _search_ = [TavilySearchResults(name="Intermediate Answer")]

      prompt = hub.pull("hwchase17/self-ask-with-search")

      # Construct the Self Ask With Search Agent
    #   agent = create_self_ask_with_search_agent(llm, ddg_search_, prompt)
      agent = create_self_ask_with_search_agent(llm, _search_, prompt)
      # Create an agent executor by passing in the agent and tools
    #   return AgentExecutor(agent=agent, tools=ddg_search_, verbose=True, handle_parsing_errors=True)
      return AgentExecutor(agent=agent, tools=_search_, verbose=True, handle_parsing_errors=True)

    else:
      agent = create_agent_func(llm, tools, prompt)
      return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


def parse_html(content) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    text_content_with_links = soup.get_text()
    return text_content_with_links

def fetch_web_page(url: str) -> str:
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'}
    response = requests.get(url, headers=HEADERS)
    return parse_html(response.content)
