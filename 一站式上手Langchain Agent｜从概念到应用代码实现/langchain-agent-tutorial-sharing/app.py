import streamlit as st
import os
from langchain_openai import ChatOpenAI
from utils import create_tools, create_agent


st.set_page_config(page_title="LangChain Agents Tutorial", page_icon=None, layout="wide")

st.title("Langchain Agents")
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_PROJECT"] = st.secrets['LANGCHAIN_PROJECT']
# Define tabs
tab1, tab2, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "What is LangChain agent?", 
        "LangChain Agent vs. Chain",
        # "The main goal of using agents",
        "Agent Types",
        "Tools",
        "Agent Executor",
        "Defining prompt",
        "Langchain Agent Demo",
        "Agent Playground"
    ]
)


with tab1:
	what = """
	### What is LangChain agent?
	The idea behind the agent in LangChain is to utilize a language model along with a sequence of actions to take. The agent is using a reasoning engine to determine which actions to take to get a result.

	Agents are crucial for handling tasks ranging from simple automated responses to complex, context-aware interactions.

	For example, you may have an agent integrated with Google Search, Wikipedia and OpenAI LLM. With given agent tools, they can search for results in Google, then use retrieved context in Wikipedia tool to find detailed information and expand the context.

	Bear in mind that you have to put clearly defined instructions to make sure that the agent will invoke tools in a proper order.

	The illustration presents the example of agent and his components:

	"""
	ai_agent = """
### What is an AI agent?
AI agents are software programs that have been around since the 1980s. 

An AI agent works by perceiving its environment, processing information, and taking action to achieve specific goals or tasks.
"""
	st.markdown(ai_agent)
	image0 = 'img/AI-Agent-Workflow.png'
	st.image(image0)

	st.markdown("""
### Structure of an AI Agent
""")
	st.image('img/Structure-of-an-AI-Agent.png')
	
	st.markdown("""
At its core, an AI agent is made up of four components: **the environment, sensors, actuators, and the decision-making mechanism**.

**1. Environment**
			 
The environment refers to the area or domain in which an AI agent operates. It can be a physical space, like a factory floor, or a digital space, like a website.

**2. Sensors**
			 
Sensors are the tools that an AI agent uses to perceive its environment. These can be cameras, microphones, or any other sensory input that the AI agent can use to understand what is happening around it.

**3. Actuators**
			 
Actuators are the tools that an AI agent uses to interact with its environment. These can be things like robotic arms, computer screens, or any other device the AI agent can use to change the environment.

**4. Decision-making mechanism**
			 
A decision-making mechanism is the brain of an AI agent. It processes the information gathered by the sensors and decides what action to take using the actuators. The decision-making mechanism is where the real magic happens.

AI agents use various decision-making mechanisms, such as rule-based systems, expert systems, and neural networks, to make informed choices and perform tasks effectively.

**5. Learning system(Optional)**
			 
The learning system enables the AI agent to learn from its experiences and interactions with the environment. It uses techniques like reinforcement learning, supervised learning, and unsupervised learning to improve the performance of the AI agent over time.
			 """)

	st.markdown(what)

	image1 = "img/llm_agents.png"
	st.image(image1)

with tab2:
	agent_vs_chain = """
	### LangChain Agent vs. Chain
The core idea of agents is to use an LLM to choose a sequence of actions to take. 

In chains, a sequence of actions is hardcoded (in code). 

In agents, a language model is used as a reasoning engine to determine which actions to take and in which order.
	"""
	st.markdown(agent_vs_chain)
	
	# image2 = "img/llm_agents_loop.png"
	# st.image(image2)


with tab4:
	agent_types_1 = """
LangChain categorizes agents based on several dimensions:

- Model type;

- Support for chat history;

- Multi-input tools;

- Parallel function calling;

- Required model parameters.

It is important to choose the option that fits to your use case:

### 1. OpenAI functions

There are certain models fine-tuned where input is a bit different than usual. There are special functions that can be called and the role of this agent is to determine when it should be invoked. This agent is designed to work with this kind of OpenAI model. It supports chat history.

### 2. OpenAI tools

This agent is designed to work with OpenAI tools, so its role is to interact and determine whether to use e.g. image generation tool or another built-in one. The main difference between OpenAI function is the fact that the function is trying to find the best fitting algorithm/part of an algorithm to do better reasoning, while OpenAI tool is about built-in tools like image generation, and executing code. It supports chat history.

### 3. XML Agent

There are models, where reasoning/writing XML is on a very advanced level (a good example is Anthropic Claude's model). If you're operating on XML files, that might be the right one to be considered. It supports chat history.

### 4. JSON Chat Agent

Several LLMs available in the market are particularly handy when it comes to reading JSON. JSON is also a very common standard of some e.g. entity representation. If you're building some kind of integration that operates on JSON files and the model is supporting it, you can try to use this agent. It supports chat history.

### 5. Structured chat

Intended for multi-input tools. It supports chat history.

### 6. ReAct agent

Made for simple models (LLM - not conversational). It supports chat history.

### 7. Self-ask with search

This kind of agent supports only one tool as an input. The main goal is to divide your query into smaller ones, use tools to get the answer, and then combine it into a full answer to your question. This kind of agent doesn’t support chat history.
"""

	st.header("Agent Types")
	st.text("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
	st.markdown(agent_types_1)
	st.text("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
	st.header("When to Use")
	image3 = "img/when_to_use.png"
	st.image(image3)

# Tools
with tab5: 
	st.markdown("""
		### Tools
Tools are the main components of agents that perform individual tasks. It may be web search, vector database search, or any other action. You can choose from many complete tools done by the community or write your own.

LangChain also features a highly useful capability to create tools from retrievers: """)
	
	st.code("""
		tool = create_retriever_tool(
		   retriever,
		   name="companies_database",
		   description="Useful when you need to find information about company."
		)
		# Where retriever might be created from Chroma in-memory database
		retriever = Chroma.from_documents(documents_list, embedding_function).as_retriever()

		""")
	st.markdown("""It is very important to create a good description of retriever tool like in this example because agent step decision mechanism is based on that. 
If your description is missing or incomplete, it might result in skipping the action taken by the agent.
It’s worth mentioning that the tool name should be also unique.

### Toolkits
Toolkits are the combination of tools with predefined actions that can be used in our agent. Usually, they're assembled to the specific domain. There are different examples:

- CSV agent toolkit to operate on `CSV` files (read, write),

- Github agent toolkit that has implemented different operations on Github, e.g. creating new issues, creating new pull requests, and so on.

Compared to the tools, toolkits have an implementation of several actions.
""")

# Agent Executor
with tab6:
	st.markdown("""
		### Agent Executor
From LangChain v0.1.0 version, the recommended way of creating new agents is to use AgentExecutor. You can easily define your executor by passing agents and tools.
The old way of using initialize_agent is marked as deprecated from v0.1.0 version.

		agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
		""")

# Defining prompt
with tab7:
	st.markdown("""
		### Defining prompt
LangChain v0.1.0 version provided a new approach of initializing agents. Instead of using initialize_agent, we have to use a clearly defined method for every type. There is an additional parameter called prompt. We can use the default prompt (you can refer to the documentation to see a prompt for every agent).

Example of default prompt for OpenAI functions agent:
		
		prompt = hub.pull("hwchase17/react-chat")

		""")
	st.markdown("""
		**react-chat prompt**
			
			Assistant is a large language model trained by OpenAI.

			Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. 
			As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

			Assistant is constantly learning and improving, and its capabilities are constantly evolving. 
			It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. 
			Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

			Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
		""")

# Agent Code Demo
with tab8:
	st.header("Langchain Agent Code Demo - Create your Agent in Langchain!")
	st.code("""
		# Langchain Agents

!pip install langchain openai python-dotenv requests duckduckgo-search langchainhub

!pip install -U langchain-openai

### 1. Importing Necessary Libraries

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

from langchain import hub
from langchain.agents import AgentExecutor, create_xml_agent

### 2. Load API Key

import os
from google.colab import userdata
from getpass import getpass

os.environ['OPENAI_API_KEY'] = 'xx'

### 3. Setting Up the DuckDuckGo Search Tool

ddg_search = DuckDuckGoSearchResults()

### 4. Parsing HTML Content

def parse_html(content) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    text_content_with_links = soup.get_text()
    return text_content_with_links

### 5. Fetching Web Page Content

def fetch_web_page(url: str) -> str:
  HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
}
  response = requests.get(url, headers=HEADERS)
  return parse_html(response.content)

### 6. Creating the Web Fetcher Tool

web_fetch_tool = Tool.from_function(
    func=fetch_web_page,
    name="WebFetcher",
    description="Fetches the content of a web page"
)

### 7. Setting Up the Summarizer

prompt_template = "Summarize the following content: {content}"
llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

summarize_tool = Tool.from_function(
    func=llm_chain.run,
    name="Summarizer",
    description="Summarizes a web page"
)

### 9. Initializing the Agent

from langchain.agents import create_openai_tools_agent

tools = [ddg_search, web_fetch_tool, summarize_tool]
prompt = hub.pull("hwchase17/openai-tools-agent")

# Openai Tools Agent
agent = create_openai_tools_agent(llm, tools, prompt)

### 10. Running the Agent

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what is Agent in LangChain? Please give a use case for the explanation"})

agent_executor.invoke({"input": "what is the difference between the Agent and Chain in LangChain? Please give a use case for the explanation according to the content you searched"})

### Test with different Agent Types

# Note: The self-ask with search agent only needs one tool, see the below code:

```
# ddg_search_ = [DuckDuckGoSearchResults(name="Intermediate Answer")]

# # ddg_search_ = DuckDuckGoSearchResults()
# prompt = hub.pull("hwchase17/self-ask-with-search")

# # Construct the Self Ask With Search Agent
# agent = create_self_ask_with_search_agent(llm, ddg_search_, prompt)
# # Create an agent executor by passing in the agent and tools
# agent_executor = AgentExecutor(agent=agent, tools=ddg_search_, verbose=True, handle_parsing_errors=True)
```

from langchain.chains import LLMChain
from langchain.agents import Tool
#llms
openai_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")

#tools
def create_tools(llm):

    ddg_search = DuckDuckGoSearchResults()

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

    tools = [ddg_search, web_fetch_tool, summarize_tool]

    return tools

from typing import Dict, Sequence
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from langchain.agents import (create_json_chat_agent,
                              create_react_agent, \
                              create_xml_agent, \
                              create_structured_chat_agent,
                              create_openai_tools_agent, \
                              create_openai_functions_agent,
                              create_xml_agent,
                              create_self_ask_with_search_agent)

import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import AgentType, AgentExecutor

def create_agent(
    agent_type: str,
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: PromptTemplate = None
) -> AgentExecutor:
    Create a LangChain agent executor.

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
    # try:
    #     print(prompt.get_prompts()[0].template)
    # except:
    #     print(prompt.messages)
    # finally:
    #     pass

    # Return an agent executor by passing in the agent and tools
    if agent_type=='self_ask_search':
      ddg_search_ = [DuckDuckGoSearchResults(name="Intermediate Answer")]

      prompt = hub.pull("hwchase17/self-ask-with-search")

      # Construct the Self Ask With Search Agent
      agent = create_self_ask_with_search_agent(llm, ddg_search_, prompt)
      # Create an agent executor by passing in the agent and tools
      return AgentExecutor(agent=agent, tools=ddg_search_, verbose=True, handle_parsing_errors=True)

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

agent_type_dict = {"openai_funcs": create_openai_functions_agent,
                   "openai_tools": create_openai_tools_agent,
                   "xml": create_xml_agent,
                   "json_chat":create_json_chat_agent ,
                   "structured": create_structured_chat_agent,
                   "react": create_react_agent,
                   "self_ask_search": create_self_ask_with_search_agent
                  }

llm = openai_llm

tools = create_tools(llm)
# agent = create_agent('xml', llm, tools)

query = "What is the value of cash and cash equivalents for Tesla in 2022?"

for agent_type_key in agent_type_dict.keys():
  agent = create_agent(agent_type_key, llm, tools)
  results = agent.invoke({"input": query})
  print(f'Agent Type: {agent_type_key}\n\nQuestion: {query}\n\nAnswer👇👇👇: \n{results}')
		""")

# Test with different agent types
with tab9:
	# Define agent name map
	agent_name_map = {"OpenAI Functions": 'openai_funcs', 
                      "OpenAI Tools": 'openai_tools',  
                      "XML Agent": 'xml',
                      "JSON CHAT Agent": 'json_chat',
                      "Structured Chat Agent": 'structured',
                      "ReAct": 'react',
                      "Self-ask with search": 'self_ask_search'}

	# Tools and LLM
	model_name = "gpt-3.5-turbo-16k"                     
	llm = ChatOpenAI(model=model_name)  
	tools = create_tools(llm)

	options = list(agent_name_map.keys())
	default_ix = options.index('OpenAI Tools')

	# Create selectbox for agent type  
	selected_agent = st.selectbox("Select an agent", options, default_ix)

	# Input for query
	query = st.text_input("Enter your query", placeholder='e.g. What is Langchain Agent?')

	# Create agent and execute
	if selected_agent and query:
		with st.spinner('Agent running...'): 
			agent = create_agent(agent_name_map[selected_agent], llm, tools) 
			output = agent.invoke({"input": query})["output"]
			# Display output
			st.write(output)
