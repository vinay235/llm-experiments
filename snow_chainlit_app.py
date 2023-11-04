
from typing import Dict, Optional, Union

import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, config_list_from_json
import chainlit as cl
import requests
#from bs4 import BeautifulSoup
import json

#import tiktoken
import openai
import pandas as pd

import autogen
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from snowflake.snowpark import Session
from importlib.metadata import version



WELCOME_MESSAGE = f"""Snowflake Agent Team ❄️👨‍💻⚙️⛓️❄️
\n\n
What can we do for you today?
"""

# Agents
USER_PROXY_NAME = "Admin"
CODING_PLANNER = "Snow Engineer"
CODING_RUNNER = "Snow SQL Agent"
DATA_ANALYZER = "Analysis Agent"


# Snowflake config
connection_parameters = {
    "account": "",
    "user": "",
    "password": "",
    "warehouse": "COMPUTE_WH",
    "database": "DEMO_DB", 
    "schema": "DEMO_SCH" 
}

session = Session.builder.configs(connection_parameters).create()
#print("Current role: " + session.get_current_role() + ", Current schema: " + session.get_fully_qualified_current_schema())



async def ask_helper(func, **kwargs):
    res = await func(**kwargs).send()
    while not res:
        res = await func(**kwargs).send()
    return res

class ChainlitAssistantAgent(AssistantAgent):
    """
    Wrapper for AutoGens Assistant Agent
    """
    def send(
        self,
        message:  Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = True,
    ) -> bool:
        cl.run_sync(
            cl.Message(
                content=f'*Sending message to "{recipient.name}":*\n\n{message}',
                author=self.name,
                language='sql'
            ).send()
        )
        super(ChainlitAssistantAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

class ChainlitUserProxyAgent(UserProxyAgent):
    """
    Wrapper for AutoGens UserProxy Agent. Simplifies the UI by adding CL Actions. 
    """
    def get_human_input(self, prompt: str) -> str:
        if prompt.startswith(
            "Provide feedback to chat_manager. Press enter to skip and use auto-reply"
        ):
            res = cl.run_sync(
                ask_helper(
                    cl.AskActionMessage,
                    content="Continue or provide feedback?",
                    actions=[
                        cl.Action( name="continue", value="continue", label="✅ Continue" ),
                        cl.Action( name="feedback",value="feedback", label="💬 Provide feedback"),
                        cl.Action( name="exit",value="exit", label="🔚 Exit Conversation" )
                    ],
                )
            )
            if res.get("value") == "continue":
                return ""
            if res.get("value") == "exit":
                return "exit"

        reply = cl.run_sync(ask_helper(cl.AskUserMessage, content=prompt, timeout=60))

        return reply["content"].strip()

    def send(
        self,
        message: Union[Dict, str],
        recipient: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = True,
    ):
        cl.run_sync(
            cl.Message(
                content=f'*Sending message to "{recipient.name}"*:\n\n{message}',
                author=self.name,
                #language='sql'
            ).send()
        )
        super(ChainlitUserProxyAgent, self).send(
            message=message,
            recipient=recipient,
            request_reply=request_reply,
            silent=silent,
        )

@cl.on_chat_start
async def on_chat_start():
  try:

    def run_sql(query):
        res = session.sql(f"{query}").collect()
        df = pd.DataFrame(data = res)
        if(len(df) > 0):
            #df.columns = header['name']
            return df.to_markdown(index=False)
        else:
            return "NO ROWS RETURNED"
  

    #config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
    config_list = [
            {
                'model': 'gpt-3.5-turbo-16k-0613',
                'api_key': '',
            }
    ]

    llm_config_code = {
         "temperature": 0,
        "request_timeout": 600,
        "seed": 42,
        "functions": [
        {
            "name" : "run_sql",
            "description" : "running SQL query on Snowflake",
            "parameters" : {
                "type" : "object",
                "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query",
                        }
                    },
                    "required": ["query"],
                },
            },
         ],
    "config_list": config_list   
    }


    coding_assistant = ChainlitAssistantAgent(
        name="SnowEngineers",
        llm_config=llm_config_code,
        system_message="""SnowEngineers. You write Snowflake SQL code to solve tasks. 
            Wrap the code in a code block that specifies the script type. The user can't modify your code. The tables can contain large data. DO not query all rows. ACT RESPONSIBLY.
            You are responsible to check the metadata of the tables by using the SHOW TABLES command and finding the table and columns that is needed.
            Use DESCRIBE TABLE to understand its data and columns. Provide a sample of 10 rows as output along with the query in code block.
            Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. 
            Check the execution result returned by the executor.
            If the result indicates there is an error, fix the error and output the code again. 
            Suggest the full code instead of partial code or code changes. 
            If the error can't be fixed or if the task is not solved even after the code is executed successfully, 
            analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
            """,
    )
    coding_runner = ChainlitUserProxyAgent(
        name="SnowSQLAgent",
        system_message="SnowSQLAgent. Execute the SQL code written by the SnowEngineer and report back the result to the Analysis Agent",
        human_input_mode="NEVER",
        code_execution_config={"last_n_messages": 3, "work_dir": "sql"},
        function_map={
                "run_sql": run_sql,
                #"search": search,
                #"scrape": scrape,
            },
    )
    analysis_agent = ChainlitAssistantAgent(
        name="AnalysisAgent", llm_config=llm_config_code,
        system_message="""Analysis agent. You will always analyse the data outputted by SnowSQLAgent.Convert the data into natural language. Be concise and always summarize the data when possible.
                Communicate with the Admin user when the data is analyzed."""
    )
    user_proxy = ChainlitUserProxyAgent(
        name="Admin",
        system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
        code_execution_config=False, 
    )
    
    cl.user_session.set(USER_PROXY_NAME, user_proxy)
    cl.user_session.set(CODING_PLANNER, coding_assistant)
    cl.user_session.set(CODING_RUNNER, coding_runner)
    cl.user_session.set(DATA_ANALYZER, analysis_agent)
    
    await cl.Message(content=WELCOME_MESSAGE, author="Admin").send()
    
  except Exception as e:
    #print("Error: ", e)
    await cl.Message(content="ERROR: "+e, author="Admin").send()
    pass

@cl.on_message
async def run_conversation(message: cl.Message):
  try:
    TASK = message.content
    print("Task: ", TASK)
    coding_assistant = cl.user_session.get(CODING_PLANNER)
    user_proxy = cl.user_session.get(USER_PROXY_NAME)
    coding_runner = cl.user_session.get(CODING_RUNNER)
    analysis_agent = cl.user_session.get(DATA_ANALYZER)
    
    groupchat = autogen.GroupChat(agents=[user_proxy, coding_assistant, coding_runner, analysis_agent], messages=[], max_round=50)
    manager = autogen.GroupChatManager(groupchat=groupchat)
    
    print("GC messages: ", len(groupchat.messages))
    
    if len(groupchat.messages) == 0:
      await cl.Message(content="Snowflake Connection Success! \nCurrent role: " + session.get_current_role() + ",\nCurrent schema: " + session.get_fully_qualified_current_schema()+",\nCurrent Warehouse"+session.get_current_warehouse()+"\n\n").send()
      await cl.Message(content=f"""Starting agents on task: {TASK}...""").send()
      await cl.make_async(user_proxy.initiate_chat)( manager, message=TASK, )
    else:
      await cl.make_async(user_proxy.send)( manager, message=TASK, )
      
  except Exception as e:
    #print("Error: ", e)
    await cl.Message(content="ERROR: "+e, author="Admin").send()
    pass