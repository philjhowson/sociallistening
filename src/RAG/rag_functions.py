import asyncio
from typing import List, Literal, TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END
import shared_functions
from dotenv import load_dotenv
import os
import sys

path_to_processed = 'data/processed'

"""
Defines the state, contents, index, summary, and question for:
    contents: a list of all the text to be passed to the LLM
    index: index of the chunk of context that is being processed
    summary: the current LLM summary
    question: the question for the LLM
"""

class State(TypedDict):
    contents: List[str]
    index: int
    summary: str
    question: str

"""
Loads in the env and API and then loads in the initial prompt and refinement prompt.
Sets up the API and llm for 'gpt-4o'. Temperature is set to 0.2 to allow some paraphrasing
but reduces the chance of hallucinations.
"""

load_dotenv()
api_key = os.getenv('OPENAI_KEY')

llm = ChatOpenAI(openai_api_key = api_key, model_name = "gpt-4o", temperature = 0.2)

"""
Tries to load in the prompt templates, if they are not found a print will trigger, alerting the
user to which files are missing. The files are set to none and the summary chains are not
build as they will trigger an error. This is the desired behaviour if someone wants to use
the --build function. If this obtains and someone is running the --rag and these files
are not present, it will exit the script with a notification.
"""

try:
    initial_prompt_template = shared_functions.safe_loader(f"{path_to_processed}/RAG_initial_prompt.pkl")
    refine_prompt_template = shared_functions.safe_loader(f"{path_to_processed}/RAG_refine_prompt.pkl")
except FileNotFoundError:
    initial_prompt_template = None
    refine_prompt_template = None

if initial_prompt_template:
    summarize_prompt = ChatPromptTemplate([('human', initial_prompt_template)])
    initial_summary_chain = summarize_prompt | llm | StrOutputParser()

if refine_prompt_template:
    refine_prompt = ChatPromptTemplate([('human', refine_prompt_template)])
    refine_summary_chain = refine_prompt | llm | StrOutputParser()

"""
Triggers the initial call to the LLM and returns the summary and index 1.
"""

async def generate_initial_summary(state: State, config: RunnableConfig):
    question = state['question']
    summary = await initial_summary_chain.ainvoke({'context': state['contents'][0],
                                                   'question': question}, config)
    return {'summary': summary, 'index': 1}

"""
Triggers the refine call and gives the previous summary, the context, and the question.
Returns the new summary and increases the index by one.
"""

async def refine_summary(state: State, config: RunnableConfig):
    question = state['question']
    content = state['contents'][state['index']]
    summary = await refine_summary_chain.ainvoke(
        {'existing_answer': state['summary'], 'context': content,
         'question': question},
        config)
    return {'summary': summary, 'index': state['index'] + 1}

"""
Determines if more refinement is necessary by checking the length of
state['index'] against the length of state['contents'].
"""
    
def should_refine(state: State) -> Literal['refine_summary', END]:
        return END if state['index'] >= len(state['contents']) else 'refine_summary'

"""
Runs the built app and pauses the thread while it awaits an answer.
Returns the final summary as an answer to the question.
"""

def run_langgraph(app, state):
    async def inner():
        final_state = await app.ainvoke(state)
        return final_state['summary']
    return asyncio.run(inner())
