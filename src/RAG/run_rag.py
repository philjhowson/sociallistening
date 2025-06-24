from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from typing import List, TypedDict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import shared_functions
import argparse
import os

import asyncio
import operator
from typing import List, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

path_to_processed = 'data/processed'

def build_prompt():

    initial_prompt_template = """
        You are an expert assistant. Use the provided context to answer the question as accurately and thoroughly as possible.
        You will later receive additional chunks of information that should be used to refine and expand your answer.

        Respond in a clear, structured manner. If the answer is not present in the context, state that explicitly.
        If the context is empty, say so.

        Your response must include:

        1. A detailed summary of relevant information in **at least 10 sentences** but **no more than 20 sentences**.
        2. A brief section on **Pain points** – negative sentiments, complaints, or challenges.
        3. A brief section on **Drivers** – positive sentiments, motivations, or benefits.

        Do not fabricate or infer beyond the provided content. Disregard irrelevant or off-topic information.

        ---
        Context:
        {context}

        Question:
        {question}

        Answer:
        """

    refine_prompt_template = """
        You are an expert assistant refining a previous answer based on new context. Use this new information to improve,
        refine, expand, or correct your prior response. Preserve useful content from your previous answer but revise
        it to reflect new insights.

        Respond in a clear, structured manner. If the answer is still not present in the context, say so.
        If the context is empty, state that explicitly.

        Your updated response must include:

        1. A detailed summary of relevant information in **at least 10 sentences** but **no more than 20 sentences**.
        2. A brief section on **Pain points** – negative sentiments, complaints, or challenges.
        3. A brief section on **Drivers** – positive sentiments, motivations, or benefits.

        Do not fabricate or make unsupported assumptions. Focus only on relevant information.

        ---
        Existing Answer:
        {existing_answer}

        New Context:
        {context}

        Question:
        {question}

        Refined Answer:
        """

    shared_functions.safe_saver(initial_prompt_template, f"{path_to_processed}/RAG_initial_prompt.pkl")
    shared_functions.safe_saver(initial_prompt_template, f"{path_to_processed}/RAG_refine_prompt.pkl")

def RAG(question, k = 500):

    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    archive = FAISS.load_local(f"{path_to_processed}/faiss_archive", embeddings = embedding_model,
                               allow_dangerous_deserialization = True)

    load_dotenv()
    api_key = os.getenv('OPENAI_KEY')

    retriever = archive.as_retriever(search_kwargs={'k': k})
    docs = retriever.invoke(question)

    batched_metadata = {'Question' : question,
                        'Answer' : '',
                        'Hash' : '',
                        'Sources' : {}}
    
    batches = []

    for doc_batch in range(0, len(docs), 100):
        batch_docs = docs[doc_batch : doc_batch + 100]
        combined_text = "\n\n".join([doc.metadata.get('text', 'Unknown') for doc in batch_docs])

        batches.append(combined_text)

        for doc in batch_docs:

            index = doc.metadata.get('index', 'Unknown')
            date = doc.metadata.get('date', 'Unknown')

            batched_metadata['Sources'][index] = {
                'index': index,
                'text': doc.metadata.get('text', 'Unknown'),
                'source': doc.metadata.get('source', 'Unknown'),
                'date': date.isoformat()}

    llm = ChatOpenAI(openai_api_key = api_key, model_name = "gpt-4o", temperature = 0.2)
    initial_prompt_template = shared_functions.safe_loader(f"{path_to_processed}/RAG_initial_prompt.pkl")
    refine_prompt_template = shared_functions.safe_loader(f"{path_to_processed}/RAG_refine_prompt.pkl")

    class State(TypedDict):
        contents: List[str]
        index: int
        summary: str

    state = {'contents': batches,
             'index': 0,
             'summary': ''}

    summarize_prompt = ChatPromptTemplate([('human', initial_prompt_template)])
    initial_summary_chain = summarize_prompt | llm | StrOutputParser()

    refine_prompt = ChatPromptTemplate([('human', refine_prompt_template)])
    refine_summary_chain = refine_prompt | llm | StrOutputParser()

    async def generate_initial_summary(state: State, config: RunnableConfig):
        summary = await initial_summary_chain.ainvoke({'context' : state['contents'][0],
                                                       'question' : question}, config)
        return {'summary': summary, 'index': 1}

    async def refine_summary(state: State, config: RunnableConfig):
        content = state['contents'][state['index']]
        summary = await refine_summary_chain.ainvoke(
            {'existing_answer': state['summary'], 'context': content,
             'question' : question},
            config,
        )
        return {'summary': summary, 'index': state['index'] + 1}

    def should_refine(state: State) -> Literal['refine_summary', END]:
        return END if state['index'] >= len(state['contents']) else 'refine_summary'

    graph = StateGraph(State)
    graph.add_node('generate_initial_summary', generate_initial_summary)
    graph.add_node('refine_summary', refine_summary)
    graph.add_edge(START, 'generate_initial_summary')
    graph.add_conditional_edges('generate_initial_summary', should_refine)
    graph.add_conditional_edges('refine_summary', should_refine)
    app = graph.compile()

    def run_langgraph(app, state):
        async def inner():
            final_state = await app.ainvoke(state)
            return final_state['summary']
        return asyncio.run(inner())
    
    answer = run_langgraph(app, state)
    print('Answer:', answer)

    key = shared_functions.hash_question(question)
    batched_metadata['Answer'] = answer
    batched_metadata['Hash'] = key

    shared_functions.safe_saver(batched_metadata, f"data/RAG/{key}.json")

def run_functions(function, q, k):

    match function:
        case 'build':
            build_prompt()
        case 'query':
            RAG(question = q, k = k)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Build prompt and query ChatGPT.')
    parser.add_argument('--function', help = 'build to build a prompt and query to run the RAG for ChatGPT.')
    parser.add_argument('--q', default = None, help = 'write your question for the data here.')
    parser.add_argument('--k', default = 500, help = 'How many text chunks should be retrieved by the RAG model.')

    args = parser.parse_args()

    run_functions(args.function, args.q, args.k)