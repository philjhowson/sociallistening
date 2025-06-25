from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
import shared_functions
import rag_functions
import argparse
import rag_functions
import os
import sys

path_to_processed = 'data/processed'
path_to_rag = 'data/RAG'

def build_prompt():
    """
    Builds the initial prompt for the LLM and the refine prompt used for chaining prompts and refining responses.
    """

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

    """
    Checks for the RAG prompt files and exists if they do not exist. The loading function will alert the
    user which files are missing.
    """

    if not os.path.exists(f"{path_to_processed}/RAG_initial_prompt.pkl") or not os.path.exists(f"{path_to_processed}/RAG_refine_prompt.pkl"):
        sys.exit(f"Please ensure you generate the initial and refine prompts using the --build function and that "
                 f"the files are present in {path_to_processed}.")
        
    """
    Loads in the embedding model and the FAISS archive, sets up the retriever and then generates the docs
    based on the question the user asked.
    """

    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    archive = FAISS.load_local(f"{path_to_rag}/complete_faiss_archive", embeddings = embedding_model,
                               allow_dangerous_deserialization = True)

    retriever = archive.as_retriever(search_kwargs = {'k': k})
    docs = retriever.invoke(question)

    """
    This processes the metadata and batches. Because I felt it makes more sense to give the LLM more than
    one comment at a time to ensure more context and better, quicker summaries, I connect 100 comments at
    a time and produce a list that will be passed using the state dictionary. Each chunk is processed
    and metadata is appended in the 'Sources' section which includes the index in the data frame, the
    original text, the source, and the date. Date is converted to .isoformat() for .json serialization.
    """

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

    """
    Sets up the initial state dictionary and initiates the workflow graph using the State class.
    then adds nodes for the initial summary and the refine summary nodes. An edge is created
    at the initial_summary node and conditional edges are drawn between initial_summary
    and refine_summary node based on the should_refine function. The graph is then compiled.
    """

    state = {'contents': batches,
             'index': 0,
             'summary': '',
             'question': question}

    graph = StateGraph(rag_functions.State)
    graph.add_node('generate_initial_summary', rag_functions.generate_initial_summary)
    graph.add_node('refine_summary', rag_functions.refine_summary)
    graph.add_edge(START, 'generate_initial_summary')
    graph.add_conditional_edges('generate_initial_summary', rag_functions.should_refine)
    graph.add_conditional_edges('refine_summary', rag_functions.should_refine)
    app = graph.compile()

    """
    The answer is generated using the run_langgraph() function and the answer is printed
    to the console. The final answer is saved to the batched metadata as is the hash
    generated based on the question. The entire metadata for the batch is saved as a .json.
    A hash function was used to generate the file names to avoid collisions or multiple
    files potentially having the same name.
    """
    
    answer = rag_functions.run_langgraph(app, state)
    print('Answer:', answer)

    key = shared_functions.hash_question(question)
    batched_metadata['Answer'] = answer
    batched_metadata['Hash'] = key

    shared_functions.safe_saver(batched_metadata, f"{path_to_rag}/{key}.json")

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