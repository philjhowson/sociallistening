from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import shared_functions
import torch
import pandas as pd

path_to_processed = 'data/processed'

def build_database():

    data = pd.read_parquet(f"{path_to_processed}/final_cleaned_data.parquet")

    custom_stops = shared_functions.safe_loader('data/raw/custom_stopwords.pkl')
    whitelist = shared_functions.safe_loader('data/raw/whitelisted_characters.pkl')

    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    documents = []

    print('Chunking comments for archive...')

    for index, (idx, row) in enumerate(data.iterrows()):

        if index % 1000 == 0:
            print(f"Processing row {index}/{len(data)}...")

        text = row['text']
        date = row['date']
        source = row['source']
        cleaned_text = row['cleaned_text']
        chunks = splitter.split_text(text)

        for chunk in chunks:

            cleaned_chunk = shared_functions.pipeline(chunk, custom_stops, whitelist)

            documents.append(Document(page_content = cleaned_chunk,
                                      metadata = {'index' : index,
                                                  'raw_index' : idx,
                                                  'text' : chunk,
                                                  'date' : date,
                                                  'source' : source}))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_function = HuggingFaceEmbeddings(model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                                               model_kwargs = {'device': device})

    print('Building FAISS Archive...')
    faiss_archive = FAISS.from_documents(documents, embedding_function)
    faiss_archive.save_local(f"{path_to_processed}/faiss_archive")
    print('Archive successfully built and saved!')

if __name__ == '__main__':
    build_database()