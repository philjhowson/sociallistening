from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import shared_functions
import pandas as pd
import argparse
import pickle
import os

path_to_processed = 'data/processed'

def preprocess():

    from data_formatting_pipeline import pipeline

    data = pd.read_parquet('data/raw/youtube_results.parquet')

    custom_stops = shared_functions.safe_loader('data/raw/custom_stopwords.pkl')
    whitelist = shared_functions.safe_loader('data/raw/whitelisted_characters.pkl')

    data = pipeline(data, 'video_title', custom_stops = custom_stops, whitelist = whitelist)
    data = pipeline(data, 'comment_text', custom_stops = custom_stops, whitelist = whitelist,
                    geolocation = False)

    data.to_parquet(f"{path_to_processed}/cleaned_youtube_results.parquet")

def embeddings():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_youtube_results.parquet")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data['cleaned_comment_text'].tolist())

    shared_functions.safe_saver(embeddings, f"{path_to_processed}/youtube_comment_embeddings.pkl")

def cluster(eps = 0.5):

    if os.path.exists(f"{path_to_processed}/cluster_youtube_results.parquet"):
        data = pd.read_parquet(f"{path_to_processed}/cluster_youtube_results.parquet")
    else:
        data = pd.read_parquet(f"{path_to_processed}/cleaned_youtube_results.parquet")
    
    embeddings = shared_functions.safe_loader(f"{path_to_processed}/youtube_comment_embeddings.pkl")

    dbscan = DBSCAN(eps = eps, min_samples = 5, metric = 'cosine')

    clusters = f"clusters_{eps}"
    data[clusters] = dbscan.fit_predict(embeddings)

    os.makedirs(path_to_processed, exist_ok = True)

    columns_to_drop = ['channel_id', 'channel_name', 'video_id', 'country',
                       'lat', 'lon', 'comment_author', 'enough_char']
    
    safe_to_drop = [col for col in columns_to_drop if col in data.columns]

    data.drop(columns = safe_to_drop, inplace = True)

    data.to_parquet(f"{path_to_processed}/cluster_youtube_results.parquet")

def format_youtube(function = None, eps = 0.5):

    if function == 'preprocess':
        preprocess()

    if function == 'embeddings':
        embeddings()

    if function == 'cluster':
        cluster(eps = 0.5)

    if function == 'all':
        preprocess()
        embeddings()
        cluster(eps = 0.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'prepare youtube data for analysis')
    parser.add_argument('--function', required = True, help = 'Options: "preprocess" to prepare data, "embeddings" to creating embeddings, "cluster" to generate clusters (optional argument --eps for this), "all" to do all steps.')
    parser.add_argument('--eps',  type = float, default = 0.5)

    arg = parser.parse_args()

    format_youtube(function = arg.function, eps = arg.eps)
