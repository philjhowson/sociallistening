from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import shared_functions
import pandas as pd
import argparse
import numpy as np
import os

path_to_processed = 'data/processed'

def preprocess():

    from data_formatting_pipeline import pipeline

    data = pd.read_parquet(f"{path_to_processed}/cleaned_youtube_results.parquet")

    iso_codes = shared_functions.safe_loader('data/raw/iso_codes.pkl')
    custom_stops = shared_functions.safe_loader('data/raw/custom_stopwords.pkl')
    whitelist = shared_functions.safe_loader('data/raw/whitelisted_characters.pkl')

    data = pipeline(data, 'comment_text', iso_codes = iso_codes, custom_stops = custom_stops, whitelist = whitelist,
                    geolocation = False, detect_lang = False)
    data = pipeline(data, 'video_title', iso_codes = iso_codes, custom_stops = custom_stops, whitelist = whitelist,
                    geolocation = False, detect_lang = False)

    data.to_parquet(f"{path_to_processed}/cleaned_youtube_results.parquet")

def embeddings(english = False):

    data = pd.read_parquet(f"{path_to_processed}/cleaned_youtube_results.parquet")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if English:
        model = SentenceTransformer('all-MiniLM-L6-v2', device = device)
    else:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device = device)
    embeddings = model.encode(data['cleaned_comment_text'].tolist())
    embeddings = normalize(embeddings)

    np.savez_compressed(f"{path_to_processed}/youtube_comment_embeddings.npz", embeddings = embeddings)

def cluster(eps = None, size = None):

    if os.path.exists(f"{path_to_processed}/cluster_youtube_results.parquet"):
        data = pd.read_parquet(f"{path_to_processed}/cluster_youtube_results.parquet")

    else:
        data = pd.read_parquet(f"{path_to_processed}/cleaned_youtube_results.parquet")
    
    embeddings_archive = np.load(f"{path_to_processed}/youtube_comment_embeddings.npz")
    embeddings = embeddings_archive['embeddings']

    num_cpus = os.cpu_count() or 1  # fallback to 1 if None
    half_cpus = max(1, num_cpus // 2)

    for cluster_size in size:
    
            for radius in eps:

                cluster_col = f"cluster_{cluster_size}_{radius}"
                clusterer = DBSCAN(eps = radius, min_samples = cluster_size, metric = 'cosine', n_jobs = half_cpus)
                data[cluster_col] = clusterer.fit_predict(embeddings)

                clusters = np.array(data[cluster_col])
                unique_clusters = set(data[cluster_col])

                for cluster in unique_clusters:

                    cluster_texts = data[data[cluster_col] == cluster]['cleaned_comment_text'].tolist()

                    vectorizer = TfidfVectorizer(max_features = 10)
                    X = vectorizer.fit_transform(cluster_texts)

                    feature_names = vectorizer.get_feature_names_out()
                    avg_tfidf = X.mean(axis = 0).A1

                    keywords = pd.DataFrame({f"word": feature_names, f"tfidf": avg_tfidf})

            print(f"eps: {radius}, min_samples = {cluster_size}")
            print(f"Printing most common words for cluster #{cluster}:\n", keywords.head(10))

    os.makedirs(path_to_processed, exist_ok = True)

    columns_to_drop = ['channel_id', 'channel_name', 'video_id', 'country',
                       'lat', 'lon', 'comment_author', 'enough_char']
    
    safe_to_drop = [col for col in columns_to_drop if col in data.columns]

    data.drop(columns = safe_to_drop, inplace = True)

    data.to_parquet(f"{path_to_processed}/cluster_youtube_results.parquet")

def format_youtube(function, eps, size):

    if function == 'preprocess':
        preprocess()

    elif function == 'embeddings':
        embeddings()

    elif function == 'cluster':
        cluster(eps, size)

    elif function == 'all':
        preprocess()
        embeddings()
        cluster(eps, size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'prepare youtube data for analysis')
    parser.add_argument('--function', required = True, help = 'Options: "preprocess" to prepare data, "embeddings" to creating embeddings, "cluster" to generate clusters (optional argument --minimum for this), "all" to do all steps.')
    parser.add_argument('--size', nargs = '+', default = [10], help = 'Optional. List of min_cluster_size to examine. Default is [5, 10, 15].')    
    parser.add_argument('--eps', nargs = '+', default = [0.7], help = 'Optional. List of eps to examine. Default is [0.4, 0.45, 0.].')

    arg = parser.parse_args()

    format_youtube(function = arg.function, eps = arg.eps, size = arg.size)
