from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import argparse
import os

def evaluate_clusters(source, eps):

    base_path = 'data/processed'
    missing_files = []

    if source == 'youtube':
        if os.path.exists(f"{base_path}/cluster_youtube_results.parquet"):
            data = pd.read_parquet(f"{base_path}/cluster_youtube_results.parquet")
        else:
            missing_files.append('cluster_youtube_results.parquet')

        if os.path.exists(f"{base_path}/youtube_comment_embeddings.npz"):
            embeddings_archive = np.load(f"{base_path}/youtube_comment_embeddings.npz")
            embeddings = embeddings_archive['embeddings']
        else:
            missing_files.append('youtube_comment_embeddings.npz')
        
        if missing_files:
            print(f"Could not find file(s): {', '.join(missing_files)}, at path: {base_path}.")

            return None
        
    if source == 'threads':
        if os.path.exists(f"{base_path}/cluster_threads_results.parquet"):
            data = pd.read_parquet(f"{base_path}/cluster_threads_results.parquet")
        else:
            missing_files.append('cluster_threads_results.parquet')

        if os.path.exists(f"{base_path}/threads_embeddings.npz"):
            embeddings_archive = np.load(f"{base_path}/threads_embeddings.npz")
            embeddings = embeddings_archive['embeddings']
        else:
            missing_files.append('threads_embeddings.npz')
        
        if missing_files:
            print(f"Could not find file(s): {', '.join(missing_files)}, at path: {base_path}.")

            return None
    if source == 'reddit':
        if os.path.exists(f"{base_path}/cluster_reddit_results.parquet"):
            data = pd.read_parquet(f"{base_path}/cluster_reddits_results.parquet")
        else:
            missing_files.append('cluster_reddit_results.parquet')

        if os.path.exists(f"{base_path}/reddit_embeddings.npz"):
            embeddings_archive = np.load(f"{base_path}/reddit_embeddings.npz")
            embeddings = embeddings_archive['embeddings']
        else:
            missing_files.append('reddit_embeddings.npz')
        
        if missing_files:
            print(f"Could not find file(s): {', '.join(missing_files)}, at path: {base_path}.")

            return None

    cluster_col = f"clusters_{eps}"

    clusters = np.array(data[cluster_col])
    unique_clusters = set(data[cluster_col])

    for cluster in unique_clusters:
        cluster_texts = data[data[cluster_col] == cluster]['cleaned_comment_text'].tolist()

        vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 10)
        X = vectorizer.fit_transform(cluster_texts)

        feature_names = vectorizer.get_feature_names_out()
        avg_tfidf = X.mean(axis = 0).A1

        keywords = pd.DataFrame({f"word": feature_names, f"tfidf": avg_tfidf})

        print(f"Printing most common words for cluster #{cluster}:\n", keywords.head(10))

    print('Generating t-SNE plot...')

    tsne = TSNE(n_components = 2, random_state = 42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize = (10, 10))

    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        label = 'Noise' if cluster_id == -1 else f'Cluster: {cluster_id}'
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label = label, alpha = 0.6)

    plt.legend()
    plt.title(f"DBSCAN Clusters (eps = {eps}) Visualized with t-SNE")
    plt.xlabel('t-SNE Dim 1')
    plt.ylabel('t-SNE Dim 2')
    plt.tight_layout()

    plt.savefig(f"images/DBSCAN_{source}_{eps}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'inspect clusters from data formatting')
    parser.add_argument('--source', required = True, help = 'From which source do you want to examine data')
    parser.add_argument('--eps', default = 0.5, help = 'Eps value you plan to examine. Searches the df for the relevant column to use.')

    arg = parser.parse_args()

    evaluate_clusters(source = arg.source, eps = arg.eps)