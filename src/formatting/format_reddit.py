from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import shared_functions
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import argparse
import numpy as np
import os

path_to_processed = 'data/processed'

def preprocess():

    from data_formatting_pipeline import pipeline
    
    data = pd.read_parquet(f"data/raw/reddit_results.parquet")
    print('Beginning preprocessing...')
    before = len(data)
    custom_stops = shared_functions.safe_loader('data/raw/custom_stopwords.pkl')
    whitelist = shared_functions.safe_loader('data/raw/whitelisted_characters.pkl')
    subreddits_to_region = shared_functions.safe_loader('data/raw/subreddit_to_region.pkl')

    original_posts = data.drop_duplicates(subset = 'post_body')
    original_posts.loc[:, 'comment_body'] = original_posts['post_body']
    original_posts.loc[:, 'comment_score'] = original_posts['post_score']
    original_posts.loc[:, 'comment_created_utc'] = original_posts['post_created_utc']
    data = pd.concat([data, original_posts], ignore_index = True)

    data = pipeline(data, 'comment_body', custom_stops = custom_stops, whitelist = whitelist,
                    geolocation = False)
    
    data['region'] = data['subreddit'].map(subreddits_to_region).fillna('Unknown')

    columns_to_drop = ['post_id', 'post_url', 'post_score', 'post_created_utc',
                       'post_author', 'post_body', 'comment_id', 'comment_author']
    
    data.drop(columns = columns_to_drop, inplace = True)
    data.to_parquet(f"{path_to_processed}/cleaned_reddit_results.parquet")
    after = len(data)

def embeddings():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_reddit_results.parquet")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device = device)
    preembedded = shared_functions.safe_loader('data/processed/custom_phrases_for_embeddeding.pkl')
    
    print('Generating topic embeddings...')
    
    topics = {}

    for topic, themes in preembedded.items():
        embeddings = model.encode(themes)
        average = np.mean(embeddings, axis = 0)
        average = normalize(average.reshape(1, -1))[0]
        topics[topic] = average

    shared_functions.safe_saver(topics, 'data/processed/custom_embeddings.pkl')

    embeddings = model.encode(data['cleaned_comment_body'].to_list())
    embeddings = normalize(embeddings)
    np.savez_compressed(f"{path_to_processed}/reddit_comment_embeddings.npz", embeddings = embeddings)

def cosine_similarity():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_reddit_results.parquet")

    topic_embeddings = shared_functions.safe_loader('data/processed/custom_embeddings.pkl')
    comment_embeddings_archive = np.load(f"{path_to_processed}/reddit_comment_embeddings.npz")
    comment_embeddings = comment_embeddings_archive['embeddings']

    low_similarity_column_names = []
    threshold = 0.5

    for topic in topic_embeddings:

        similarity = f"{topic}_similarity"
        low_similarity = f"low_{topic}_similarity"
        low_similarity_column_names.append(low_similarity)
        values = topic_embeddings.get(topic)

        data[similarity] = [np.dot(values, comment_embeddings[row]) for row in range(len(data))]
        data[low_similarity] = data[similarity] < threshold

        fig = plt.figure(figsize = (10, 10))

        sns.histplot(data[similarity], bins = 50, kde = True)
        plt.axvline(data[similarity].median(), color = 'purple', linestyle = '--', label = 'Median')
        plt.title(f"Distribution of Cosine Similarities for {topic}")
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"images/reddit_cosine_similarities_{topic}.png")

    previous_length = len(data)
    data = data[~data[low_similarity_column_names].all(axis = 1)]
    new_length = len(data)
    data.to_parquet(f"{path_to_processed}/topic_reduced_reddit_data.parquet")

    print(f"{100 - round(new_length / previous_length * 100, 2)}% of comments removed as irrelevant. "
          f"There are now {new_length} data points in the set.")
    
    print('Regenerating embeddings for reduced comment set...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device = device)

    embeddings = model.encode(data['cleaned_comment_body'].tolist())
    embeddings = normalize(embeddings)
    np.savez_compressed(f"{path_to_processed}/topic_reduced_reddit_comment_embeddings.npz", embeddings = embeddings)

    print('Cosine similarity reduction complete. All files saved successfully.')

def format_reddit(function):

    match function:
        case 'preprocess':
            preprocess()
        case 'embeddings':
            embeddings()
        case 'similarity':
            cosine_similarity()
        case 'all':
            preprocess()
            embeddings()
            cosine_similarity()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'prepare reddit data for analysis')
    parser.add_argument('--function', default = 'all', help = 'Default: "all", Options: "preprocess" to prepare data, "embeddings" to creating embeddings, "similarity" to do cosine similarity for comments, "all" to do all steps.')

    arg = parser.parse_args()

    format_reddit(function = arg.function)
