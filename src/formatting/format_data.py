from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from data_formatting_pipeline import pipeline
import shared_functions
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import argparse
import numpy as np
import os

path_to_processed = 'data/processed'

def preprocess_youtube():

    data = pd.read_parquet(f"data/raw/youtube_results.parquet")
    print('Beginning preprocessing...')
    iso_codes = shared_functions.safe_loader('data/raw/iso_codes.pkl')
    custom_stops = shared_functions.safe_loader('data/raw/custom_stopwords.pkl')
    whitelist = shared_functions.safe_loader('data/raw/whitelisted_characters.pkl')

    data = pipeline(data, 'comment_text', iso_codes = iso_codes, custom_stops = custom_stops, whitelist = whitelist)
    data = pipeline(data, 'video_title', iso_codes = iso_codes, custom_stops = custom_stops, whitelist = whitelist)

    data.to_parquet(f"{path_to_processed}/cleaned_youtube_results.parquet")

def preprocess_reddit():
    
    data = pd.read_parquet(f"data/raw/reddit_results.parquet")
    print('Beginning preprocessing...')

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

def preprocess_threads():

    pass

def create_masterdata():

    youtube = pd.read_parquet('data/processed/cleaned_youtube_results.parquet')

    country_to_region = shared_functions.safe_loader('data/processed/youtube_countries_to_region.pkl')
    youtube['region'] = youtube['geolocation'].map(country_to_region).fillna('Unknown')
    youtube['comment_time'] = pd.to_datetime(youtube['comment_time'], errors = 'coerce').dt.date
    youtube['comment_time'] = pd.to_datetime(youtube['comment_time'])
    columns_to_keep = ['video_title', 'hashtags', 'comment_text', 'comment_time',
                       'comment_likes', 'cleaned_comment_text', 'region',
                       'source']
    youtube = youtube[columns_to_keep]
    renamed_columns = {'video_title' : 'title', 'comment_text' : 'text', 'comment_time' : 'date',
                       'comment_likes' : 'likes', 'cleaned_comment_text' : 'cleaned_text',
                       'hashtags' : 'hashtag'}
    youtube.rename(renamed_columns, axis = 1, inplace = True)
    youtube['hashtag'] = youtube['hashtag'].fillna('[]')

    reddit = pd.read_parquet('data/processed/cleaned_reddit_results.parquet')

    country_to_region = shared_functions.safe_loader('data/processed/reddit_countries_to_region.pkl')
    reddit['region'] = reddit['subreddit'].map(country_to_region).fillna('Unknown')
    reddit['hashtag'] = [[] for i in range(len(reddit))]
    reddit['comment_created_utc'] = pd.to_datetime(reddit['comment_created_utc'], unit = 's').dt.date
    reddit['comment_created_utc'] = pd.to_datetime(reddit['comment_created_utc'])
    columns_to_keep = ['post_title', 'hashtag', 'comment_body', 'comment_created_utc',
                       'comment_score', 'cleaned_comment_body', 'region', 'source']
    reddit = reddit[columns_to_keep]
    renamed_columns = {'post_title' : 'title', 'comment_body' : 'text', 'comment_created_utc' : 'date',
                       'comment_score' : 'likes', 'cleaned_comment_body' : 'cleaned_text'}
    reddit.rename(renamed_columns, axis = 1, inplace = True)
    
    data = pd.concat([youtube, reddit], ignore_index = True)
    data['likes'] = pd.to_numeric(data['likes'], errors = 'coerce')
    data['likes'] = data['likes'].clip(lower = 0)
    data['likes'] = data['likes'].fillna(0).astype(int)
    data['log_likes'] = np.log1p(data['likes'])
    data['month_year'] = data['date'].dt.to_period('M').astype(str)
    data.to_parquet('data/processed/cleaned_masterdata.parquet')

def embeddings():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata.parquet")

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

    embeddings = model.encode(data['cleaned_text'].to_list())
    embeddings = normalize(embeddings)
    np.savez_compressed(f"{path_to_processed}/masterdata_embeddings.npz", embeddings = embeddings)

def cosine_similarity():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata.parquet")

    topic_embeddings = shared_functions.safe_loader('data/processed/custom_embeddings.pkl')
    comment_embeddings_archive = np.load(f"{path_to_processed}/masterdata_embeddings.npz")
    comment_embeddings = comment_embeddings_archive['embeddings']

    low_similarity_columns = []
    similarity_columns = []
    threshold = 0.5

    for topic in topic_embeddings:

        similarity = f"{topic}_similarity"
        similarity_columns.append(similarity)
        low_similarity = f"low_{topic}_similarity"
        low_similarity_columns.append(low_similarity)
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

        plt.savefig(f"images/cosine_similarities_{topic}.png")

    low_similarity = (data[low_similarity_columns] >= threshold).all(axis = 1)
    before = len(data)
    data = data[~low_similarity]
    data.drop(columns = low_similarity_columns + similarity_columns, inplace = True)
    data.to_parquet(f"{path_to_processed}/cosine_reduced_masterdata.parquet")
    data.to_csv(f"{path_to_processed}/cosine_reduced_masterdata.csv", index = False)
    after = len(data)
    shared_functions.safe_saver(low_similarity, f"{path_to_processed}/low_similarity_{threshold}.pkl")

    print(f"Cosine similarity calculated for {threshold}. There are now {after} data points "
          f"or {round(after/before * 100, 3)}% of the original dataframe. All files saved successfully.")
    

def format_data(function):

    match function:
        case 'youtube':
            preprocess_youtube()
        case 'reddit':
            preprocess_reddit()
        case 'threads':
            preprocess_threads()
        case 'master':
            create_masterdata()
        case 'embeddings':
            embeddings()
        case 'similarity':
            cosine_similarity()
        case 'all':
            preprocess_youtube()
            preprocess_reddit()
            preprocess_threads()
            create_masterdata()
            embeddings()
            cosine_similarity()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'prepare data for analysis')
    parser.add_argument('--function', default = 'all', help = 'Default: "all", Options: "preprocess" to prepare data, "embeddings" to creating embeddings, "similarity" to do cosine similarity for comments, "all" to do all steps.')

    arg = parser.parse_args()

    format_data(function = arg.function)
