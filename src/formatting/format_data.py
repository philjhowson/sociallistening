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

    data = pd.read_parquet(f"data/raw/threads_results.parquet")
    data['content'] = data['content'].fillna('None')
    data['likes'] = data['likes'].fillna(0)
    data.dropna(subset = ['published_on'], inplace = True)

    print('Beginning preprocessing...')

    custom_stops = shared_functions.safe_loader('data/raw/custom_stopwords.pkl')
    whitelist = shared_functions.safe_loader('data/raw/whitelisted_characters.pkl')

    data = pipeline(data, 'content', custom_stops = custom_stops, whitelist = whitelist,
                    geolocation = False)

    columns_to_drop = ['thread_id', 'parent_id', 'user_id']

    data.drop(columns = columns_to_drop, inplace = True)
    data.to_parquet(f"{path_to_processed}/cleaned_threads_results.parquet")  

def create_masterdata():

    youtube = pd.read_parquet('data/processed/cleaned_youtube_results.parquet')

    country_to_region = shared_functions.safe_loader('data/processed/youtube_countries_to_region.pkl')
    youtube['region'] = youtube['geolocation'].map(country_to_region).fillna('Unknown')
    youtube['comment_time'] = pd.to_datetime(youtube['comment_time'], errors = 'coerce').dt.date
    youtube['comment_time'] = pd.to_datetime(youtube['comment_time'])
    columns_to_keep = ['video_title', 'hashtags', 'comment_text', 'comment_time',
                       'comment_likes', 'cleaned_comment_text', 'region',
                       'source', 'comment_text_language']
    youtube = youtube[columns_to_keep]
    renamed_columns = {'video_title' : 'title', 'comment_text' : 'text', 'comment_time' : 'date',
                       'comment_likes' : 'likes', 'cleaned_comment_text' : 'cleaned_text',
                       'comment_text_language' : 'language'}
    youtube.rename(renamed_columns, axis = 1, inplace = True)
    youtube['hashtags'] = youtube['hashtags'].fillna('[]')

    reddit = pd.read_parquet('data/processed/cleaned_reddit_results.parquet')

    country_to_region = shared_functions.safe_loader('data/processed/reddit_countries_to_region.pkl')
    reddit['region'] = reddit['subreddit'].map(country_to_region).fillna('Unknown')
    reddit['hashtags'] = [[] for i in range(len(reddit))]
    reddit['comment_created_utc'] = pd.to_datetime(reddit['comment_created_utc'])
    columns_to_keep = ['post_title', 'hashtags', 'comment_body', 'comment_created_utc',
                       'comment_score', 'cleaned_comment_body', 'region', 'source',
                       'comment_body_language']
    reddit = reddit[columns_to_keep]
    renamed_columns = {'post_title' : 'title', 'comment_body' : 'text', 'comment_created_utc' : 'date',
                       'comment_score' : 'likes', 'cleaned_comment_body' : 'cleaned_text',
                       'comment_body_language' : 'language'}
    reddit.rename(renamed_columns, axis = 1, inplace = True)

    threads = pd.read_parquet(f"{path_to_processed}/cleaned_threads_results.parquet")
    threads['region'] = 'Unknown'
    threads['hashtags'] = [[] for i in range(len(threads))]
    threads['title'] = 'None'
    threads['published_on'] = pd.to_datetime(threads['published_on'])
    
    renamed_columns = {'content' : 'text', 'cleaned_content' : 'cleaned_text', 'published_on' : 'date',
                       'content_language' : 'language'}
    threads.rename(renamed_columns, axis = 1, inplace = True)

    data = pd.concat([youtube, reddit, threads], ignore_index = True)
    data.reset_index(drop = True, inplace = True)
    data['likes'] = pd.to_numeric(data['likes'], errors = 'coerce')
    data['likes'] = data['likes'].clip(lower = 0)
    data['likes'] = data['likes'].fillna(0).astype(int)
    data['log_likes'] = np.log1p(data['likes'])
    data['month_year'] = data['date'].dt.to_period('M').astype(str)

    data = data[['title', 'text', 'likes', 'source', 'region', 'date', 'language', 'hashtags', 'cleaned_text']]

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

    print('Generating text embeddings...')

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
    reduced_data = data[~low_similarity].copy()
    reduced_data.drop(columns = low_similarity_columns + similarity_columns, inplace = True)

    keywords = shared_functions.safe_loader('data/processed/custom_keywords.pkl')

    rows_to_keep = []

    for index, items in data.iterrows():
        words = set(items['text'].split())
        if any(keyword in words for keyword in keywords):
            rows_to_keep.append(index)
            continue

    keyword_matched_data = data.iloc[rows_to_keep]

    final_reduced_data = pd.concat([reduced_data, keyword_matched_data])
    final_reduced_data = final_reduced_data[~final_reduced_data.index.duplicated(keep = 'first')]

    final_reduced_data.to_parquet(f"{path_to_processed}/cosine_reduced_masterdata.parquet")
    after = len(final_reduced_data)
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
