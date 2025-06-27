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
    """
    Loads in the youtube data and the relevant iso_codes, stopwords, and whitelists
    for processed. Runs the custom pipeline() function on both the comments and
    video title.
    """

    data = pd.read_parquet(f"data/raw/youtube_results.parquet")

    print('Beginning preprocessing...')
    iso_codes = shared_functions.safe_loader('data/raw/iso_codes.pkl')
    custom_stops = shared_functions.safe_loader('data/raw/custom_stopwords.pkl')
    whitelist = shared_functions.safe_loader('data/raw/whitelisted_characters.pkl')

    data = pipeline(data, 'comment_text', iso_codes = iso_codes, custom_stops = custom_stops, whitelist = whitelist)

    """
    Because there are way more rows than unique 'video_title' rows, I
    decided to speed up the processing by only processing the unique titles
    and then mapping them to a new column in data to speed the process up.
    """

    titles = data.drop_duplicates(subset = ['video_title']).copy()
    titles = pipeline(titles, 'video_title', iso_codes = iso_codes, custom_stops = custom_stops, whitelist = whitelist)
    title_map = dict(zip(titles['video_title'], titles['cleaned_video_title']))
    data['cleaned_video_title'] = data['video_title'].map(title_map)

    data.to_parquet(f"{path_to_processed}/cleaned_youtube_results.parquet")

def preprocess_reddit():
    """
    Loads in the reddit data, custom stops, whitelists, and a file that maps each
    subreddit where the geographical region is obvious from the name to that region,
    e.g., r/France --> 'Western Europe'.
    """
    
    data = pd.read_parquet(f"data/raw/reddit_results.parquet")

    print('Beginning preprocessing...')

    custom_stops = shared_functions.safe_loader('data/raw/custom_stopwords.pkl')
    whitelist = shared_functions.safe_loader('data/raw/whitelisted_characters.pkl')
    subreddits_to_region = shared_functions.safe_loader('data/raw/subreddit_to_region.pkl')

    """
    The original formatting has a separate column for original post and replies, here,
    I stack the original posts onto the comment_body column along with relevant metrics,
    like data and likes, in order to have one column with all the content to be cleaned.
    """

    original_posts = data.drop_duplicates(subset = 'post_body')
    original_posts.loc[:, 'comment_body'] = original_posts['post_body']
    original_posts.loc[:, 'comment_score'] = original_posts['post_score']
    original_posts.loc[:, 'comment_created_utc'] = original_posts['post_created_utc']
    data = pd.concat([data, original_posts], ignore_index = True)

    """
    Runs the pipeline without geolocation since this uses lat/lon that are not
    present in the reddit data. Then it maps the subreddits to regions that can be
    and the rest to 'Unknown'. Drops content that is not of interest.
    """

    data = pipeline(data, 'comment_body', custom_stops = custom_stops, whitelist = whitelist,
                    geolocation = False)
    
    data['region'] = data['subreddit'].map(subreddits_to_region).fillna('Unknown')

    columns_to_drop = ['post_id', 'post_url', 'post_score', 'post_created_utc',
                       'post_author', 'post_body', 'comment_id', 'comment_author']
    
    data.drop(columns = columns_to_drop, inplace = True)
    data.to_parquet(f"{path_to_processed}/cleaned_reddit_results.parquet")

def preprocess_threads():
    """
    Scraping resulted in a few NA values, fortunately not many, so
    those are filled and columns with NA dates are dropped.
    """

    data = pd.read_parquet(f"data/raw/threads_results.parquet")
    data['content'] = data['content'].fillna('None')
    data['likes'] = data['likes'].fillna(0)
    data.dropna(subset = ['published_on'], inplace = True)

    print('Beginning preprocessing...')

    """
    Loads in the relevant custom_stops and whitelist and then runs the pipeline
    without the geolocation function because it is specialized for lat/lon that
    is only available from youtube. Drops a few irrelevant columns and saves.
    """

    custom_stops = shared_functions.safe_loader('data/raw/custom_stopwords.pkl')
    whitelist = shared_functions.safe_loader('data/raw/whitelisted_characters.pkl')

    data = pipeline(data, 'content', custom_stops = custom_stops, whitelist = whitelist,
                    geolocation = False)

    columns_to_drop = ['thread_id', 'parent_id', 'user_id']

    data.drop(columns = columns_to_drop, inplace = True)
    data.to_parquet(f"{path_to_processed}/cleaned_threads_results.parquet")  

def create_masterdata():
    """
    This function loads in all the datasets to create one parquet file that has
    all the masterdata in it.
    """

    youtube = pd.read_parquet('data/processed/cleaned_youtube_results.parquet')

    """
    Countries are converted to regions here e.g., Brazil --> South America,
    converts the date-time to a consistent format and marks the columns to keep
    and drops the rest. Renames columns so that they will all be consistent across
    datasets.
    """
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

    """
    Adds an empty hashtags column, formats the data-time, and keeps only columns of interest.
    Renames columns for consistency.
    """

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
    """
    Regional data isn't available for threads, so the whole column is 'Unknown', an
    empty hashtags column is created, there is no title so an empty 'title' column
    is created and the data-time is formatted. Renames columns for consistency.
    """
    threads['region'] = 'Unknown'
    threads['hashtags'] = [[] for i in range(len(threads))]
    threads['title'] = 'None'
    threads['published_on'] = pd.to_datetime(threads['published_on'])
    
    renamed_columns = {'content' : 'text', 'cleaned_content' : 'cleaned_text', 'published_on' : 'date',
                       'content_language' : 'language'}
    threads.rename(renamed_columns, axis = 1, inplace = True)

    data = pd.concat([youtube, reddit, threads], ignore_index = True)
    data.reset_index(drop = True, inplace = True)
    """
    There were a few likes errors in the data, so I coerce them to numeric,
    clip the lower bound to 0, fill any NAs as 0. Converts the date-time to
    just the year and month of post. This makes time analysis easier because
    more general trends are desired, and further anonymizes the data, complying
    with GDPR and DSGVO regulations.
    """
    data['likes'] = pd.to_numeric(data['likes'], errors = 'coerce')
    data['likes'] = data['likes'].clip(lower = 0)
    data['likes'] = data['likes'].fillna(0).astype(int)
    data['month_year'] = data['date'].dt.to_period('M').astype(str)

    """
    Just organizes the dataframe columns the way I wanted them and then
    saves the data.
    """
    data = data[['title', 'text', 'likes', 'source', 'region', 'date', 'language', 'hashtags', 'cleaned_text']]

    data.to_parquet('data/processed/cleaned_masterdata.parquet')

def embeddings():
    """
    This is for making the embeddings that are used in cosine_similarity. I use
    'paraphrase-multilingual-MiniLM-L12-v2' because the dataset contains over 50
    languages and is extremely large so translation into English is not possible. 
    """

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata.parquet")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device = device)
    preembedded = shared_functions.safe_loader('data/processed/custom_phrases_for_embeddeding.pkl')
    
    print('Generating topic embeddings...')
    
    """
    I made custom topics/themes that I described in sentences and loaded them above.
    Those themes are then embedded and the average value is an average value vector
    is create for each theme. For the cosine similarity function later, comment vectors
    will be compared against these themes for similarity.
    """
    
    topics = {}

    for topic, themes in preembedded.items():
        embeddings = model.encode(themes)
        average = np.mean(embeddings, axis = 0)
        average = normalize(average.reshape(1, -1))[0]
        topics[topic] = average

    shared_functions.safe_saver(topics, 'data/processed/custom_embeddings.pkl')

    print('Generating text embeddings...')

    """
    Embeddings are then generated for all the comments and saved using np.savez_compressed.
    I used this because it does save some space compared to .pkl and the data is quite
    plentiful.
    """

    embeddings = model.encode(data['cleaned_text'].to_list())
    embeddings = normalize(embeddings)
    np.savez_compressed(f"{path_to_processed}/masterdata_embeddings.npz", embeddings = embeddings)

def cosine_similarity():
    """
    Loads in the data and the previously made embeddings.
    """

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata.parquet")

    topic_embeddings = shared_functions.safe_loader('data/processed/custom_embeddings.pkl')
    comment_embeddings_archive = np.load(f"{path_to_processed}/masterdata_embeddings.npz")
    comment_embeddings = comment_embeddings_archive['embeddings']

    """
    Here I create variables to keep track of the names of every low_similarity_column
    that gets generated and each similarity column that gets generated. I set the
    general threshold to 0.5, such that any comment with less than 0.5 similarity
    score gets marked for removal.
    """

    low_similarity_columns = []
    similarity_columns = []
    threshold = 0.5

    for topic in topic_embeddings:

        similarity = f"{topic}_similarity"
        similarity_columns.append(similarity)
        low_similarity = f"low_{topic}_similarity"
        low_similarity_columns.append(low_similarity)
        values = topic_embeddings.get(topic)
        """
        Because I normalized the vectors for the embeddings, the cosine similarity
        is just the dot product, so I used it here.
        """
        data[similarity] = [np.dot(values, comment_embeddings[row]) for row in range(len(data))]
        data[low_similarity] = data[similarity] < threshold

        """
        Plots the distributions of similarities scores for each topic for later inspection
        and possible adjustment of thresholding.
        """

        fig = plt.figure(figsize = (10, 10))

        sns.histplot(data[similarity], bins = 50, kde = True)
        plt.axvline(data[similarity].median(), color = 'purple', linestyle = '--', label = 'Median')
        plt.title(f"Distribution of Cosine Similarities for {topic}")
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"images/cosine_similarities_{topic}.png")

    """
    This finds every row that has low similarity for all the topics and marks it.
    Then I remove all those rows from the dataset and drop the columns used to find
    similarities.
    """
    low_similarity = (data[low_similarity_columns] >= threshold).all(axis = 1)
    before = len(data)
    reduced_data = data[~low_similarity].copy()
    reduced_data.drop(columns = low_similarity_columns + similarity_columns, inplace = True)

    """
    Here I have a few keywords I look for in the text and I use this just to be
    on the safe side, incase the similarity scores still filtered them out.
    I use a set to compare because the lookup time for each value is O(1).
    It exits the comparison for each item as soon as any keyword is found, speeding
    the search up a little.
    """
    keywords = shared_functions.safe_loader('data/processed/custom_keywords.pkl')

    rows_to_keep = []

    for index, items in data.iterrows():
        words = set(items['text'].split())
        if any(keyword in words for keyword in keywords):
            rows_to_keep.append(index)
            continue

    keyword_matched_data = data.iloc[rows_to_keep]

    """
    I keep the rows that meet the keywords, then I combine with the cosine reduced
    dataset. Because there is likely some overlap and I don't want duplicates, I
    drop any duplicate indices. The data is saved as is the low_similarity vector that
    was used to filter comments.
    """

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

if __name__ == '__main__':
    """
    Simple argument parser.
        --function: youtube, reddit, or threads to preprocess those datasets.
                    master to create the masterdata from the preprocessed datasets.
                    embeddings to create embeddings, similarity to use
                    cosine similarity to filter comments.
                    all to run everything.
    """
    
    parser = argparse.ArgumentParser(description = 'prepare data for analysis')
    parser.add_argument('--function', required = True, help = 'Options: "youtube", "reddit", or "threads" to prepare those datasets, "master" to create masterdata set from all datasets, "embeddings" to creating embeddings, "similarity" to do cosine similarity for comments, "all" to do all steps.')

    arg = parser.parse_args()

    format_data(function = arg.function)
