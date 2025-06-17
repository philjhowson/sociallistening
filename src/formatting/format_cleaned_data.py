import pandas as pd
import shared_functions
import matplotlib.pyplot as plt
import geopandas as gpd
import shared_functions

def create_masterdata():

    youtube = pd.read_parquet('data/processed/cleaned_youtube_results.parquet')

    country_to_region = shared_functions.safe_loader('data/processed/youtube_countries_to_region.pkl')
    youtube['region'] = youtube['geolocation'].map(country_to_region).fillna('Unknown')
    youtube['comment_time'] = pd.to_datetime(youtube['comment_time'], errors = 'coerce').dt.date
    youtube['comment_time'] = pd.to_datetime(youtube['comment_time'])
    youtube.dropna(subset = ['comment_time', 'comment_text'], inplace = True)
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
    data['likes'] = data['likes'].fillna(0).astype(int)
    data.to_parquet('data/processed/cleaned_masterdata.parquet')   


def format_results():

    youtube = pd.read_parquet('data/processed/topics_sentiment_youtube_results.parquet')
    country_to_region = shared_functions.safe_loader('data/processed/youtube_countries_to_region.pkl')
    youtube['region'] = youtube['geolocation'].map(country_to_region).fillna('Unknown')
    columns_to_keep = ['video_title', 'hashtags', 'comment_text', 'comment_time',
                       'comment_likes', 'cleaned_comment_text', 'region',
                       'topic', 'sentiment', 'source']
    youtube = youtube[columns_to_keep]
    renamed_columns = {'video_title' : 'title', 'comment_text' : 'text', 'comment_time' : 'date',
                       'comment_likes' : 'likes', 'cleaned_comment_text' : 'cleaned_text'}
    youtube['hashtag'].fillna('[]', place = True)
    youtube.rename(renamed_columns, axis = 1, inplace = True)

    reddit = pd.read_parquet('data/processed/topics_sentiment_reddit_results.parquet')
 
    country_to_region = shared_functions.safe_loader('data/processed/reddit_countries_to_region.pkl')
    reddit['region'] = reddit['subreddit'].map(country_to_region).fillna('Unknown')
    reddit['hashtag'] = [[] for i in range(len(reddit))]
    columns_to_keep = ['post_title', 'hashtag', 'comment_body', 'comment_created_utc',
                       'comment_score', 'cleaned_comment_body', 'region', 'topic',
                       'sentiment', 'source']
    reddit = reddit[columns_to_keep]
    renamed_columns = {'post_title' : 'title', 'comment_body' : 'text', 'comment_created_utc' : 'date',
                       'comment_score' : 'likes', 'cleaned_comment_body' : 'cleaned_text'}
    reddit.rename(renamed_columns, axis = 1, inplace = True)

    data = pd.concat([youtube, reddit], ignore_index = True)
    data = data[data['topic'] != -1]
    data.dropna(subset = 'topic', inplace = True)
    data.to_parquet('data/processed/final_cleaned_data.parquet')

if __name__ == '__main__':
    create_masterdata()