import pandas as pd
import datetime

def data_formatting():

    reddit_data = pd.read_parquet('data/raw/reddit_results.parquet')

    reddit_posts = len(reddit_data['post_id'].unique())
    reddit_comments = reddit_data['comments'].sum()

    reddit_data['post_created_datetime'] = pd.to_datetime(reddit_data['post_created_utc'], unit = 's')
    
    oldest_comment = reddit_data['post_created_datetime'].min()
    newest_comment = reddit_data['post_created_datetime'].max()

    youtube_data = pd.read_parquet('data/raw/youtube_results.parquet')

    youtube_posts = len(youtube_data['video_id'].unique())
    youtube_comments = youtube_data['number_of_comments'].sum()

    print(youtube_data.columns)

    print(f"Searches have found a total of {reddit_posts + youtube_posts} "
          f"posts and a total of {reddit_comments + youtube_comments} comments. "
          f"This is a total of {reddit_posts + youtube_posts + reddit_comments + youtube_comments} "
          f"data points so far.")

    print(f"The oldest reddit comment is {oldest_comment} and the newest "
          f"comment is {newest_comment}.")

if __name__ == '__main__':
    data_formatting()
