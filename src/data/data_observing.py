import pandas as pd
import datetime
from shared_functions import safe_saver, safe_loader

def data_observing():

    reddit_data = pd.read_parquet('data/raw/reddit_results.parquet')

    reddit_posts = len(reddit_data['post_id'].unique())
    reddit_comments = reddit_data['comments'].sum()

    reddit = len(reddit_data) + reddit_comments

    reddit_data['post_created_utc'] = pd.to_datetime(reddit_data['post_created_utc'], unit = 's').dt.date
    
    oldest_reddit = reddit_data['post_created_utc'].min()
    newest_reddit = reddit_data['post_created_utc'].max()

    youtube_data = pd.read_parquet('data/raw/youtube_results.parquet')

    youtube_data['comment_time'] = pd.to_datetime(youtube_data['comment_time']).dt.date

    youtube_posts = len(youtube_data['video_id'].unique())
    youtube_comments = len(youtube_data) - youtube_posts

    youtube = len(youtube_data)

    oldest_youtube = youtube_data['comment_time'].min()
    newest_youtube = youtube_data['comment_time'].max()
    
    threads_data = pd.read_parquet('data/raw/threads_results.parquet')

    threads_data.dropna(subset = ['content'], inplace = True)

    threads_posts = len(threads_data['parent_id'].unique()) - 1
    threads_comments = len(threads_data) - threads_posts
    
    threads = len(threads_data)

    threads_data['published_on'] = threads_data['published_on'].astype(int)
    threads_data['published_on'] = pd.to_datetime(threads_data['published_on'], unit = 's').dt.date

    oldest_threads = threads_data['published_on'].min()
    newest_threads = threads_data['published_on'].max()

    data = {'comments' : [youtube, threads, reddit],
            'source' : ['Google', 'Threads', 'Reddit']}
    
    safe_saver(data, 'data/processed/comment_counts.pkl')

    print(f"Searches have found {reddit_posts} Reddit posts, {youtube_posts} "
          f"YouTube posts, and {threads_posts} Threads posts, for a total of " 
          f"{reddit_posts + youtube_posts + threads_posts} posts. There were "
          f"{reddit_comments} Reddit comments, {youtube_comments} YouTube comments, "
          f"and {threads_comments} Threads comments for a total of "
          f"{reddit_comments + youtube_comments + threads_comments} comments. "
          f"This is a total of {reddit_posts + youtube_posts + threads_posts + reddit_comments + youtube_comments + threads_comments} "
          f"data points so far.")

    print(f"The oldest Reddit comment was posted at {oldest_reddit} "
          f"and the newest comment was posted at {newest_reddit}. The "
          f"oldest YouTube comment was posted at {oldest_youtube} and "
          f"the newest comment was posted at {newest_youtube}. The "
          f"oldest Threads comment was posted at {oldest_threads} and "
          f"the newest comment was posted at {newest_threads}.")
    
if __name__ == '__main__':
    data_observing()
