from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from shared_functions import safe_saver, safe_loader, to_utc_iso
from datetime import datetime
import json
import pandas as pd
import argparse
import os
import sys

def youtube_scraper(query = None, max_videos = 100, max_comments = 100,
                    before = None, after = None):

    """
    This loads in previous search terms, if none exists, it creates
    and empty set of search terms. If the current search term is
    not in the set of search terms, it adds it and saves it.
    This acts as a record of all previous search terms.
    """

    path_to_search_terms = 'data/raw/youtube_search_terms.pkl'

    if os.path.exists(path_to_search_terms):
        all_search_terms = safe_loader(path_to_search_terms)
        
        if not isinstance(all_search_terms, set):
            all_search_terms = set(all_search_terms)

    else:
        all_search_terms = set()

    if query not in all_search_terms:

        all_search_terms.add(query)
        safe_saver(all_search_terms, path_to_search_terms)

    """
    Loads in the google api key and builds the youtube scraper.
    Sets the parameters for the search.
    """

    load_dotenv()
    api_key = os.getenv('GOOGLE_KEY')
    youtube = build('youtube', 'v3', developerKey = api_key)

    params = {
        'q': query,
        'part': 'snippet',
        'maxResults': 50,
        'type': 'video'
    }

    """
    Sets a date before and/or after if specified by in the argument.
    """

    if before:
        try:
            date_obj = datetime.strptime(before, "%d-%m-%y")
            params['publishedBefore'] = to_utc_iso(before)
        except ValueError:
            sys.exit("Invalid date format for --before. Please use DD-MM-YY.")

    if after:
        try:
            date_obj = datetime.strptime(after, "%d-%m-%y")
            params['publishedBefore'] = to_utc_iso(after)
        except ValueError:
            sys.exit("Invalid date format for --after. Please use DD-MM-YY.")

    """
    Loads in or creates a cache of searched youtube videos.
    """

    path_to_cache = 'data/raw/youtube_cache.pkl'

    if os.path.exists(path_to_cache):
        cache = safe_loader(path_to_cache)
    else:
        cache = set()

    results = []
    videos_fetched = 0
    next_page_token = None

    """
    This is the main function that will search through page results
    and extract relevant information on the OP and the replies. It
    will continue to fetch videos until either max_videos is reached
    or there are no videos left.
    """

    while videos_fetched < max_videos:
        if next_page_token:
            params['pageToken'] = next_page_token
        else:
            params.pop('pageToken', None)

        remaining = max_videos - videos_fetched
        """
        This either sets the maxResults to 50 (the maximum for the API) or
        the remaining videos left to find, whichever is lower.
        """
        params['maxResults'] = min(50, remaining)

        search_response = youtube.search().list(**params).execute()
        """
        This pulls out the relevant videos from the search unless the
        video is already in the cache, where it will skip over it.
        """
        for item in search_response.get('items', []):
            video_id = item['id']['videoId']

            if video_id in cache:
                continue
            else:
                cache.add(video_id)

            title = item['snippet']['title']

            video_response = youtube.videos().list(
                part = 'snippet,recordingDetails,statistics',
                id=  video_id
            ).execute()

            if not video_response['items']:
                continue

            """
            Gets relevant information, including video_info, id, title, location,
            view count, and likes.
            """

            video_info = video_response['items'][0]
            channel_id = video_info['snippet']['channelId']
            channel_title = video_info['snippet']['channelTitle']
            location = video_info.get('recordingDetails', {}).get('location', None)
            view_count = int(video_info['statistics'].get('viewCount', 0))
            like_count = int(video_info['statistics'].get('likeCount', 0))

            """
            This will either retrieve the lon/lat if it's available and if not,
            set it to 0, 0. The processing script will ignore 0, 0 inputs when
            using a geolocator.
            """

            if location:
                lat = location.get('latitude', 0)
                lon = location.get('longitude', 0)
            else:
                lat = 0
                lon = 0

            channel_response = youtube.channels().list(
                part = 'snippet,statistics',
                id = channel_id
            ).execute()

            """
            Gets summary information about the video, such as date, description, hashtags,
            and location, if available.
            """

            channel_info = channel_response['items'][0]
            country = channel_info['snippet'].get('country', 'N/A')
            subscribers = channel_info['statistics'].get('subscriberCount', 'N/A')
            publish_date = video_info['snippet']['publishedAt']
            description = video_response['items'][0]['snippet']['description']
            hashtags = [word for word in description.split() if word.startswith('#')]

            comments_next_page_token = None

            """
            This chunk goes through the comments to extract their content, like count,
            date, and so forth for recording.
            """

            try:
                comments_response = youtube.commentThreads().list(
                    part = 'snippet',
                    videoId = video_id,
                    maxResults = max_comments,
                    textFormat = 'plainText'
                ).execute()
                
            except HttpError as e:
                if e.resp.status == 403 and 'commentsDisabled' in str(e):
                    print(f"Comments are disabled for video {video_id}, skipping comments.")
                    video_comments = []
                    comment_counter = 0
                else:
                    raise
                
            else:
                video_comments = []
                comment_counter = 0
                
                for comment in comments_response['items']:
                    author = comment['snippet']['topLevelComment']['snippet']['authorDisplayName']
                    text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                    date = comment['snippet']['topLevelComment']['snippet']['publishedAt']
                    like_count = int(comment['snippet']['topLevelComment']['snippet'].get('likeCount', 0))
                    video_comments.append([author, date, text, like_count])
                    comment_counter += 1

            results.append({
                'channel_id': channel_id,
                'channel_name': channel_title,
                'video_id': video_id,
                'video_title': title,
                'country': country,
                'date': publish_date,
                'lat': lat,
                'lon': lon,
                'views': view_count,
                'likes': like_count,
                'subscribers': subscribers,
                'hashtags': hashtags, 
                'comments': video_comments,
                'number_of_comments': comment_counter,
                'source': 'YouTube'
            })

            videos_fetched += 1
            if videos_fetched >= max_videos:
                break

        next_page_token = search_response.get('nextPageToken')
        if not next_page_token:
            break

    """
    Creates a df that has the video information in it with the information
    for each comment on that video as a seperate row. This maintains the
    connection between the original video and the replies if it becomes
    relevant for the user to make that association. The loads and appends
    the results file if it exists, otherwise, it creates a new file and
    saves it.
    """

    all_rows = []

    for video in results:
        base_info = {
            'channel_id': video['channel_id'],
            'channel_name': video['channel_name'],
            'video_id': video['video_id'],
            'video_title': video['video_title'],
            'country': video['country'],
            'lat': video['lat'],
            'lon': video['lon'],
            'views': video['views'],
            'likes': video['likes'],
            'hashtags': video['hashtags'], 
            'subscribers': video['subscribers'],
            'source': video['source'],
        }

        for comment in video['comments']:
            row = base_info.copy()
            row['comment_author'] = comment[0]
            row['comment_time'] = comment[1]
            row['comment_text'] = comment[2]
            row['comment_likes'] = comment[3]
            all_rows.append(row)

    if all_rows:

        path = 'data/raw/youtube_results.parquet'

        if os.path.exists(path):
            all_results = pd.read_parquet(path)
            data = pd.DataFrame(all_rows)
            all_results = pd.concat([all_results, data], ignore_index = True)
            all_results.to_parquet(path, engine = 'pyarrow', index = False)

        else:
            os.makedirs('data/raw/', exist_ok = True)
            data = pd.DataFrame(all_rows)
            data.to_parquet(path, engine = 'pyarrow', index = False)
            
        unique_videos = len(data['video_id'].unique())
        comments = len(data) - unique_videos

        safe_saver(cache, path_to_cache)

        print(f"Found a total of {unique_videos} videos and {comments} comments "
              f"for search query: {query}. Parquet file saved successfully.")

    else:
        print(f"No results found for {query}")

if __name__ == '__main__':
    """
    Arguments:
        --query: what to search youtube for.
        --videos: Max videos to search.
        --before: fetch videos before a specified date.
        --after: fetch videos after a specified date.
    """
    parser = argparse.ArgumentParser(description = 'query for YouTube search.')
    parser.add_argument('--query', required = True,
                        help = 'Required. What to search YouTube for.')
    parser.add_argument('--videos', default = 100, type = int,
                        help = 'Optional, default is 100. How many videos you want to pull')
    parser.add_argument('--before', default = None,
                        help = 'Optional. Fetch videos published before this date (dd.mm.yyyy)')
    parser.add_argument('--after', default = None,
                        help = 'Optional. Fetch videos published after this date (dd.mm.yyyy)')

    arg = parser.parse_args()

    youtube_scraper(query = arg.query, max_videos = arg.videos,
                    before = arg.before, after = arg.after)
