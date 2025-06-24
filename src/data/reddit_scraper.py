from dotenv import load_dotenv
from shared_functions import safe_loader, safe_saver
import time
import argparse
import os
import praw
import prawcore
import pandas as pd

def reddit_scrape(subreddit = None, search_term = None,
                  limit = 2000, comments = 50):

    """
    This loads in private information from a .env file that contains
    passwords, usernames, and so on.
    """
    load_dotenv()

    ID = os.getenv('REDDIT_CLIENT_ID')
    password = os.getenv('REDDIT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')

    """
    creates the reddit object containing the id, password, and the
    user_agent, so that searches can be performed.
    """
    reddit = praw.Reddit(
        client_id = ID,
        client_secret = password,
        user_agent = user_agent
    )

    """
    allow 18+ content
    """

    reddit.config._allow_nsfw = True

    """

    """

    path_to_subreddits = 'data/raw/reddit_subreddits.pkl'

    if os.path.exists(path_to_subreddits):
        all_subreddits = safe_loader(path_to_subreddits)
        if not isinstance(all_subreddits, set):
            all_subreddits = set(subreddit)

    else:
        all_subreddits = set()

    if not subreddit:

        if all_subreddits:
            subreddit = all_subreddits
        
        else:
            print(f"❗❗❗ Error! List of subreddits not found at {path_to_subreddits}."
                  f"If the file exists elsewhere, move it to the correct folder or change the "
                  f"path in the source file. Otherwise, you must manually specify a subreddit "
                  f"to search.")

            return None

    else:
        length_of_search = len(all_subreddits)

        all_subreddits = all_subreddits.union(subreddit)

        if length_of_search < len(all_subreddits):
            safe_saver(all_subreddits, path_to_subreddits)

    path_to_search_terms = 'data/raw/reddit_search_terms.pkl'

    if os.path.exists(path_to_search_terms):
        all_search_terms = safe_loader(path_to_search_terms)
        if not isinstance(all_search_terms, set):
            all_search_terms = set(subreddit)

    else:
        all_search_terms = set()

    if not search_term:

        if all_search_terms:

            search_term = all_search_terms

        else:
            print(f"❗❗❗ Error! List of search terms not found at {path_to_search_terms}."
                  f"If the file exists elsewhere, move it to the correct folder or change the "
                  f"path in the source file. Otherwise, you must manually specify a search term "
                  f"to search.")

            return None

    else:
        length_of_search = len(search_term)

        for term in search_term:
            if term not in all_search_terms:
                all_search_terms.add(term)

        if length_of_search < len(all_search_terms):
            safe_saver(all_search_terms, path_to_search_terms)

    """
    searches for a cache of previously searched posts. If it does not exist, it
    creates an empty set for cashing of post ids.
    """

    path_to_cache = 'data/raw/reddit_cache.pkl'

    if os.path.exists(path_to_cache):
        cache = safe_loader(path_to_cache)

    else:
        cache = set()

    """
    sorts through each post, under the defined subreddit, with the defined
    search term
    """

    path_to_results = 'data/raw/reddit_results.parquet'

    if os.path.exists(path_to_results):
        all_data = pd.read_parquet(path_to_results)

    else:
        all_data = pd.DataFrame()

    for sub in subreddit:

        for term in search_term:
            data = []
            max_retries = 4

            for attempt in range(max_retries):
                try:
                    for post in reddit.subreddit(sub).search(term, sort = 'relevance',
                                                             limit = limit):

                        if post.id in cache:
                            continue

                        cache.add(post.id)
                        
                        time.sleep(1)

                        post.comments.replace_more(limit = 0)

                        for top_comment in post.comments[:comments]:
                            replies_data = []
                            comments = 1
                            for reply in top_comment.replies[:comments]:
                                replies_data.append({
                                    'reply_id': reply.id,
                                    'reply_author': str(reply.author),
                                    'reply_body': reply.body,
                                    'reply_score': reply.score,
                                    'reply_created_utc': reply.created_utc
                                })

                                comments += 1

                            data.append({
                                'post_id' : post.id,
                                'post_title' : post.title,
                                'post_url' : post.url,
                                'post_score' : post.score,
                                'post_created_utc' : post.created_utc,
                                'post_author' : str(post.author),
                                'post_body' : post.selftext,
                                'comment_id' : top_comment.id,
                                'comment_author' : str(top_comment.author),
                                'comment_body' : top_comment.body,
                                'comment_score' : top_comment.score,
                                'comment_created_utc' : top_comment.created_utc,
                                'replies' : replies_data,
                                'comments' : comments,
                                'subreddit' : str(post.subreddit),
                                'source' : 'Reddit'
                            })

                    break

                except prawcore.exceptions.ServerError:
                    wait_time = 2 ** attempt
                    print(f"Server error. Retry {attempt + 1}/{max_retries} after {wait_time} seconds.")
                    time.sleep(wait_time)

                except prawcore.exceptions.TooManyRequests:
                    wait_time = 2 ** attempt
                    print(f"Server Error. Too many requests. Retrying after {wait_time} seconds.")

                except Exception as e:
                    wait_time = 2 ** attempt
                    print(f"Unexpected error: {e}. Retrying after {wait_time} seconds.")

            else:
                print(f"❗❗❗ Reddit API failed after {max_retries} retries. Giving up on "
                      f"search term {term} in subreddit {sub}.")

            if data:

                safe_saver(cache, path_to_cache)

                results = pd.DataFrame(data)
                all_data = pd.concat([all_data, results], ignore_index = True)

                all_data.to_parquet(path_to_results, engine = 'pyarrow', index = False)

                unique_posts = len(set(results['post_id']))
                all_posts = results['comments'].sum()

                print(f"Found {unique_posts} unique posts in the subreddit: "
                      f"{sub}, using search term: {term}. Recorded "
                      f"a total of {all_posts} posts/comments from this search. "
                      f"Results file and cache successfully updated.")            

    if not all_data.empty:

        unique_posts = len(set(all_data['post_id']))
        all_posts = all_data['comments'].sum()

        print(f"The search on reddit has found a total of {unique_posts} "
              f"unique posts and a total of {all_posts} comments/posts.")

    else:

        print(f"Found 0 posts using the search term: {search_term}, "
              f"across {len(subreddit)} subreddits.")

if __name__ == '__main__':
    """
    Simply creates the commands needed to run in powershell. Takes four
    optional arguments:

    --subreddit: the subreddit you want to search, 'all' for no specific subreddit.
        Exclude this argument if you want to search through a set of subreddits that
        should be saved in 'data/raw/reddit_subreddits.pkl'
        default: None.
    --search_term: the term are you searching for in the title/body
        of the text.
        Exclude this argument if you want to search through a set of search terms that
        should be saved in 'data/raw/reddit_subreddits.pkl'
        defeault: None
    --limit: limit the number of posts you want to scrape per topic. Default 10,000.
    --comments: limits the number of comments on a post that will be scraped. Default 100.
    """
    parser = argparse.ArgumentParser(description = 'search terms for reddit.')
    parser.add_argument('--subreddit', nargs='+', default = None,
                        help = 'subreddit to search. Default is None. If left as None, this function will load a list of subreddits to search.')
    parser.add_argument('--search_term', nargs='+', default = None,
                        help = 'what search term do you want to use. Default is None. If left as None, this function will load a list of terms to search.')
    parser.add_argument('--limit', default = 10000, type = int,
                        help = 'maximum number of posts to collect')
    parser.add_argument('--comments', default = 100, type = int,
                        help = 'maximum number of comments on a post to collect')

    args = parser.parse_args()

    reddit_scrape(subreddit = args.subreddit, search_term = args.search_term,
                  limit = args.limit, comments = args.comments)
