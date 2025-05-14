from dotenv import find_dotenv
import argparse
import os
import praw
import pandas as pd

def reddit_scrape(subreddit = 'AppleWatch', search_term = 'warranty'):

    """
    This loads in private information from a .env file that contains
    passwords, usenames, and so on.
    """
    load_dotenv()

    ID = os.getenv('REDDIT_CLIENT_ID')
    password = os.getenv('REDDIT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')

    """
    creates the reddit object cotaining the id, password, and the
    user_agent, so that searches can be performed.
    """
    reddit = praw.Reddit(
        client_id = ID,
        client_secret = password,
        user_agent = user_agent
    )

    """
    The search term is set by the arguments passed when the script is
    run, otherwise, it defaults to '/rAppleWatch' subreddit with a
    search for 'warranty'. LIMIT_POST sets the maximum number of posts
    that will be scraped and LIMIT_COMMENTS sets the maximum number of
    replies that will be scrapped.
    """

    SUBREDDIT = subreddit
    SEARCH_TERM = search_term
    LIMIT_POSTS = 1000
    LIMIT_COMMENTS = 50

    """
    Initialize an empty list to store all the extracted information
    for later conversion to a dataframe.
    """

    data = []

    """
    sorts through each post, under the defined subreddit, with the defined
    search term
    """
    for post in reddit.subreddit(SUBREDDIT).search(SEARCH_TERM,
                                                   limit = LIMIT_POSTS):
        """
        this specifies how many 'more comments' the API will attempt to
        load. 0 means that the script will attempt to retrieve all comments.
        1 means it will retreive the more comments 1 time.
        """
        post.comments.replace_more(limit = 0)

        """
        loops through the specified number of comments, which you
        can find the variable above. It then initializes an empty
        list of replies, then it goes over the comments on the each
        reply in the same way, thus creating a series of comments
        and replies. Critical information that is extracted:

        post.id = the id for the post
        post.title = the title of the post
        post.url = the webaddress of the post
        post.score = the upvotes for the post - the number of downvotes
        post.created_utc = UTC timestamp for post creation
        post.author = username of poster
        post.selftext = the body of the comment, if empty it is a link/image/video etc.
        top_comment.id = comment identifier
        top_comment.author = author of the comment
        top_comment.created_utc = the time the comment was posted
        replies_text = a list of replies to the comment.
        """
        for top_comment in post.comments[:LIMIT_COMMENTS]:
            replies_text = []
            for reply in top_comment.replies[:LIMIT_COMMENTS]:
                replies_text.append(reply.body)
            
            data.append({
                'post_id': post.id,
                'post_title': post.title,
                'post_url': post.url,
                'post_score': post.score,
                'post_created_utc': post.created_utc,
                'post_author': str(post.author),
                'post_body': post.selftext,
                'comment_id': top_comment.id,
                'comment_author': str(top_comment.author),
                'comment_body': top_comment.body,
                'comment_score': top_comment.score,
                'comment_created_utc': top_comment.created_utc,
                'replies': replies_text
            })

    data = pd.DataFrame(data)

    print(data.head(10))

    data.to_json(orient = 'records')
    

if __name__ == '__main__':
    """
    Simply creates the commands needed to run in powershell. Takes two
    optional arguments:

    --subreddit: the subreddit you want to search, 'all' for no subreddit
        default: 'AppleWatch'
    --search_term: the term are you searching for in the title/body
        of the text.
        defeault: 'warranty'
    """
    parser = argparse.ArgumentParser(description = 'search terms for reddit.')
    parser.add_argument('--subreddit', default = 'AppleWatch',
                        help = 'subreddit to search, use all for no subreddit search.')
    parser.add_argument('--search_term', default = 'warranty',
                        help = 'what search term do you want to use')

    args = parser.parse_args()

    reddit_scrape(subreddit = args.subreddit, search_term = args.search_term)
