from mastodon_functions import collect_from_instance
from shared_functions import safe_saver
from dotenv import load_dotenv
import os
import argparse

def mastodon_scraper(instance, max_posts):

    load_dotenv()
    token = os.getenv('MASTODON_ACCESS_TOKEN')

    keywords = [
        'warranty', 'extended warranty', 'applecare', 'applecare+',
        'samsung care+', 'microsoft complete', 'geek squad',
        'amazon protect', 'sony protect'
    ]
    
    results = collect_from_instance(
        base_url = instance,
        keywords = keywords,
        access_token = token,
        max_posts_per_instance = max_posts
    )

    path = 'data/raw/'

    instance = instance.split('//')[-1]
    
    file = f"{instance}_mastodon_raw.json"

    safe_saver(results, path, file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'search terms for Mastodon.')
    parser.add_argument('--instance', default = 'https://mastodon.social',
                        help = 'the mastodon instance you want to search')
    parser.add_argument('--max_posts', default = 200, type = int,
                        help = 'the maximum number of posts you want to extract.')

    args = parser.parse_args()

    mastodon_scraper(instance = args.instance, max_posts = args.max_posts)

    """
        https://mastodon.social
        https://mastodon.online
        https://fosstodon.org
        https://mstdn.social
        https://techhub.social
        https://mastodon.hostux.social
        https://Pawoo.net #Japan
        https://mastodon-japan.net #Japan
        https://famichiki.jp #Japan
        https://stella.place #Korea
        https://planet.moe/explore #Korea
        https://k.lapy.link/ #Korea
        https://respublicae.eu #Europe
    """
