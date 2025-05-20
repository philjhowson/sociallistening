from playwright.sync_api import sync_playwright
from parsel import Selector
from shared_functions import safe_saver, safe_loader
import numpy as np
import argparse
import time
import os

def search_threads(keyword, max_posts = 5):

    path_to_cache = 'data/raw/threads_cache.pkl'

    if os.path.exists(path_to_cache):
        cache = safe_loader(path_to_cache)

        if not isinstance(cache, set):
            cache = set(cache)

    else:
        cache = set()

    path_to_codes = 'data/raw/threads_post_ids.pkl'

    if os.path.exists(path_to_codes):
        post_codes = safe_loader(path_to_codes)

        if not isinstance(post_codes, set):
            post_codes = set(post_codes)

    else:
        post_codes = set()

    path_to_state = 'data/raw/threads_auth_state.json'
    
    if os.path.exists(path_to_state):
        auth_state = safe_loader(path_to_state)

    else:
        print(f"❗❗❗Error. Browser save state not found at: {path_to_state}. "
              f"Please check directory or run threads_get_login.py to "
              f"generate a new save state.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless = False)  # Set headless=True for background run
        context = browser.new_context(storage_state = auth_state)
        page = context.new_page()

        search_url = f"https://www.threads.net/search?q={keyword}"
        page.goto(search_url)
        time.sleep(np.random.randint(3, 10))  # Let JS initialize

        last_height = 0
        retries = 0

        while len(post_codes) < max_posts and retries < 5:
            page.mouse.wheel(0, 5000)  # Scroll down
            time.sleep(np.random.randint(10, 15))  # Wait for content to load

            sel = Selector(text = page.content())
            links = sel.css('a[href^="/@"]::attr(href)').getall()

            codes = [
                href for href in links
                if '/post/' in href            # only keep hrefs with '/post/'
                and not href.endswith('/media')  # skip media links
            ]

            before_count = len(post_codes)
            post_codes.update(codes)
            cache.update(codes)
            
            # If no new posts found, increment retry
            if len(post_codes) == before_count:
                retries += 1
            else:
                retries = 0

        browser.close()

    safe_saver(post_codes, path_to_codes)
    safe_saver(cache, path_to_cache)
        
    return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'search for threads.')
    parser.add_argument('--keyword', required = True, help = 'Required. Keyword to search for.')
    parser.add_argument('--max_posts', type = int, default = 5, help = 'Optional, default is 5. How many post ids to scrape.')

    arg = parser.parse_args()

    search_threads(keyword = arg.keyword, max_posts = arg.max_posts)
