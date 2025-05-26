from playwright.sync_api import sync_playwright
from parsel import Selector
from shared_functions import safe_saver, safe_loader
from browser_scraping_functions import human_sleep, random_mouse_move, random_mouse_hover, open_random_link
import random
import argparse
import os

def search_threads(keyword = None, max_posts = 300):

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

    path_to_searches = 'data/raw/general_search_terms.pkl'

    if not keyword:

        if os.path.exists(path_to_searches):
            searches = safe_loader(path_to_searches)

        else:
            print(f"❗❗❗Error. List of search terms not found at: "
                  f"{path_to_searches}. Please check create a list "
                  f"of searches or enter an argument with --keyword.")

    else:
        searches = keyword

    all_posts_found = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless = False)  # Set headless=True for background run
        context = browser.new_context(storage_state = auth_state)
        page = context.new_page()

        base_url = 'https://www.threads.com'

        for keyword in searches:

            search_url = f"{base_url}/search?q={keyword}"
            page.goto(search_url)

            retries = 0
            last_height = 0
            gathered_posts = set()

            while len(gathered_posts) < max_posts and retries < 5:
                human_sleep(3, 10)
                page.mouse.wheel(0, random.randint(3000, 5000))

                chance = random.random()

                new_height = page.evaluate("() => document.body.scrollHeight")

                if new_height == last_height:
                    retries += 1
                else:
                    retries = 0
                    last_height = new_height

                sel = Selector(text = page.content())
                links = sel.css('a[href^="/@"]::attr(href)').getall()

                codes = [href for href in links
                         if '/post/' in href
                         and not href.endswith('/media')
                         and href not in cache]

                if 0.35 < chance < 0.6:
                    random_mouse_move(page)

                elif chance < 0.35 and links:
                    target = random.choice(links)
                    random_mouse_hover(page, target)

                    if chance < 0.10:
                        open_random_link(target, 'https://www.threads.com', context)

                gathered_posts.update(codes)
                post_codes.update(codes)
                cache.update(codes)
                all_posts_found.update(codes)
                
                safe_saver(post_codes, path_to_codes)
                safe_saver(cache, path_to_cache)
                
                print(f"Gathered {len(codes)} post ids. Continuing search...") 

        browser.close()
        
    print(f"Search complete, found a total of {len(all_posts_found)} "
          f"post ids. There are a total of {len(post_codes)} post ids "
          f"in threads_post_ids.pkl file to be scraped for further "
          f"investigation.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'search for threads.')
    parser.add_argument('--keyword', nargs='+', help = 'Optional. Keyword(s) to search for. If None, a list of searches is loaded and searched for')
    parser.add_argument('--max_posts', type = int, default = 300, help = 'Optional, default is 5. How many post ids to scrape.')

    arg = parser.parse_args()

    search_threads(keyword = arg.keyword, max_posts = arg.max_posts)
