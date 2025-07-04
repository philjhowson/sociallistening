import json
import time
import random
import jmespath
import os
import sys
import pandas as pd
from browser_scraping_functions import human_sleep, random_mouse_move, random_mouse_hover, open_random_link
from typing import Dict
from parsel import Selector
from nested_lookup import nested_lookup
from playwright.sync_api import sync_playwright
from shared_functions import safe_saver, safe_loader

def parse_thread(data: Dict) -> Dict:
    """Parse Threads JSON dataset for the most important fields"""
    result = jmespath.search(
        """{
        text: post.caption.text,
        published_on: post.taken_at,
        id: post.id,
        pk: post.pk,
        code: post.code,
        username: post.user.username,
        user_pic: post.user.profile_pic_url,
        user_verified: post.user.is_verified,
        user_pk: post.user.pk,
        user_id: post.user.id,
        has_audio: post.has_audio,
        reply_count: view_replies_cta_string,
        like_count: post.like_count,
        images: post.carousel_media[].image_versions2.candidates[1].url,
        image_count: post.carousel_media_count,
        videos: post.video_versions[].url
    }""",
        data,
    )
    
    result['videos'] = list(set(result['videos'] or []))

    if result['reply_count'] and type(result['reply_count']) != int:
        result['reply_count'] = int(result['reply_count'].split(' ')[0])
        
    result['url'] = f"https://www.threads.com/@{result['username']}/post/{result['code']}"
    
    return result

def scrape_thread(page, url: str) -> dict:
    """Scrape Threads post and replies from a given URL"""

    page.goto(url)
    page.wait_for_selector('[data-pressable-container=true]')

    """
    Randomly selects a number of scrolls to perform before moving
    on with scraping. Therefore, it should be noted as is, this will
    not necessarily scrape every reply to very long threads. The
    longer human sleep is to insure the page fully loads.
    """

    max_scrolls = random.randint(9, 15)
    last_height = 0

    human_sleep(5, 10)

    for scroll in range(max_scrolls):

        """
        Scrolls down through the page, getting all the links to
        poster profiles and media and generates a random number
        """
        
        page.mouse.wheel(0, 5000)
        human_sleep(2, 7)

        chance = random.random()
        sel = Selector(text = page.content())
        links = sel.css('a[href^="/@"]::attr(href)').getall()

        """
        If the generated number is between 0.35 and 0.6, it will
        randomly move the mouse. If the random number is
        below 0.35 it will move the mouse and hover over an url.
        Finally, if it's below 0.1, it will open the link in
        a new tab and inspect it. This is designed to simulate
        human mouse moving patterns and curiosity.
        """

        if 0.35 < chance < 0.6:
            random_mouse_move(page)

        elif chance < 0.35 and links:
            target = random.choice(links)
            random_mouse_hover(page, target)

            if chance < 0.10:
                open_random_link(target, 'https://www.threads.com', context)

        """
        Retrieves the new_height after page loading and compares it to the last
        height. If they are the same, it executes scraping. However, scraping
        will also be executed if the current scroll is the last scroll in the
        range.
        """

        new_height = page.evaluate("() => document.body.scrollHeight")

        if new_height == last_height or scroll == max_scrolls - 1:

            """
            Brief human sleep to simulate a persons scanning the resting
            position of the current webpage. Gathers page content.
            """
            
            human_sleep(2, 5)
            selector = Selector(page.content())
            hidden_datasets = selector.css('script[type="application/json"][data-sjs]::text').getall()

            for hidden_dataset in hidden_datasets:
                """
                If the relevant items for scraping threads are not present
                in hidden datasets, it continues with the rest of the script.
                If there are no thread_items in the extracted dictionary,
                the script will continue with the rest.
                """
                
                if '"ScheduledServerJS"' not in hidden_dataset:
                    continue
                if "thread_items" not in hidden_dataset:
                    continue

                data = json.loads(hidden_dataset)
                thread_items = nested_lookup('thread_items', data)

                if not thread_items:
                    continue

                threads = [parse_thread(t) for thread in thread_items for t in thread]

                return {
                    'thread': threads[0],
                    'replies': threads[1:]
                }

            break

        last_height = new_height

    raise ValueError('Could not find thread data in page after scrolling')

"""
Loads previously gathered post ids from threads, shuffles the codes.
It opens the cache file if there is one, if not, it creates one.
Set is used because it delivers and O(1) lookup time and ensures
no unnecessary duplicate items are cached.
"""

path_to_ids = 'data/raw/threads_post_ids.pkl'

codes = safe_loader(path_to_ids)
codes = list(codes)
random.shuffle(codes)

path_to_cache = 'data/raw/threads_cache.pkl'

if os.path.exists(path_to_cache):
    cache = safe_loader(path_to_cache)
else:
    cache = set()

if __name__ == "__main__":

    """
    opens up the an auth_state to avoid login problems
    and checks if there is an existing dataframe to save
    results to, else it creates a new results dataframe.
    """

    path_to_state = 'data/raw/threads_auth_state.json'
    
    if os.path.exists(path_to_state):
        auth_state = safe_loader(path_to_state)

    else:
       sys.exit(f"❗❗❗Error. Browser save state not found at: {path_to_state}. "
                f"Please check directory or run threads_get_login.py to "
                f"generate a new save state.")

    path_to_results = 'data/raw/threads_results.parquet'

    if os.path.exists(path_to_results):
        results = pd.read_parquet(path_to_results)

    else:
        results = pd.DataFrame(columns = ['thread_id', 'parent_id', 'likes', 'user_id',
                                          'content', 'published_on', 'source'])

    old_results_length = len(results)

    base_url = 'https://www.threads.com'

    with sync_playwright() as pw:
        """
        opens up a new browser with the previous auth_state. The browser is not
        headless because this can help avoid bot detection, although it is a bit
        slower.
        """

        browser = pw.chromium.launch(headless = False)
        context = browser.new_context(viewport = {'width': 1920, 'height': 1080},
                                      storage_state = auth_state)
        page = context.new_page()

        page.goto(base_url)
        page.wait_for_selector("[data-pressable-container=true]")

        """
        Scrolls down a bit on the threads feed before starting the search,
        again, to simulate some human curiosity.
        """

        page.mouse.wheel(0, 5000)
        human_sleep(5, 7)
        random_mouse_move(page)
    
        for i, code in enumerate(codes):
            """
            this goes through the previously opened cash of threads post_ids
            and extracts the content of the main post, and the content of replies.
            It utilizes a cash of previously accessed posts in order to efficiently
            skip previously visited pages.
            """
            try:

                if code not in cache:
                
                    data = scrape_thread(page, base_url + code)

                    df = pd.DataFrame({'thread_id' : [data['thread']['code']],
                                       'parent_id' : ['is_parent'],
                                       'likes' : [data['thread']['like_count']],
                                       'user_id' : [data['thread']['username']],
                                       'content' : [data['thread']['text']],
                                       'published_on' : [data['thread']['published_on']],
                                       'source' : ['threads']})

                    results = pd.concat([results, df], ignore_index = True)

                    for reply in data['replies']:
                        
                        df = pd.DataFrame({'thread_id' : [reply['code']],
                                           'parent_id' : [code],
                                           'likes' : [reply['like_count']],
                                           'user_id' : [reply['username']],
                                           'content' : [reply['text']],
                                           'published_on' : [reply['published_on']],
                                           'source' : ['threads']})

                        results = pd.concat([results, df], ignore_index = True)

                    """
                    results are saved after iterating through a post, previously
                    visited codes are removed from the codes set and added to the
                    cache set. Both are saved. This is done at every step as a
                    precaution because lost internet connections and throttled
                    or blocked access to Threads may impact the collected data in
                    memory.
                    """

                    results.to_parquet(path_to_results)

                    codes.remove(code)
                    safe_saver(codes, path_to_ids)
                    cache.add(code)
                    safe_saver(cache, path_to_cache)
                
            except Exception as e:
                print(f"❗❗❗Failed to scrape post {code}: {str(e)}.")


            """
            This is a short sleep after each post, to simulate some human
            randomness and to not access too much too fast, which can
            flag the use of a bot. A longer break is performed every
            10 to 15 posts to simulate human behavior.
            """

            human_sleep(3, 5)
            break_time = random.randint(10, 15)
            
            if (i + 1) % break_time == 0:
                human_sleep(7, 10)

    """
    Simple check on the length of the dataset at the start and at the end,
    due to the fact that internet and blocking can affect functionality.
    """

    new_results_length = len(results)

    if old_results_length < new_results_length:
        print(f"Script complete. Results saved to {path_to_results} successfully! A "
              f"total of {new_results_length - old_results_length} comments were "
              f"retrieved.")
    else:
        print("Failed to save results or no new results were retrieved, please check "
              "to ensure that there aren't any issues with the webscrapping.")
