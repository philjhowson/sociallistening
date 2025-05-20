import json
import csv
import time
import random
import logging
from typing import Dict
from parsel import Selector
from nested_lookup import nested_lookup
from playwright.sync_api import sync_playwright
from shared_functions import safe_loader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_thread(data: Dict) -> Dict:
    """Parse Twitter tweet JSON dataset for the most important fields"""
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
    result["videos"] = list(set(result["videos"] or []))
    if result["reply_count"] and type(result["reply_count"]) != int:
        result["reply_count"] = int(result["reply_count"].split(" ")[0])
    result[
        "url"
    ] = f"https://www.threads.net/@{result['username']}/post/{result['code']}"
    return result

def scrape_thread(url: str) -> dict:
    """Scrape Threads post and replies from a given URL"""
    with sync_playwright() as pw:
        # start Playwright browser
        browser = pw.chromium.launch()
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # go to url and wait for the page to load
        page.goto(url)
        # wait for page to finish loading
        page.wait_for_selector("[data-pressable-container=true]")
        # find all hidden datasets
        selector = Selector(page.content())
        hidden_datasets = selector.css('script[type="application/json"][data-sjs]::text').getall()
        # find datasets that contain threads data
        for hidden_dataset in hidden_datasets:
            # skip loading datasets that clearly don't contain threads data
            if '"ScheduledServerJS"' not in hidden_dataset:
                continue
            if "thread_items" not in hidden_dataset:
                continue
            data = json.loads(hidden_dataset)
            # datasets are heavily nested, use nested_lookup to find 
            # the thread_items key for thread data
            thread_items = nested_lookup("thread_items", data)
            if not thread_items:
                continue
            # use our jmespath parser to reduce the dataset to the most important fields
            threads = [parse_thread(t) for thread in thread_items for t in thread]
            return {
                # the first parsed thread is the main post:
                "thread": threads[0],
                # other threads are replies:
                "replies": threads[1:],
            }
        raise ValueError("could not find thread data in page")

def human_sleep(min_seconds=2.5, max_seconds=6.5):
    """Sleep for a random float duration between min and max seconds"""
    duration = random.uniform(min_seconds, max_seconds)
    logging.info(f"Sleeping for {duration:.2f} seconds")
    time.sleep(duration)

path_to_ids = 'data/raw/threads_post_ids.pkl'

codes = safe_loader(path_to_ids)
codes = list(codes)
random.shuffle(codes)

if __name__ == "__main__":
    
    for i, c in enumerate(codes[0:50]):
        logging.info(f"Scraping post {i+1}: {c}")
        try:
            data = scrape_thread("https://www.threads.net/" + c)

            parent_thread_code = c
            with open("threads_data.csv", 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([data['thread']['code'], "", data['thread']['like_count'],
                                 data['thread']['username'], data['thread']['text'], time.time()])
                for r in data['replies']:
                    writer.writerow([r['code'], parent_thread_code, r['like_count'],
                                     r['username'], r['text'], time.time()])
        except Exception as e:
            logging.error(f"Failed to scrape post {c}: {str(e)}")

        # Short sleep after every request
        human_sleep()

        # Take a longer break every 10 posts
        if (i + 1) % 10 == 0:
            long_sleep = random.uniform(10, 20)
            logging.info(f"Taking a longer break for {long_sleep:.2f} seconds")
            time.sleep(long_sleep)
