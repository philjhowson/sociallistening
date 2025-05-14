import time
from mastodon import Mastodon
from html.parser import HTMLParser


class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
    def handle_data(self, data):
        self.text.append(data)
    def get_data(self):
        return ''.join(self.text)

def strip_html_tags(html):
    stripper = HTMLStripper()
    stripper.feed(html)
    return stripper.get_data()

def connect_to_instance(base_url, access_token):
    return Mastodon(
        access_token=access_token,
        api_base_url=base_url
    )

def matches_keywords(text, keywords):
    return any(keyword.lower() in text.lower() for keyword in keywords)

def collect_from_instance(base_url, keywords, access_token, max_posts_per_instance):
    print(f"\n🔍 Connecting to {base_url}")
    mastodon = connect_to_instance(base_url, access_token)
    matched = []
    last_id = None

    while len(matched) < max_posts_per_instance:
        try:
            timeline = mastodon.timeline_public(max_id=last_id, limit=40)
            if not timeline:
                break

            for status in timeline:
                content_text = strip_html_tags(status["content"])
                if matches_keywords(content_text, keywords):
                    matched.append({
                        "instance": base_url,
                        "poster": status["account"]["acct"],
                        "time": status["created_at"].isoformat(),
                        "content": content_text
                    })

            print(f"Collected {len(matched)} matches from {base_url}")
            last_id = timeline[-1]["id"]
            time.sleep(1)

        except Exception as e:
            print(f"Error on {base_url}: {e}")
            break

    return matched
