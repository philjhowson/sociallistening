from playwright.sync_api import sync_playwright

def login_and_save_session():
    """
    This opens a playwright browser to threads so that the user can login and
    their auth_state will be saved. There is sufficient time for the user
    to enter in credentials and even login through Instagram if necessary.
    This must be run before the threads scraper, unless you have a file
    with the auth_state in it already.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless = False)
        context = browser.new_context()
        page = context.new_page()

        page.goto("https://www.threads.com/login")
        
        print("➡ Please log in manually in the opened browser window within 60 seconds...")

        page.wait_for_timeout(60000)

        path_to_state = 'data/raw/threads_auth_state.json'
        context.storage_state(path = path_to_state)
        print(f"✅ Logged-in session saved to {path_to_state}")

        browser.close()

if __name__ == "__main__":
    login_and_save_session()
