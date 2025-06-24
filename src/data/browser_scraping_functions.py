import numpy as np
import random
import time

def human_sleep(min_sec = 3.0, max_sec = 5.0):
    """
    Simple function to simulate how people will periodically stop
    scrolling to look at something. Randomized value.
    args:
        min_sec: minimum possible sleep time.
        max_sec: maximum possible sleep time.
    """
    time.sleep(np.random.uniform(min_sec, max_sec))

def random_mouse_move(page):
    """
    Moves the mouse to a random x,y position using a non-straight
    line.
    args:
        page: the page object being used by playwright.
    """
    x = random.randint(0, 800)
    y = random.randint(0, 600)
    page.mouse.move(x, y, steps = random.randint(5, 15))

def random_mouse_hover(page, target):
    """
    Moves the mouse to hover over an object on a page, designed
    to simulate how people may be curious about certain links, posts,
    and users on social media.
    args:
        page: page object used by playwright.
        target: the target to hover over.
    """

    selector = f'a[href="{target}"]'
    
    try:
        element = page.query_selector(selector)
        
        if element:
            
            bbox = element.bounding_box()

            if bbox:
                x_offset = random.uniform(3, bbox['width'] - 5)
                y_offset = random.uniform(3, bbox['height'] - 5)
                page.mouse.move(bbox['x'] + x_offset, bbox['y'] + y_offset)
                human_sleep(0.5, 3.5)
                
    except Exception:
        pass


def open_random_link(target, base_url, context):
    """
    Opens a randomized link from the page in order to simulate
    curiosity that people will have when scrolling. It will spend
    a small amount of time looking at the picture, if an image was
    loaded, if it is a page, it will scroll down a bit before closing
    the tab.
    args:
        target: target link to open.
        base_url: the base url of the page, the threads page in this case
        context: the original context opened with playwrite such that this
                 will simulate opening a new tab, not a new window.
    """

    url = f"{base_url}{target}"

    try:
        tab = context.new_page()
        tab.goto(url)

        human_sleep(2, 4)  

        if target.endswith('/media'):
            human_sleep(2, 5)
            tab.close()
            human_sleep(2, 3)

        else:
            tab.mouse.wheel(0, random.randint(500, 1000))
            human_sleep(5, 10)

        tab.close()
        human_sleep(1, 2)

    except Exception as e:
        print(f"❗❗❗ Error visiting post: {e}")
