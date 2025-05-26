import numpy as np
import random
import time

def human_sleep(min_sec = 3.0, max_sec = 5.0):
    time.sleep(np.random.uniform(min_sec, max_sec))

def random_mouse_move(page):
    x = random.randint(0, 800)
    y = random.randint(0, 600)
    page.mouse.move(x, y, steps = random.randint(5, 15))

def random_mouse_hover(page, target):

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

    url = f"{base_url}{target}"

    try:
        tab = context.new_page()
        tab.goto(url)

        human_sleep(2, 4)  

        if target.endswith('/media'):
            human_sleep(10, 15)
            tab.close()
            human_sleep(5, 10)

        else:
            tab.mouse.wheel(0, random.randint(500, 1000))
            human_sleep(5, 10)

        tab.close()
        human_sleep(1, 2)

    except Exception as e:
        print(f"❗❗❗ Error visiting post: {e}")
