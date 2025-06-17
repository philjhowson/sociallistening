import json
import pickle
import os

def safe_saver(item, path = None):
    """
    Make sure to save items safely! It will print an error if there is
    an issue such as a PermissionError and it will print the error
    for visual notification of the issue.
    
    args:
        item: the object to save.
        path: the path to the folder you want to save it including the
            file name. Handles .pkl and .json endings, if it is neither
            of these, it defaults to .pkl, and will add the .pkl
            extension to the name you specified.

    e.g. safe_saver(item_to_save, 'path/to/item/item_name.json')
    """

    if path:

        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok = True)
        
    else:
        print("❗❗❗ No Path given. File not saved.")

        return None

    try:

        extension = path.split('.')[-1]

        if extension == 'json':
            with open(path, 'w', encoding = 'utf-8') as f:
                json.dump(item, f, indent = 3, ensure_ascii = False)
        elif extension == 'pkl':
            with open(path, 'wb') as f:
                pickle.dump(item, f)
        else:
            path += '.pkl'
            with open(path, 'wb') as f:
                pickle.dump(item, f)

    except Exception as e:
        print(f"❗❗❗ Failed to save file '{path}': {e}")

    return None

def safe_loader(path):
    """
    easy loader for .json and .pkl files.
    args:
        path: path to the file, expects .json or .pkl at the end
              e.g. safe_loader('data/raw/youtube/youtube_cache.json')
    """
    if os.path.exists(path):
        extension = path.split('.')[-1]

        try:
            if extension == 'json':
                with open(path, 'r') as f:
                    return json.load(f)
                
            elif extension == 'pkl':
                with open(path, 'rb') as f:
                    return pickle.load(f)
                
            else:
                print(f"❗❗❗ Error. File at {path} is unsupported file type!")

        except Exception as e:
            print(f"❗❗❗ Error loading file at {path}: {e}")

    else:
        print(f"❗❗❗ File at {path} not found!")

    return None