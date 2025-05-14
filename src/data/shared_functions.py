
import json
import pickle
import os

def safe_saver(item, path, file):
    """
    Make sure to save items safely! It will print an error if there is
    an issue such as a PermissionError and it will print the error
    for visual notification of the issue.
    
    args:
        item: the object to save.
        path: the path to the folder you want to save it.
            If saving in the root directory, leave this argument blank.
        file: the file name. Handles .pkl and .json endings,
            if it is neither of these, it defaults to .pkl, and will add
            the .pkl extension to the name you specified.
    """

    if path:
        os.makedirs(path, exist_ok = True)
        full_path = os.path.join(path, file)
    else:
        full_path = file

    extension = file.split('.')[-1]

    try:

        if extension == 'json':
            with open(full_path, 'w') as f:
                json.dump(item, f, indent = 3)
        elif extension == 'pkl':
            with open(full_path, 'wb') as f:
                pickle.dump(item, f)
        else:
            full_path += '.pkl'
            with open(full_path, 'wb') as f:
                pickle.dump(item, f)

    except Exception as e:
        print(f"❗❗❗ Failed to save file '{file}': {e}")
