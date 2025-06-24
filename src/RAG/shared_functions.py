from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import hashlib
import json
import pickle
import os
import pandas as pd
import spacy
import re
import string
import stanza
import html

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

def hash_question(question):
    return hashlib.sha256(question.encode('utf-8')).hexdigest()

def preprocess_data(text):

    text = html.unescape(text)   
    text = text.lower().strip()
    text = text.replace('“', "\"").replace('”', "\"").replace('‘', "'").replace('’', "'")
    text = text.replace('apple care', 'applecare').replace('samsung care', 'samsungcare')
    text = re.sub(r'？', '?', text)
    text = re.sub(r'¿', '?', text)
    text = re.sub(r'！', '!', text)
    text = re.sub(r'﹗', '!', text)
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'(?<= )|(?<=^)!(?=.)', '', text)
    text = re.sub(r'\?+', '?', text)
    text = re.sub(r'(?:(?<= )|(?<=^))\?(?=.)', '', text)

    if re.search(r'\w\s*>\s*\w', text):
        text = re.sub(r'\s*>\s*', ' then ', text)
    else:
        text = text.replace('>', '')

    text = ' '.join([re.sub(r'n(o{2,})', 'no', word) for word in text.split()])

    return text

punct_to_remove = set(string.punctuation)
punct_to_remove -= {'!', '?', "'"}
punct_to_remove.update(['«', '»', '„', '“', '”', '‹', '›',
                        '–', '—', '…', '‚', '‐', '′', '″',
                        '′′', '«', '»', '´', '`', '¸', '·',
                        '•', '。', '、', '，', '．', '：', '；',
                        '（', '）', '【', '】', '『', '』', '〝',
                        '〞'])

def remove_punctuation(text, whitelist = None):

    cleaned_words = []

    if whitelist:

        for word in text.split():
            if word in whitelist:
                cleaned_words.append(word)

            elif word.endswith('...') or word.endswith('…'):
                core = ''.join(char for char in word if char not in punct_to_remove)
                cleaned_words.append(core + '...')         

            else:
                cleaned_word = ''.join(char for char in word if char not in punct_to_remove)
                cleaned_words.append(cleaned_word)

        text = ' '.join(cleaned_words)

    else:
        
        for word in text.split():
            cleaned_word = ''.join(char for char in word if char not in punct_to_remove)
            cleaned_words.append(cleaned_word)

        text = ' '.join(cleaned_words)

    return re.sub(r'\s+', ' ', text)


def reduce_repeats(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

try:
    #spacy_codes = safe_loader('data/raw/spacy_codes.pkl')
    spacy_codes = {'en' : 'en_core_web_sm'}
except Exception as e:
    print(f"Failed to load spacy_codes: {e}")
    spacy_codes = None
    
spacy_models = {}

if spacy_codes:
    for lang in spacy_codes:
        model_name = spacy_codes[lang]
        spacy_models[lang] = spacy.load(model_name)


def remove_stopwords(text, lang, custom = None):
    
    stopwords = set()

    spacy_model = spacy_models.get(lang)

    if spacy_model:
        stopwords.update(spacy_model.Defaults.stop_words)

        if lang == 'en':
            stopwords.update({'be', 'have', 'do', 'does', 'did', 'are', 'am', 'was', 'were', 'being', 'been'})

    elif custom and lang in custom:
        stopwords.update(custom[lang])

    elif not stopwords:

        return text

    filtered_words = [word for word in text.split() if word not in stopwords]
    
    return ' '.join(filtered_words)

_stanza_pipelines = {}

def get_stanza_pipeline(lang):

    if lang not in _stanza_pipelines:
        try:
            stanza.download(lang, quiet = True)
            _stanza_pipelines[lang] = stanza.Pipeline(lang = lang, processors = 'tokenize,mwt,pos,lemma',
                                                      use_gpu = False, verbose = False)

        except Exception as e:
            _stanza_pipelines[lang] = None
            
    return _stanza_pipelines.get(lang)

def lemmatize(text, lang):

    spacy_model = spacy_models.get(lang)
        
    if spacy_model:
        doc = spacy_model(text)
        lemmas = [token.lemma_ for token in doc]
        return ' '.join(lemmas)
    
    stanza_pipeline = get_stanza_pipeline(lang)
    
    if stanza_pipeline:
        doc = stanza_pipeline(text)
        lemmas = []
        
        for sentence in doc.sentences:
            for word in sentence.words:
                lemmas.append(word.lemma if word.lemma else word.text)
                
        return ' '.join(lemmas)

    return text

def detect_language(text):
    
    try:
        return detect(text)

    except LangDetectException:
        return 'unknown'

def pipeline(text, custom_stops = None, whitelist = None, spacy = None):

    lang = detect_language(text)
    cleaned_text = preprocess_data(text)
    cleaned_text = remove_stopwords(cleaned_text, lang, custom_stops)
    cleaned_text = lemmatize(cleaned_text, lang)
    cleaned_text = remove_punctuation(cleaned_text, whitelist)

    return cleaned_text