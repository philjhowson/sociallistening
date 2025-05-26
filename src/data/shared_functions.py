import json
import pickle
import reverse_geocoder as rg
import os
import re
import regex
import string
import numpy as np
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import nltk
from nltk.corpus import stopwords
import spacy
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

def get_country_code(row):
    
    if row['lat'] != 0 and row['lon'] != 0:
        results = rg.search((row['lat'], row['lon']), mode = 1)
        country_code = results[0]['cc']
        
        return country_code
    
    return np.nan

def preprocess_data(text):

    text = html.unescape(text)   
    text = text.lower().strip()
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
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
    
def detect_language(text):
    
    try:
        return detect(text)

    except LangDetectException:
        return 'unknown'

def reduce_repeats(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def remove_stopwords(text, lang, custom = None, spacy_models = None):
    
    stopwords = set()

    if spacey_models:
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

def has_enough_char(text, min_char = 3):
    text = re.sub(r'\s+', '', text)
    return len(regex.findall(r'\p{L}', str(text))) >= min_char
    
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

def lemmatize(text, lang, spacy_models = None):

    if spacy_models:
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

def pipeline(df, input_col, custom_stops = None, whitelist = None, spacy = None,
             geolocation = True, detect_lang = True, preprocessing = True,
             remove_stops = True, lemma = True, punctuation = True,
             characters = True):

    language_column = f"{input_col}_language"
    cleaned_column = f"cleaned_{input_col}"

    if geolocation:
        df['geolocation'] = df.apply(get_country_code, axis = 1)
    if detect_lang:
        df[language_column] = df[input_col].apply(detect_language)
    if preprocessing:
        df[cleaned_column] = df[input_col].apply(preprocess_data)
    if remove_stops:
        df[cleaned_column] = df.apply(lambda x: remove_stopwords(x[cleaned_column], x[language_column], custom_stops, spacy), axis = 1)
    if lemma:
        df[cleaned_column] = df.apply(lambda x: lemmatize(x[cleaned_column], x[language_column], spacy), axis = 1)
    if punctuation:
        df[cleaned_column] = df[cleaned_column].apply(lambda x: remove_punctuation(x, whitelist))
    if characters:
        df['enough_char'] = df[cleaned_column].apply(has_enough_char)
        df = df[df['enough_char']]

    return df
