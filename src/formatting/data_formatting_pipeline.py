import reverse_geocoder as rg
import re
import regex
import string
import stanza
import numpy as np
import pandas as pd
from shared_functions import safe_loader
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import spacy
import html

def get_country_code(row):
    """
    Takes a row from a dataset and converts lon/lat to a
    country code. Returns np.nan if nothing is found.
    """
    
    if row['lat'] != 0 and row['lon'] != 0:
        results = rg.search((row['lat'], row['lon']), mode = 1)
        country_code = results[0]['cc']
        
        return country_code
    
    return np.nan

keywords1 = safe_loader('data/raw/keywords1.pkl')
keywords2 = safe_loader('data/raw/keywords2.pkl')

def preprocess_data(text):
    """
    Cleans up text by removing html code, lowercasing, removing trailing and leading
    white space, normalizes a large number of items to a consistent single format.
    keywords1 and keywords2 is loaded above and used here to maintain the privacy of
    the words to be normalized because this was not built as part of a personal project.
    """

    text = html.unescape(text)   
    text = text.lower().strip()
    text = text.replace('“', "\"").replace('”', "\"").replace('‘', "'").replace('’', "'")
    text = text.replace(keywords1[0], keywords1[1]).replace(keywords2[0], keywords2[1])
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

"""
Updates the list of punctuations to include other non-English
punctuation that is commonly used.
"""

punct_to_remove = set(string.punctuation)
punct_to_remove -= {'!', '?', "'"}
punct_to_remove.update(['«', '»', '„', '“', '”', '‹', '›',
                        '–', '—', '‚', '‐', '′', '″',
                        '′′', '«', '»', '´', '`', '¸', '·',
                        '•', '。', '、', '，', '．', '：', '；',
                        '（', '）', '【', '】', '『', '』', '〝',
                        '〞'])

def remove_punctuation(text, whitelist = None):
    """
    Removes punctuation. Splits a text up into chunks and removes all
    punctuation, although '...' for example is kept as it can be useful
    for many sentiment analysis systems and embeddings.
    """

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
    """
    Uses langdetect to attempt to identify the language.
    """
    
    try:
        return detect(text)

    except LangDetectException:
        return 'unknown'

"""
Loads spacy for English. More spacy models can be loaded
using a spacy_codes.pkl that should be set up to look like a
dictionary.
"""

try:
    #spacy_codes = safe_loader('data/raw/spacy_codes.pkl')
    spacy_codes = {'en' : 'en_core_web_sm'}
except Exception as e:
    print(f"Failed to load spacy_codes: {e}")
    spacy_codes = None
    
spacy_models = {}

"""
Loads all the models specified in spacy_codes.
"""

if spacy_codes:
    for lang in spacy_codes:
        model_name = spacy_codes[lang]
        spacy_models[lang] = spacy.load(model_name)

def remove_stopwords(text, lang, custom = None):
    """
    Removes english stopwords in spacy (or other languages if modified above)
    and utilizes a custom set of stopwords for other languages to remove
    other stopwords. This was largely done because spacy does not have strong
    enough models to support all the languages in my dataset.
    """
    
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

def has_enough_char(text, min_char = 3):
    """
    Determines if a string has enough characters to be kept. If it has 3
    characters or less it will be marked for removal.
    """
    text = re.sub(r'\s+', '', text)
    return len(regex.findall(r'\p{L}', str(text))) >= min_char
    
_stanza_pipelines = {}

def get_stanza_pipeline(lang):
    """
    Gets all the relevant stanza pipelines for lemmatization. This was done largely
    because spacy did not have all the required and relevant language support.
    """

    if lang not in _stanza_pipelines:
        try:
            stanza.download(lang, quiet = True)
            _stanza_pipelines[lang] = stanza.Pipeline(lang = lang, processors = 'tokenize,mwt,pos,lemma',
                                                      use_gpu = False, verbose = False)

        except Exception as e:
            _stanza_pipelines[lang] = None
            
    return _stanza_pipelines.get(lang)

def lemmatize(text, lang):
    """
    Lemmatizes each word using either spacy (if available) or
    stanza.
    """

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

def pipeline(data, input_col, iso_codes = None, custom_stops = None, whitelist = None,
             spacy = None, geolocation = True, detect_lang = True,
             preprocessing = True, remove_stops = True, lemma = True,
             punctuation = True, characters = True):
    
    """
    Pass a whole df and get the processed df back. This performs geolocation,
    language detection, preprocessing functions, stop word removal, lemmatization,
    and finally, checks if there are enough characters and removes rows where
    text is too short to reliably find any information.
    """

    df = data.copy()

    language_column = f"{input_col}_language"
    cleaned_column = f"cleaned_{input_col}"

    if geolocation:
        print('Geolocation started...')
        df['geolocation'] = df.apply(get_country_code, axis = 1)
        df['geolocation'] = df['geolocation'].fillna(df['country'])
    
    if detect_lang:
        print('Language detection started...')
        df[language_column] = df[input_col].apply(detect_language)

    if preprocessing:
        print('Preprocessing started...')
        df[cleaned_column] = df[input_col].apply(preprocess_data)

    if remove_stops:
        print('Remove stops started...')
        df[cleaned_column] = df.apply(lambda x: remove_stopwords(x[cleaned_column], x[language_column], custom_stops), axis = 1)
    
    if lemma:
        print('Lemmatization started...')
        df[cleaned_column] = df.apply(lambda x: lemmatize(x[cleaned_column], x[language_column]), axis = 1)
    
    if punctuation:
        print('Punctuation removal started...')
        df[cleaned_column] = df[cleaned_column].apply(lambda x: remove_punctuation(x, whitelist))
    
    if characters:
        print('Removing stray strings. started...')
        df['enough_char'] = df[cleaned_column].apply(has_enough_char)
        all_text = len(df)
        df = df[df['enough_char']]
        removed = len(df)
        print(f"After data processing {round(removed/all_text * 100, 2)}% of data points remain.")

    return df
