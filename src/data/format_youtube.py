from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import shared_functions
import pandas as pd
import pickle
import os

def format_youtube():

    path_to_processed = 'data/processed'

    data = pd.read_parquet('data/raw/youtube_results.parquet')

    custom_stops = shared_functions.safe_loader('data/raw/custom_stopwords.pkl')
    whitelist = shared_functions.safe_loader('data/raw/whitelisted_characters.pkl')
    spacy = shared_functions.safe_loader('data/raw/spacy_codes.pkl')

    data = shared_functions.pipeline(data, 'video_title', custom_stops = custom_stops,
                                     spacy = spacy, whitelist = whitelist)
    data = shared_functions.pipeline(data, 'comment_text', custom_stops = custom_stops,
                                     spacy = spacy, whitelist = whitelist,
                                     geolocation = False)
    
    """
    data['geolocation'] = data.apply(shared_functions.get_country_code, axis = 1)
    data['country_code'] = data['geolocation'].fillna(data['country'])

    data['title_language'] = data['video_title'].apply(shared_functions.detect_language)
    data['comment_language'] = data['comment_text'].apply(shared_functions.detect_language)

    data['cleaned_video_title'] = data['video_title'].apply(shared_functions.preprocess_data)
    data['cleaned_comment'] = data['comment_text'].apply(shared_functions.preprocess_data)



    data['cleaned_video_title'] = data.apply(lambda x: shared_functions.remove_stopwords(x['cleaned_video_title'], x['title_language'], custom_stops), axis = 1)
    data['cleaned_comment'] = data.apply(lambda x: shared_functions.remove_stopwords(x['cleaned_comment'], x['comment_language'], custom_stops), axis = 1)

    data['cleaned_video_title'] = data.apply(lambda x: shared_functions.lemmatize(x['cleaned_video_title'], x['title_language']), axis = 1)
    data['cleaned_comment'] = data.apply(lambda x: shared_functions.lemmatize(x['cleaned_comment'], x['comment_language']), axis = 1)



    data['cleaned_video_title'] = data['cleaned_video_title'].apply(lambda x: shared_functions.remove_punctuation(x, whitelist))
    data['cleaned_comment'] = data['cleaned_comment'].apply(lambda x: shared_functions.remove_punctuation(x, whitelist))

    data['enough_char'] = data['cleaned_comment'].apply(shared_functions.has_enough_char)
    data = data[data['enough_char']]
    """

    data.to_parquet(f"{path_to_processed}/cleaned_youtube_results.parquet")
    
    print(data[['cleaned_video_title', 'cleaned_comment_text']].head(20))
    print(data['comment_language'].value_counts())
    print(data.columns)

    """

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data['video_title'].tolist())

    dbscan = DBSCAN(eps = 0.5, min_samples = 5, metric = 'cosine')
    clusters = dbscan.fit_predict(embeddings)
    
    os.makedirs(path_to_processed, exist_ok = True)

    columns_to_drop = ['channel_id', 'channel_name', 'video_id',
                       'comment_author', 'country', 'lat', 'lon',
                       'geolocation', 'enough_char']

    data.drop(columns = columns_to_drop, inplace = True)

    data.to_parquet(f"{path_to_processed}/cleaned_youtube_results.parquet")
    """
if __name__ == '__main__':
    format_youtube()
