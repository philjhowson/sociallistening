from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import shared_functions
import pandas as pd
import numpy as np
import torch
import argparse


path_to_processed = 'data/processed/'

def frequency_grouping():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device = device)

    results = []

    for topic in data['topic'].unique():
        subset = data[data['topic'] == topic].copy()
        index = subset.index

        embeddings = model.encode(subset['text'].tolist(), show_progress_bar = True)

        dbscan = DBSCAN(eps = 0.4, min_samples = 3, metric = 'cosine')
        labels = dbscan.fit_predict(embeddings)

        results.extend(zip(index, labels))

    results_df = pd.DataFrame(results, columns = ['index', 'labels']).set_index('index').sort_index()
    data['labels'] = results_df['labels']

    data.to_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")

def scoring():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")

    scalar = 1

    grouped_data = data.groupby(['topic', 'labels']).agg(likes = ('likes', 'sum'), count = ('labels', 'size')).reset_index()
    grouped_data['log_likes'] = np.log1p(grouped_data['likes'])
    grouped_data['log_count'] = np.log1p(grouped_data['count'])
    grouped_data['score'] = grouped_data['log_likes'] + scalar * grouped_data['log_count']
    grouped_data.sort_values(['topic', 'score'], ascending = False, inplace = True)
    grouped_data.to_csv(f"{path_to_processed}/scores.csv")

    zipped = zip(grouped_data['topic'], grouped_data['labels'], grouped_data['score'])

    mappings = {(topic, label) : score for topic, label, score in zipped}
    data['score'] = data.apply(lambda x : mappings.get((x['topic'], x['labels'])), axis = 1)

    data.to_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")

def find_themes():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")

    themes = {}
    topics = sorted(data['topic'].unique())

    for index, topic in enumerate(topics, start = 1):

        print(f"Processing topic {index} of {len(topics)}...")

        temp = data[data['topic'] == topic].copy()
        themes[topic] = shared_functions.get_keywords(temp['text'], temp['labels'], temp['score'], top_n = 20, top_score = 10)

    themes_df = pd.DataFrame(themes)

    themes_df.to_csv(f"{path_to_processed}/themes.csv")

def example_text():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")

    zipped = list(zip(data['topic'], data['labels']))
    unique_combos = np.unique(zipped, axis = 0)
    unique_combos = [tuple(row) for row in unique_combos]

    for topic, label in unique_combos:

        temp = data[(data['topic'] == topic) & (data['labels'] == label)]
        ranked_scores = sorted(temp['score'].unique(), reverse = True)
        temp = temp[temp['score'].isin(ranked_scores[0:10])]
        n_samples = min(10, len(temp))
        choice = np.random.choice(temp.index, replace = False, size = n_samples)
        temp = temp.loc[choice]

        print(f"Example text from topic: {topic}; label: {label}")
        for item in temp['text']:
            print(item)
            print('\n\n\n\n\n--------------------\n\n\n\n\n')
        print(choice)

def run_function(function):

    match function:
        case 'freq':
            frequency_grouping()
        case 'score':
            scoring()
        case 'theme':
            find_themes()
        case 'example':
            example_text()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'parser for scoring functions')
    parser.add_argument('--function', default = 'score', help = 'freq, score, theme, or example.')

    arg = parser.parse_args()

    run_function(function = arg.function)
