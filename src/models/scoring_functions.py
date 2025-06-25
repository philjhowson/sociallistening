from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import shared_functions
import pandas as pd
import numpy as np
import torch
import argparse


path_to_processed = 'data/processed/'

def frequency_grouping():
    """
    This uses DBScan and embeddings to find counts for similar comments.
    Comments that are clustered together in the DBScan space are assumed
    to be similar in nature.
    """

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device = device)

    results = []

    """
    for each unique topic, subsets the data to include just that topic and finds
    comments that cluster together.
    """

    for topic in data['topic'].unique():
        subset = data[data['topic'] == topic].copy()
        index = subset.index

        embeddings = model.encode(subset['text'].tolist(), show_progress_bar = True)

        dbscan = DBSCAN(eps = 0.4, min_samples = 3, metric = 'cosine')
        labels = dbscan.fit_predict(embeddings)

        results.extend(zip(index, labels))

    """
    Creates a dataframe of the results from DBScan and sorts by index so they can be correctly
    added to the master data.
    """

    results_df = pd.DataFrame(results, columns = ['index', 'labels']).set_index('index').sort_index()
    data['labels'] = results_df['labels']

    data.to_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")

def scoring():
    """
    This groups all the data by topic and labels, i.e., the BERTopic defined topics and the
    clusters found from DBScan within those topics. Then the sum of likes and the count
    are created with the aggregate function.
    """

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")

    scalar = 1

    grouped_data = data.groupby(['topic', 'labels']).agg(likes = ('likes', 'sum'), count = ('labels', 'size')).reset_index()
    """
    Creates log_likes and log_count to reduce the effect of viral posts and content.
    Then scoring is done with a simple formula:

        log(likes) + α ⋅ log(count)

    This formula permits more heavy weighting of likes or frequency count, but for this
    project, I kept the weight at 1. Saves the scoring as a .csv and then adds a 'score'
    column to the dataset and saves it.
    """
    grouped_data['log_count'] = np.log1p(grouped_data['count'])
    grouped_data['log_likes'] = np.log1p(grouped_data['likes'])
    grouped_data['score'] = grouped_data['log_likes'] + scalar * grouped_data['log_count']
    grouped_data.sort_values(['topic', 'score'], ascending = False, inplace = True)
    grouped_data.to_csv(f"{path_to_processed}/scores.csv")

    zipped = zip(grouped_data['topic'], grouped_data['labels'], grouped_data['score'])

    mappings = {(topic, label) : score for topic, label, score in zipped}
    data['score'] = data.apply(lambda x : mappings.get((x['topic'], x['labels'])), axis = 1)

    data.to_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")

def run_function(function):

    match function:
        case 'freq':
            frequency_grouping()
        case 'score':
            scoring()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'parser for scoring functions')
    parser.add_argument('--function', default = 'score', help = 'freq or score.')

    arg = parser.parse_args()

    run_function(function = arg.function)
