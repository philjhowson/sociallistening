import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def evaluate_clusters(path = None, base_path = 'data/processed'):

    full_path = os.path.join(base_path, path)
    data = pd.read_parquet(full_path)

    for cluster in set(data['clusters']):
        clust = data[data['clusters'] == cluster]
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'inspect clusters from data formatting')
    parser.add_argument('--path', required = True, help = 'Enter filename. Assumes the file is a parquet.')
    parser.add_argument('--base_path', default = 'data/processed', help = 'Default data/processed. Enter path relative to the root directory here to change it.')

    arg = parser.parse_args()

    evaluate_clusters(path = arg.path, base_path = arg.base_path)

