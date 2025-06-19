import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bertopic import BERTopic
import shared_functions
import argparse

path_to_processed = 'data/processed'
    
def sentiment_analysis():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata.parquet")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment").to(device)

    print('Analyzing sentiment...')

    all_preds = []
    all_probs = []

    texts = data['cleaned_text']
    batch_size = 256

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size].tolist()
            # Make sure all entries are strings (replace None or non-str with empty string)
            batch_texts = [str(t) if isinstance(t, str) else "" for t in batch_texts]

            inputs = tokenizer(batch_texts, 
                               return_tensors = 'pt', 
                               truncation = True, 
                               padding = True,
                               max_length = 512).to(device)

            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim = 1)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())

    data['sentiment_label'] = all_preds
    data['sentiment_label'] = data['sentiment_label'].map({0: -1, 1: 0, 2: 1})

    data.to_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment.parquet")

def topics():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment.parquet")

    comment_embeddings_archive = np.load(f"{path_to_processed}/masterdata_embeddings.npz")
    comment_embeddings = comment_embeddings_archive['embeddings']

    docs = data['cleaned_text'].tolist()
    topic_model = BERTopic(nr_topics = 5, min_topic_size = 100)

    print('Analyzing topics...')
    
    topics, probs = topic_model.fit_transform(docs, embeddings = comment_embeddings)
    data['topic_label'] = topics

    topics_info = topic_model.get_topic_info()

    data.to_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")
    topics_info.to_csv(f"{path_to_processed}/reddit_topics.csv")

    fig = topic_model.visualize_barchart()
    fig.write_html('images/masterdata_topics.html')

def sentiment_text():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")

    data = data[['topic_label', 'sentiment_label', 'text', 'likes']]
    data.to_csv('data/processed/filtered_masterdata_sentiment_topics.csv')

    for topic in data['topic_label'].unique():
        temp = data[data['topic_label'] == topic]
        negative = temp[temp['sentiment_label'] == -1]
        positive = temp[temp['sentiment_label'] == 1]
        if not negative.empty:
            negative = np.random.choice(negative['text'], 10)
            print('Negative comments:\n', negative)
        else:
            print(f"No negative comments found for topic #{topic}.")
        if not positive.empty:
            positive = np.random.choice(positive['text'], 10)
            print('Positive comments:\n', positive)
        else:
            print(f"No positive comments found for topic #{topic}.")

def arg_parse(function):

    match function:
        case 'analysis':
            sentiment_analysis()
        case 'topic':
            topics()
        case 'text':
            sentiment_text()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Just a simple argument parser.')
    parser.add_argument('--function', required = True, help = 'Choose either analysis, topic, text.')
    
    arg = parser.parse_args()

    arg_parse(function = arg.function)