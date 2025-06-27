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
    """
    Loads in the data and 'nlptown/bert-base-multilingual-uncased-sentiment' for
    sentiment analysis. This is an accurate but relatively slow model compared
    to 'cardiffnlp/twitter-xlm-roberta-base-sentiment', therefore, if you have
    a very large dataset, you may consider using a different model, but some
    coding will need to be changed, namely the mapping of sentiment labels.
    """

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata.parquet")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    print('Analyzing sentiment...')

    """
    Sets up the cleaned text, and records all predictions and probabilities.
    Passes the text through the model in batches of 256. The probabilities
    are not saved, but can be easily saved if you wish to have that data.
    Assigns the predictions to a column in the dataset.
    """

    all_preds = []
    all_probs = []

    texts = data['cleaned_text']
    batch_size = 256
    total_batches = (len(texts) + batch_size - 1) // batch_size

    model.eval()
    with torch.no_grad():
        for index, i in enumerate(range(0, len(texts), batch_size)):
            if index % 100 == 0:
                print(f"Analyzing batch {index}/{total_batches}")

            batch_texts = texts[i:i + batch_size].tolist()
            batch_texts = [str(t) if isinstance(t, str) else '' for t in batch_texts]

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

    data['sentiment'] = all_preds
    data['sentiment'] = data['sentiment'].map({0: -1, 1: -0.5, 2: 0, 3: 0.5, 4: 1})

    data.to_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment.parquet")

def topics():
    """
    This uses BERTopic to find clustering of general trends or themes in the data
    using the embeddings, saves those to a 'topics' column and prints a bar chart
    of the topics found.
    """

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment.parquet")

    comment_embeddings_archive = np.load(f"{path_to_processed}/masterdata_embeddings.npz")
    comment_embeddings = comment_embeddings_archive['embeddings']

    docs = data['cleaned_text'].tolist()
    topic_model = BERTopic(nr_topics = 5, min_topic_size = 100)

    print('Analyzing topics...')
    
    topics, probs = topic_model.fit_transform(docs, embeddings = comment_embeddings)
    data['topic'] = topics

    topics_info = topic_model.get_topic_info()

    data.to_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")
    topics_info.to_csv(f"{path_to_processed}/reddit_topics.csv")

    fig = topic_model.visualize_barchart()
    fig.write_html('images/masterdata_topics.html')

def arg_parse(function):

    match function:
        case 'analysis':
            sentiment_analysis()
        case 'topic':
            topics()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Just a simple argument parser.')
    parser.add_argument('--function', required = True, help = 'Choose either analysis or topic.')
       
    arg = parser.parse_args()

    arg_parse(function = arg.function)