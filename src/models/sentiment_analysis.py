import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bertopic import BERTopic
import numpy as np
import matplotlib.pyplot as plt

path_to_processed = 'data/processed'
    
def sentiment_analysis():

    data = pd.read_parquet(f"{path_to_processed}/topic_reduced_reddit_data.parquet")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment").to(device)

    print('Analyzing sentiment...')

    all_preds = []
    all_probs = []

    texts = data['cleaned_comment_body']
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

    data['sentiment'] = all_preds
    data['normalized_sentiment'] = data['sentiment'].map({0: -1, 1: 0, 2: 1})

    data.to_parquet(f"{path_to_processed}/cleaned_processed_reddit_data.parquet")

def topics():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_processed_reddit_data.parquet")

    comment_embeddings_archive = np.load(f"{path_to_processed}/topic_reduced_reddit_comment_embeddings.npz")
    comment_embeddings = comment_embeddings_archive['embeddings']

    docs = data['cleaned_comment_body'].tolist()
    topic_model = BERTopic(nr_topics = 5, min_topic_size = 100)

    print('Analyzing topics...')
    
    topics, probs = topic_model.fit_transform(docs, embeddings = comment_embeddings)
    data['topics'] = topics
    """
    title_to_topic = dict(zip(docs, topics))
    data['video_topic'] = data['cleaned_video_title'].map(title_to_topic)
    """
    topics_info = topic_model.get_topic_info()

    data.to_parquet(f"{path_to_processed}/cleaned_processed_reddit_data.parquet")
    topics_info.to_csv(f"{path_to_processed}/reddit_topics.csv")

    fig = topic_model.visualize_barchart()
    fig.write_html('images/reddit_topics.html')

def sentiment_text():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_processed_reddit_data.parquet")

    data = data[['topics', 'normalized_sentiment', 'comment_body', 'comment_score']]
    data.to_csv('data/processed/filtered_reddit_dataset.csv')

    for topic in data['topics'].unique():
        temp = data[data['topics'] == topic]
        negative = temp[temp['normalized_sentiment'] == -1]
        positive = temp[temp['normalized_sentiment'] == 1]
        negative = np.random.choice(negative['comment_body'], 10)
        positive = np.random.choice(positive['comment_body'], 10)

        print(f"Printing example sentences for topic #{topic}:")
        print('Positive comments:\n', positive)
        print('Negative comments:\n', negative)

def sentiment_visualization():

    source = 'Reddit'

    data = pd.read_parquet(f"{path_to_processed}/cleaned_processed_reddit_data.parquet")

    data['normalized_topics'] = data['topics'].map({-1: 1, 0: 1, 1: 9, 2: 3, 3: 1, 4: 4, 5: 5, 6: 1, 7: 1, 8: 2})
    warranties = data[data['normalized_topics'] != 9]
    topics = warranties.groupby('normalized_topics').agg({'normalized_sentiment' : 'mean'})

    fig = plt.figure(figsize = (10, 10))
    bars = plt.bar(topics.index, topics['normalized_sentiment'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,  height + 0.01,
            f'{height:.3f}', ha = 'center', va = 'bottom', color = 'white',
            fontweight = 'bold')
    plt.xticks([1, 2, 3, 4, 5], ['User Experiences and Policy', 'Cost Benefit Concerns', 'Extended Warranties', 'Service Experience', 'Subscription Model'],
               rotation = 45, ha = 'right', rotation_mode = 'anchor')
    plt.ylabel('Sentiment')
    plt.title('Average Sentiment for each Topic')
    plt.ylim(-0.5, 0.5)
    plt.tight_layout()

    plt.savefig('images/reddit_topics.png')

if __name__ == '__main__':
    #sentiment_analysis()
    #topics()
    sentiment_text()
    #sentiment_visualization()