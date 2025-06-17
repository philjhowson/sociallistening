import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bertopic import BERTopic
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
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

def sentiment_visualization():

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment.parquet")
    topic_names = shared_functions.safe_loader(f"{path_to_processed}/topic_names.pkl")
    warranties = data[data['topic_label'] != -1]
    topics = warranties.groupby('topic_label').agg({'sentiment_label' : 'mean'})

    fig = plt.figure(figsize = (10, 10))
    bars = plt.bar(topics.index, topics['sentiment_label'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,  height + 0.01,
            f'{height:.3f}', ha = 'center', va = 'bottom', color = 'black',
            fontweight = 'bold')
    plt.xticks([1, 2, 3, 4, 5], topic_names,
               rotation = 45, ha = 'right', rotation_mode = 'anchor')
    plt.ylabel('Sentiment')
    plt.title('Average Sentiment for each Topic')
    plt.ylim(-0.5, 0.5)
    plt.tight_layout()

    plt.savefig('images/sentiment_analysis_overview.png')

def sentiment_world_map():

    data = pd.read_parquet('data/processed/cleaned_masterdata_sentiment.parquet')
    data = data[data['topic_label'] != -1]
    world = gpd.read_file('images/maps/ne_110m_admin_0_countries.shp')
    map_conversions = shared_functions.safe_loader('data/processed/map_conversions.pkl')
    world['region'] = world['NAME'].map(map_conversions)
    topics = shared_functions.safe_loader(f"{path_to_processed}/topic_names.pkl")
    topic_ids = sorted(data['topic_label'].unique())

    for index, topic_id in enumerate(topic_ids):
        df_topic = data[data['topic_label'] == topic_id]
        agg = df_topic.groupby('region')['sentiment_label'].mean().reset_index()

        world_sentiment = world.merge(agg, how = 'left', left_on = 'region', right_on = 'region')
        fig, ax = plt.subplots(figsize = (15, 10))
        world_sentiment.plot(
            column = 'sentiment_label',
            cmap = 'coolwarm_r',
            vmin = -1, vmax = 1, 
            legend = True,
            legend_kwds = {'shrink':.5},
            ax = ax,
            edgecolor = 'black',
            missing_kwds = {
                "color": "lightgrey",
                "edgecolor": "white",
                "hatch": "///",
                "label": "No data"
            }
        )
        
        ax.margins(0)
        plt.title(f"{topics[index]}", fontsize = 16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'images/world_sentiment_{topic_id}.png', bbox_inches = 'tight', pad_inches = 0.05)

    aggregated_sentiment = data.groupby('topic_label').agg({'sentiment_label' : 'mean'})

    fig = plt.figure(figsize = (10, 10))

    bars = plt.bar(aggregated_sentiment.index, aggregated_sentiment['sentiment_label'])

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2, height + 0.005, 
            f'{round(height, 2)}', ha = 'center', va = 'bottom',
            fontweight = 'bold'
        )

    plt.xticks(aggregated_sentiment.index, topics, rotation = 45, ha = 'right', va = 'top')
    plt.ylabel('Sentiment')
    plt.title('Sentment Overview')
    plt.tight_layout()
    plt.savefig('images/sentiment_overview.png')

def time_sentiment():

    data = pd.read_parquet(f"{path_to_processed}/final_cleaned_data.parquet")
    topics = shared_functions.safe_loader(f"{path_to_processed}/topic_names.pkl")
    data['year_month'] = data['date'].dt.to_period('M').dt.to_timestamp()

    monthly_data = data.groupby(['topic', 'year_month']).agg({'sentiment' : 'mean'}).reset_index()
    colors = ['#006868ff', '#008181ff', '#0193a0ff', '#01d1d1ff', '#1bd9c5ff']

    for index, topic in enumerate(sorted(monthly_data['topic'].unique())):

        temp = monthly_data[monthly_data['topic'] == topic]
        plt.plot(temp['year_month'], temp['sentiment'], marker = 'o', label = topics[index], color = colors[index])

    plt.title('Sentiment Over Time for Each Topic')
    plt.ylim((-1, 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/sentiment_over_time.png')


def arg_parse(function):

    match function:
        case 'analysis':
            sentiment_analysis()
        case 'topic':
            topics()
        case 'text':
            sentiment_text()
        case 'vis':
            sentiment_visualization()
        case 'map':
            sentiment_world_map()
        case 'time':
            time_sentiment()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Just a simple argument parser.')
    parser.add_argument('--function', required = True, help = 'Choose either analysis, topic, text, vis, or map.')
    
    arg = parser.parse_args()

    arg_parse(function = arg.function)