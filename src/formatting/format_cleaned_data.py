import pandas as pd
import shared_functions
import matplotlib.pyplot as plt
import geopandas as gpd
import shared_functions

def format_results():

    youtube = pd.read_parquet('data/processed/topics_sentiment_youtube_results.parquet')
    country_to_region = shared_functions.safe_loader('data/processed/youtube_countries_to_region.pkl')
    youtube['region'] = youtube['geolocation'].map(country_to_region).fillna('Unknown')
    columns_to_keep = ['video_title', 'hashtags', 'comment_text', 'comment_time',
                       'comment_likes', 'cleaned_comment_text', 'region',
                       'topic', 'sentiment', 'source']
    youtube = youtube[columns_to_keep]
    renamed_columns = {'video_title' : 'title', 'comment_text' : 'text', 'comment_time' : 'date',
                       'comment_likes' : 'likes', 'cleaned_comment_text' : 'cleaned_text'}
    youtube.rename(renamed_columns, axis = 1, inplace = True)

    reddit = pd.read_parquet('data/processed/topics_sentiment_reddit_results.parquet')
 
    country_to_region = shared_functions.safe_loader('data/processed/reddit_countries_to_region.pkl')
    reddit['region'] = reddit['subreddit'].map(country_to_region).fillna('Unknown')
    reddit['hashtag'] = [[] for i in range(len(reddit))]
    columns_to_keep = ['post_title', 'hashtag', 'comment_body', 'comment_created_utc',
                       'comment_score', 'cleaned_comment_body', 'region', 'topic',
                       'sentiment', 'source']
    reddit = reddit[columns_to_keep]
    renamed_columns = {'post_title' : 'title', 'comment_body' : 'text', 'comment_created_utc' : 'date',
                       'comment_score' : 'likes', 'cleaned_comment_body' : 'cleaned_text'}
    reddit.rename(renamed_columns, axis = 1, inplace = True)

    data = pd.concat([youtube, reddit], ignore_index = True)
    data = data[data['topic'] != -1]
    data.dropna(subset = 'topic', inplace = True)
    data.to_parquet('data/processed/final_cleaned_data.parquet')

def plot_results():

    data = pd.read_parquet('data/processed/final_cleaned_data.parquet')

    world = gpd.read_file('images/maps/ne_110m_admin_0_countries.shp')
    map_conversions = shared_functions.safe_loader('data/processed/map_conversions.pkl')
    world['region'] = world['NAME'].map(map_conversions)

    topics = ['General Experience with Extended Warranty Services',
              'Role of Experience in Customer Satisfaction and Brand Impressions',
              'Willingness to Pay (WTP)', 'General Sentiment on Extended Warranties']

    topic_ids = sorted(data['topic'].unique())

    for index, topic_id in enumerate(topic_ids):
        df_topic = data[data['topic'] == topic_id]
        agg = df_topic.groupby('region')['sentiment'].mean().reset_index()

        world_sentiment = world.merge(agg, how = 'left', left_on = 'region', right_on = 'region')
        fig, ax = plt.subplots(figsize = (15, 10))
        world_sentiment.plot(
            column = 'sentiment',
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


    aggregated_sentiment = data.groupby('topic').agg({'sentiment' : 'mean'})

    fig = plt.figure(figsize = (10, 10))

    bars = plt.bar(aggregated_sentiment.index, aggregated_sentiment['sentiment'])

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

if __name__ == '__main__':
    #format_results()
    plot_results()