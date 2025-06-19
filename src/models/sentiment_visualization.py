import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.dates as mdates
import statsmodels.api as sm
import numpy as np
import shared_functions
import pandas as pd
import argparse

path_to_processed = 'data/processed'

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
    monthly_data.sort_values(['topic', 'year_month'], inplace = True)

    topics[1], topics[2], topics[3] = topics[3], topics[1], topics[2]
    monthly_data['topic'] = monthly_data['topic'].replace({1 : 3, 2 : 1, 3 : 2})
    monthly_data_filtered = monthly_data[monthly_data['year_month'] >= '2015-01-01'].copy()

    topic = sorted(monthly_data['topic'].unique())
    colors = ['#006868ff', '#008181ff', '#0193a0ff', '#01d1d1ff', '#1bd9c5ff']

    test_results = []

    for top in topic:

        temp = monthly_data_filtered[monthly_data_filtered['topic'] == top]

        y = np.array(temp['sentiment'])
        t = np.arange(len(y))

        X = sm.add_constant(t)
        model = sm.OLS(y, X).fit()

        slope = round(model.params[1], 4)
        p_value = round(model.pvalues[1], 4)

        test_results.append([slope, p_value])

    fig, ax = plt.subplots(2, 2, figsize = (25, 10))

    for index, axes in enumerate(ax.flat):

        temp = monthly_data[monthly_data['topic'] == topic[index]].copy()
        x_coord = mdates.date2num(temp['year_month'].iloc[0])

        temp['year_month_num'] = temp['year_month'].map(pd.Timestamp.toordinal)
        smoothed = sm.nonparametric.lowess(temp['sentiment'], temp['year_month_num'], frac = 0.1)

        #plt.plot(temp['year_month'], temp['sentiment'], label = 'Original', color = colors[index])
        axes.plot(pd.to_datetime([pd.Timestamp.fromordinal(int(x)) for x in smoothed[:, 0]]),
                smoothed[:, 1], label = 'Lowess Smoothed', color = colors[index])

        #axes.plot(temp['year_month'], temp['sentiment'], marker = 'o', color = colors[index])
        axes.axhline(y = 0, color = 'black', linestyle = '--', linewidth = 1)
        axes.text(x_coord, -0.55, f"Slope: {test_results[index][0]}", fontsize = 16, fontname = 'Arial')
        axes.text(x_coord, -0.7, f"p-value: {test_results[index][1]}", fontsize = 16, fontname = 'Arial')
        axes.text(x_coord, -0.85, f"Δ Sentiment: {round(test_results[index][0] * len(monthly_data_filtered)/4, 4)}", fontsize = 16, fontname = 'Arial'),       
        axes.set_title(f"{topics[index]}", fontname = 'Arial', fontsize = 16)
        axes.set_ylim((-1.05, 1.05))
        axes.xaxis.set_major_locator(mdates.YearLocator())
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig('images/sentiment_over_time.png')

def token_count():

    counts = shared_functions.safe_loader(f"{path_to_processed}/comment_counts.pkl")

    google = mpimg.imread('images/google_logo.png')
    meta = mpimg.imread('images/meta_logo.png')
    reddit = mpimg.imread('images/reddit_logo.png')

    images = [google, meta, reddit]
    brand_colors = ["#FF0000", "#1877F2", "#FF4500"]

    fig, ax = plt.subplots()
    bars = ax.barh(counts['source'], counts['comments'], color = brand_colors)

    for index, bar in enumerate(bars):

        bar_height = bar.get_height()
        y = bar.get_y() + bar_height / 2
        x = bar.get_width()

        imagebox = OffsetImage(images[index], zoom = 0.3)
        ab = AnnotationBbox(imagebox, (x, y),
                            xybox = (30, 0),
                            frameon = False,
                            boxcoords = "offset points",
                            pad = 0)

        ax.add_artist(ab)

    ax.set_xlim(0, max(counts['comments']) * 1.2)
    plt.yticks([])
    plt.title('Data Counts for each Source')
    plt.tight_layout()
    plt.savefig('images/counts_bar_plot.png')

def scores():

    scores = shared_functions.safe_loader(f"{path_to_processed}/pain_points_drivers_scores.pkl")

    topics = list(scores.keys())
    negative_topics = []
    negative_scores = []
    positive_topics = []
    positive_scores = []

    for topic in topics:
        negative_topics.append(list(scores[topic]['Pain Points'].keys()))
        negative_scores.append(list(scores[topic]['Pain Points'].values()))
        positive_topics.append(list(scores[topic]['Drivers'].keys()))
        positive_scores.append(list(scores[topic]['Drivers'].values()))

    fig, ax = plt.subplots(2, 4, figsize = (30, 20))

    for index, axes in enumerate(ax.flat):

        plt.setp(axes.get_xticklabels(), fontfamily = 'Arial', fontweight = 'bold', fontsize = 15)

        if index < 4:
            i = index
            topic = shared_functions.split_title(topics[i])
            bars = axes.bar(positive_topics[i], positive_scores[i], color = '#3b4cc0')

            for bar in bars:
                height = bar.get_height()
                axes.text(bar.get_x() + bar.get_width()/2, height - 0.2, 
                         f'{round(height, 2)}', ha = 'center', va = 'bottom',
                         fontweight = 'bold', color = 'white')

            axes.set_xticks(positive_topics[i])
            axes.set_xticklabels(positive_topics[i], rotation = 45, ha = 'right')
            axes.set_title(f"{topic}\nDrivers", fontfamily = 'Arial', fontweight = 'bold', fontsize = 15)
            axes.set_ylim((10, 15))
        else:
            i = index - 4
            bars = axes.bar(negative_topics[i], negative_scores[i], color = '#b40426')

            for bar in bars:
                height = bar.get_height()
                axes.text(bar.get_x() + bar.get_width()/2, height - 0.2, 
                         f'{round(height, 2)}', ha = 'center', va = 'bottom',
                         fontweight = 'bold', color = 'white')

            axes.set_xticks(negative_topics[i])
            axes.set_xticklabels(negative_topics[i], rotation = 45, ha = 'right')
            axes.set_title(f"Pain Points", fontfamily = 'Arial', fontweight = 'bold', fontsize = 15)
            axes.set_ylim((10, 15))           

    plt.tight_layout()
    plt.savefig('images/pain_points_drivers_scores.png')

def fuzzy():

    import pandas as pd
    from rapidfuzz import fuzz, process

    df = pd.read_csv(f"{path_to_processed}/final_cleaned_data.csv")
    df['text'] = df['text'].astype(str)
    text_list = df['text'].tolist()

    examples_df = pd.read_excel(f"{path_to_processed}/Sentence_Match_Results.xlsx")
    sentences = examples_df.iloc[:, 0].astype(str).tolist()

    results = []

    for index, sent in enumerate(sentences):

        if not index % 100:
            print(f"Starting on row {index}.")
        
        sent_clean = sent.strip()

        exact_indices = df[df['text'].str.lower() == sent_clean.lower()].index.tolist()

        fuzzy_matches = process.extract(
            sent_clean, text_list, scorer = fuzz.partial_ratio,
            score_cutoff = 70, limit = 5
        )
        fuzzy_indices = [match[2] for match in fuzzy_matches if match[2] not in exact_indices]

        results.append({
            "Example Sentence": sent_clean,
            "Exact Match Rows": ', '.join(map(str, exact_indices)),
            "Close Match Rows (Fuzzy)": ', '.join(map(str, fuzzy_indices))
        })

    # Save to Excel
    out_df = pd.DataFrame(results)
    out_df.to_excel(f"{path_to_processed}/Sentence_Match_With_RapidFuzz.xlsx", index = False)


def check_row(row):
    
    data = pd.read_parquet(f"{path_to_processed}/final_cleaned_data.parquet")

    print(data['text'].iloc[row])

def arg_parse(function, row):

    match function:
        case 'vis':
            sentiment_visualization()
        case 'map':
            sentiment_world_map()
        case 'time':
            time_sentiment()
        case 'count':
            token_count()
        case 'score':
            scores()
        case 'check':
            row = int(row)
            check_row(row)
        case 'fuzz':
            fuzzy()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Just a simple argument parser.')
    parser.add_argument('--function', required = True, help = 'Choose either vis, map, time, count, or score.')
    parser.add_argument('--row', default = None, required = False)
    
    arg = parser.parse_args()

    arg_parse(function = arg.function, row = arg.row)