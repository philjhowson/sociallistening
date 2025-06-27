import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.dates as mdates
import pymannkendall as mk
import statsmodels.api as sm
import numpy as np
import shared_functions
import pandas as pd
import argparse

path_to_processed = 'data/processed'

def sentiment_world_map():
    """
    This creates a world map with sentiment by topic. The world map is loaded and a column
    'region' is added to the world map by converting the country names into relevant
    regions using the .map() function. Loads in the topic names.
    """

    data = pd.read_parquet('data/processed/cleaned_masterdata_sentiment_topics.parquet')
    data = data[data['topic'] != -1]
    data = data[data['date'] >= '2020-01-01']
    world = gpd.read_file('images/maps/ne_110m_admin_0_countries.shp')
    map_conversions = shared_functions.safe_loader('data/processed/map_conversions.pkl')
    world['region'] = world['NAME'].map(map_conversions)
    topics = shared_functions.safe_loader(f"{path_to_processed}/topic_names.pkl")
    topic_ids = sorted(data['topic'].unique())

    """
    For each topic, the aggregated data is aggregated to a mean for that region, and
    then merged with the world map. The map is then plotted with hatching for regions
    with no data and reverse coolwarm mapping.
    """

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
        """
        Uses both margins(0) and tight_layout() to get a clean and compact image.
        """
        ax.margins(0)
        plt.title(f"{topics[index]}", fontsize = 16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'images/world_sentiment_{topic_id}.png', bbox_inches = 'tight', pad_inches = 0.05)

    """
    Creates an aggregated sentiment without regional considerations for each topic and
    plots a barplot for the sentiment of each topic. Adds font with the bar value to the
    bar.
    """

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

    topics = topics[0:4]

    plt.xticks(aggregated_sentiment.index, topics, rotation = 45, ha = 'right', va = 'top')
    plt.ylabel('Sentiment')
    plt.title('Sentment Overview')
    plt.tight_layout()
    plt.savefig('images/sentiment_overview.png')

def time_sentiment():
    """
    This function does a sentiment overtime and uses the Mann-Kendall test to determine
    if there is a trend in the data. First I ensure all the timestamps are correctly
    formatted, then aggregate mean sentiment by topic and year-month.
    """

    data = pd.read_parquet(f"{path_to_processed}/cleaned_masterdata_sentiment_topics.parquet")
    topics = shared_functions.safe_loader(f"{path_to_processed}/topic_names.pkl")
    data['year_month'] = data['date'].dt.to_period('M').dt.to_timestamp()

    monthly_data = data.groupby(['topic', 'year_month']).agg({'sentiment' : 'mean'}).reset_index()
    monthly_data.sort_values(['topic', 'year_month'], inplace = True)

    """
    This sets a cutoff date. I chose 2020 because more recent trends were more relevant
    rather than the overall trend since the beginning of the data. This subsets the data
    to that timeframe. It also changes the order here of the topics so that the plotting
    of the overall image is laid out in the order I required.
    """

    year = '2020'

    topics[1], topics[2], topics[3] = topics[3], topics[1], topics[2]
    monthly_data['topic'] = monthly_data['topic'].replace({1 : 3, 2 : 1, 3 : 2})
    monthly_data = monthly_data[monthly_data['year_month'] >= year + '-01-01'].copy()

    topic = sorted(monthly_data['topic'].unique())
    colors = ['#006868ff', '#008181ff', '#0193a0ff', '#01d1d1ff', '#1bd9c5ff']

    """
    Creates a subplot and creates a flat object for easy subplotting.
    """

    fig, ax = plt.subplots(2, 2, figsize = (20, 10))

    for index, axes in enumerate(ax.flat):

        for spine in ['top', 'right']:
            axes.spines[spine].set_visible(False)

        temp = monthly_data[monthly_data['topic'] == topic[index]].copy()
        """
        Converts the year-month format to a usable x_coordinate for plotting text.
        Then I do the Mann-Kendall test to analyze the trend. The overall trend
        is also smoothed, because the values are generally prone to sharp moves
        because the original values were just -1, 0, 1, so even with aggregation
        there can be a lot of sharp spikes. The lowess function makes it a bit easier
        to see the trends.
        """
        x_coord = mdates.date2num(temp['year_month'].iloc[0])

        result = mk.original_test(temp['sentiment'])

        temp['year_month_num'] = temp['year_month'].map(pd.Timestamp.toordinal)
        smoothed = sm.nonparametric.lowess(temp['sentiment'], temp['year_month_num'], frac = 0.1)

        axes.plot(pd.to_datetime([pd.Timestamp.fromordinal(int(x)) for x in smoothed[:, 0]]),
                smoothed[:, 1], label = 'Lowess Smoothed', color = colors[index])

        """
        I had to do this because on the first plot I wanted different y_coords to avoid
        overlapping with the smooth.
        """

        if index == 0:
            y_coords = [0.25, 0.175, 0.1]
        else:
            y_coords = [0.9, 0.825, 0.75]

        """
        Plots the mk results on the plot, namely, the slope and p-value. The slope was used
        to calculate the change in sentiment over the past 5 years.
        """

        axes.text(x_coord, y_coords[0], f"Slope: {round(result.slope, 4)}", fontsize = 16, fontname = 'Arial')
        axes.text(x_coord, y_coords[1], f"p-value: {round(result.p, 4)}", fontsize = 16, fontname = 'Arial')
        axes.text(x_coord, y_coords[2], f"Δ Sentiment: {round(result.slope * len(temp), 4)}", fontsize = 16, fontname = 'Arial'),       
        axes.set_title(f"{topics[index]}", fontname = 'Arial', fontsize = 16)
        axes.set_ylim((0, 1.05))
        axes.xaxis.set_major_locator(mdates.YearLocator())
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(f"images/sentiment_over_time_{year}.png")

def token_count():
    """
    This takes the saved .pkl file from the data_observing.py file and makes a simple
    bar chart to show how much data was collected from each source.
    """

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
    """
    This plots each of the scores for the identified pain points and drivers.
    """

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

def arg_parse(function):

    match function:
        case 'map':
            sentiment_world_map()
        case 'time':
            time_sentiment()
        case 'count':
            token_count()
        case 'score':
            scores()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Just a simple argument parser.')
    parser.add_argument('--function', required = True, help = 'Choose either map, time, count, or score.')
    
    arg = parser.parse_args()

    arg_parse(function = arg.function)