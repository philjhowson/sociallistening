# Social Listening Project

**As a note, I will not explain the overall theme of the project
or the specific results because this was completed for a client.
As such, only the flow of the project and the methods used
are discussed.**

This project uses social listening methods, including publicly
available APIs (Google and Reddit APIs) and playwright to automate
web scraping of social media platforms to better understand
customer sentiment towards the broader topic of business models
for extended services for electronic devices.

To complete this project, I build an end-to-end data ingestion
pipeline, including sentiment analysis and theme discovering.
I further build a RAG model to extra key insights, i.e. pain points
and drivers, to provide the client with actionable insights.

## Project Organization
------------------------------------------------------------------------
    root
    ├── data # not uploaded for coherance with DSGVO and GDPR regulations
    │   ├── processed # processed data files
    │   ├── RAG # contains output files from RAG queries and FAISS archives    
    │   └── raw # the raw extracted social media content
    ├── images # images used to evaluate and visualize results, some anonymized images are presented
    ├── src # contains source code for exploration, not stored on GitHub due to size limitations
    │   ├── data # code for data ingestion
    │   │    ├── create_train_test_split.py # creates the training, test, and validation indicies
    │   │    ├── browser_scraping_functions.py # functions used by playwright
    │   │    ├── reddit_scraper.py # script for scraping Reddit
    │   │    ├── shared_functions.py # functions used in multiple scraping scripts
    │   │    ├── threads_get_login.py # script to save auth_state for Threads
    │   │    ├── threads_retrieve_ids.py # script used to retrieve post ids from Threads searches
    │   │    ├── threads_scraper.py # script used to scrape Threads posts
    │   │    └── youtube_scraper.py # script used to scrape YouTube
    │   ├── formatting
    │   │    ├── data_formatting_pipeline.py # functions used for data formatting
    │   │    ├── format_data.py # formatting data, creating embeddings, and filtering data
    │   │    └── shared_functions.py # general functions for data formatting
    │   ├── models
    │   │    ├── scoring_functions.py # finds frequencies of similar comments and generates scores
    │   │    ├── sentiment_analysis.py # analyzes sentiment and finds themes
    │   │    ├── sentiment_visualization.py # visualizes sentiment
    │   │    └── shared_functions.py # general functions used by scripts
    │   └── RAG
    │        ├── build_archive.py # builds the FAISS archive
    │        ├── rag_functions.py # functions used by the RAG model
    │        ├── run_rag.py # builds prompts and queries the database
    │        └── shared_functions.py # general functions for data formatting
    ├── .env
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt

## Project Introduction

The goal of this project is to develop actionable insights from customer sentiment
about device protection and extended services.

Data were collected from three sources, Reddit, Threads, and YouTube. In the case
of Reddit and YouTube, they both have fairly reasonable APIs that can be used for
large scale data collection. Threads, on the other hand, offers no usable API that
permits users to access the search function and scrape relevant posts. Therefore,
I created a Threads scraper that utilizes playwright to scrape post ids and scrape
post and reply content. Because Threads generally discourages bot use, I developed
a scraper that mimics human behaviour by inserting random pauses, hovering over
random links, and opening random user profiles and images. The Threads scraper
functions with good success and can generally run for hours without interruption.

## Project Workflow

I first collected data from my sources and then proceeded to clean text to make it
suitable for text embeddings. This followed a variety of best practices, including
removing punctuation and html tags, lemmatizing and standardizing text, lower casing,
removing stop words, and other processes. Additionally, because this was a global
scale project, it was necessary to identify comment languages to use appropriate stop
word removal and lemmatization. I collected data from over 50 languages.

