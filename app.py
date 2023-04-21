from flask import Flask, render_template, request, url_for
from PIL import Image
import pandas as pd
import numpy as np
import tweepy
import warnings
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tweetnlp
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
import contractions
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import time
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def home():
    # Render the home template
    return render_template('home2.html')

def clear_variables():
    # Clear all variables
    for name in dir():
        if not name.startswith('_'):
            del globals()[name]
    
@app.route('/result', methods=['POST'])
def result():
    # Get user input from the form
    try:
        topic = request.form['topic']
    except:
        # If the user input is invalid, render the error template
        return render_template('error.html', message='Invalid input')
    
    # Create a Pandas DataFrame from the list of tweets
    tweets_df = pd.read_csv('TwitterExtractFull.csv',encoding= 'MacRoman')
    tweets_df = tweets_df[tweets_df['Search Criteria'] == topic].copy()
    max_tweets = len(tweets_df)
    # Check if there are any tweets
    if tweets_df.empty:
        # If there are no tweets, render the error template
        return render_template('error.html', message='No tweets found')
        
    #Change date format in python pandas yyyy-mm-dd to dd-mm-yyyy
    tweets_df['Date'] = pd.to_datetime(tweets_df['Date'], format='%Y-%m-%d')
    tweets_df['Date'] = pd.to_datetime(tweets_df['Date']).dt.strftime('%d-%m-%Y')

    tweets_df['TextClean'] = tweets_df['Text']
    
    #Remove user handles from tweets
    tweets_df['TextClean'] = tweets_df['TextClean'].str.replace('(\@\w+.*?)',"",regex=True)
    
    # remove '\n', lowercase all letters
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: x.replace('\n',' ').lower())

    # expand contractions
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: contractions.fix(x))

    # remove punctuations
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: re.sub(r'[^\w\s]','',x))

    #remove HTML tags
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: re.sub(re.compile('<.*?>'), '', x))

    #remove consecutive characters that occur three or more times in a row, and replace them with just two occurrences of that character.
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: re.compile(r"(.)\1{2,}").sub(r"\1\1", x))

    # Removing extra spaces
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: re.sub(' +',' ',x))

    #lemmatize the tweets
    lemmatizer = WordNetLemmatizer()
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    
    # Removing words less than 2 characters long
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
    
    # Load the model
    model = tweetnlp.load_model('sentiment')

    # Define a function to apply sentiment analysis to each tweet text
    def analyze_sentiment(text):
        result = model.sentiment(text, return_probability=True)
        max_prob_key = max(result['probability'], key=result['probability'].get)
        return pd.Series({'Tweetsentiment': result['label'], 
                          'TweetProbability': result['probability'][max_prob_key]})

    # Apply the function to the 'TextClean' column and store the result in two new columns
    tweets_df[['Tweetsentiment', 'TweetProbability']] = tweets_df['TextClean'].apply(analyze_sentiment)
    
    #Filter out tweets where the sentiment classification probability is less than 0.5
    #tweets_df = tweets_df.loc[tweets_df['TweetProbability'] >= 0.5]
    
    def plot_sentiment_class(tweets_df):
        fig, ax = plt.subplots(figsize=(8,6))
        sns.countplot(y='Tweetsentiment', data=tweets_df, color='darkblue', ax=ax)
        ax.set_title('Sentiment Classification')
        # Save the figure as a variable
        SentimentClass = ax.get_figure()
        # save it as an image
        SentimentClass.savefig('static/SentimentClass.png')
        plt.close(SentimentClass)
    
    #Sentiment Classification Plot
    plot_sentiment_class(tweets_df)

    #Get the overall sentiment for the period
    OverallSentiment = tweets_df['Tweetsentiment'].mode()[0]
    
    # Group the dataframe by date and sentiment class and count the number of tweets in each group
    tweet_counts = tweets_df.groupby(['Date', 'Tweetsentiment']).size().unstack(fill_value=0)

    # Sort the DataFrame by the Datetime column
    #tweet_counts = tweet_counts.sort_values(by='Date')

    def plot_sentiment_over_time(tweet_counts):
        # Plot the line graph
        fig, ax = plt.subplots(figsize=(8,6))
        tweet_counts.plot(ax=ax)

        # Add titles and labels
        ax.set_title('Tweet Sentiment over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Tweets')
        ax.legend(title='Sentiment Class', loc='upper left')

        # Save the figure and close the plot
        Sentimentovertime = ax.get_figure()
        Sentimentovertime.savefig('static/Sentimentovertime.png')
        plt.close(Sentimentovertime)
    
    # Plot the line graph
    plot_sentiment_over_time(tweet_counts)

    topic = topic.replace(',', ', ')
    topic = topic.replace('  ', ' ')
    topic = topic.replace('  ', ' ')

    def generate_wordcloud(sentiment, tweets_df, topic):
        fig = plt.figure()
        if sentiment == 'negative':
            title = "Negative Tweets - Wordcloud"
            data = tweets_df['TextClean'][tweets_df['Tweetsentiment'] == 'negative']
            filename = 'static/Nwordcloud.png'
        elif sentiment == 'positive':
            title = "Positive Tweets - Wordcloud"
            data = tweets_df['TextClean'][tweets_df['Tweetsentiment'] == 'positive']
            filename = 'static/Pwordcloud.png'
        else:
            raise ValueError('Sentiment must be "positive" or "negative".')

        stopwords = ['TextClean', 'dtype', 'Name', 'object', 'Series'] + (topic.lower()).split(', ') + list(STOPWORDS)
        wordcloud = WordCloud(background_color="white", stopwords=stopwords).generate(str(data))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(title)
        plt.axis("off")
        fig.savefig(filename)
        plt.close(fig)
        
    # Wordcloud with Negative tweets
    generate_wordcloud('negative', tweets_df, topic)
    generate_wordcloud('positive', tweets_df, topic)


    top_positive_tweets = tweets_df.loc[tweets_df['Tweetsentiment'] == 'positive'].sort_values(by=['TweetProbability'], ascending=False).loc[:, ['Date', 'Text', 'Views']].head(5)

    top_negative_tweets = tweets_df.loc[tweets_df['Tweetsentiment'] == 'negative'].sort_values(by=['TweetProbability'], ascending=False).loc[:, ['Date', 'Text', 'Views']].head(5)

    topic = topic.title()
    max_tweets = "{:,}".format(max_tweets)
    fdate = tweets_df['Date'].iloc[0]
    ldate = tweets_df['Date'].iloc[-1]
    maxpositive = tweet_counts['positive'].idxmax()
    maxnegative = tweet_counts['negative'].idxmax()
    total_tweets = len(tweets_df)
    sentiment_counts = tweets_df['Tweetsentiment'].value_counts()
    percent_positive = round((sentiment_counts['positive'] / total_tweets) * 100, 2)
    percent_negative = round((sentiment_counts['negative'] / total_tweets) * 100, 2)
    users = str(tweets_df['Username'].nunique())
    tweets_df = tweets_df.drop(['Username', 'Search Criteria', 'TextClean'], axis=1)
    tweets_df.to_csv('tweets.csv', index=False)
    plt.close('all')
    
    # Render the results template with the DataFrame as a parameter        
    return render_template('result.html', topic=topic, tweets_df=tweets_df, OverallSentiment=OverallSentiment, fdate=fdate, ldate=ldate, max_tweets=max_tweets, maxpositive=maxpositive, maxnegative=maxnegative, users=users, percent_negative=percent_negative, percent_positive=percent_positive, top_positive_tweets=top_positive_tweets.to_html(index=False), top_negative_tweets=top_negative_tweets.to_html(index=False))

@app.route('/tweets_table')
def tweets_table():
    # Render the tweets_table template
    tweets_df = pd.read_csv('tweets.csv',encoding= 'MacRoman')
    return render_template('tweets_table.html', tweets_df=tweets_df.to_html(index=False))

# Clear variables
clear_variables()

if __name__ == '__main__':
    app.run(debug=True)