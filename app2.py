from flask import Flask, render_template, request, url_for
from PIL import Image
import snscrape.modules.twitter as sntwitter
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
import contractions
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import time
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from datetime import datetime, timedelta

# set up the Flask web application 
app = Flask(__name__)

@app.route('/')

# define the home function
def home():
    # Render the home template
    return render_template('home.html')

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
        location = request.form['location']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        max_tweets = int(request.form['max_tweets'])
    except:
        # If the user input is invalid, render the error template
        return render_template('error.html', message='Invalid input')

    topic2 = topic.replace(',', " OR ")
    topic2 = topic2.replace('  ', ' ')
    
    # Convert strings to dates
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate period duration
    period_duration = (end_date - start_date) / 4

    # Create list of start and end dates for each period
    period_dates = []
    for i in range(4):
        period_start = start_date + i * period_duration
        period_end = period_start + period_duration - timedelta(days=1)
        period_dates.append((period_start, period_end))

    # Create query and scrape tweets for each period
    tweets_list = []
    Qt_tweets = int(max_tweets / 4)
    try:
        for period_start, period_end in period_dates:
            query = f'{topic}) near:"{location}" within:300km lang:en since:{period_start.strftime("%Y-%m-%d")} until:{period_end.strftime("%Y-%m-%d")} -filter:links -filter:retweet'
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                time.sleep(1) # add a time delay of 1 second
                if i > Qt_tweets:
                    break
                tweets_list.append([tweet.date, tweet.rawContent, tweet.user.username, tweet.viewCount])
        # Create a Pandas DataFrame from the list of tweets
        tweets_df = pd.DataFrame(tweets_list, columns=['Date', 'Text', 'Username', 'Views'])

        # Check if there are any tweets
        if tweets_df.empty:
            # If there are no tweets, render the error template
            return render_template('error.html', message='No tweets found')
    except:
        return render_template('error.html', message='Its not you, its us. the Twitter Scraper API is currently down. Please try later')

    # #Change date format in python pandas yyyy-mm-dd to dd-mm-yyyy
    # tweets_df['Date'] = pd.to_datetime(tweets_df['Date']).dt.strftime('%d-%m-%Y')

    tweets_df['TextClean'] = tweets_df['Text']

    # remove '\n', lowercase all letters
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: x.replace('\n',' ').lower())

    # Load the model
    model = tweetnlp.load_model('sentiment')

    # Define a function to apply sentiment analysis to each tweet text
    def analyze_sentiment(text):
        result = model.sentiment(text, return_probability=True)
        max_prob_key = max(result['probability'], key=result['probability'].get)
        return pd.Series({'Tweetsentiment': result['label'], 
                          'TweetProbability': result['probability'][max_prob_key]})

    # Apply the function to the 'Text' column and store the result in two new columns
    tweets_df[['Tweetsentiment', 'TweetProbability']] = tweets_df['TextClean'].apply(analyze_sentiment)

    #Remove user handles from tweets
    tweets_df['TextClean'] = tweets_df['TextClean'].str.replace('(\@\w+.*?)',"",regex=False)

    # expand contractions
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: contractions.fix(x))

    # remove punctuations
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: re.sub(r'[^\w\s]','',x))

    #remove consecutive characters that occur three or more times in a row, and replace them with just two occurrences of that character.
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: re.compile(r"(.)\1{2,}").sub(r"\1\1", x))

    # Removing extra spaces
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: re.sub(' +',' ',x))
    
    # # Removing stop words
    # stop_words = set(stopwords.words('english'))
    # tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    
    # Removing words less than 2 characters long
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: ' '.join([word for word in x.split() if len(word) >= 2]))
    
    #Filter out tweets where the sentiment classification probability is less than 0.5
    tweets_df = tweets_df.loc[tweets_df['TweetProbability'] > 0.5]
    
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

    tweets_df = tweets_df.sort_values(by='Date', ascending=True)
    
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

        stopwords = [topic,'TextClean', 'dtype', 'Name', 'object', 'Series'] + (topic.lower()).split(' ') + list(STOPWORDS)
        wordcloud = WordCloud(background_color="white", stopwords=stopwords).generate(str(data))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(title)
        plt.axis("off")

        # Get top 5 words from the wordcloud
        word_frequencies = wordcloud.process_text(str(data))
        word_frequencies_sorted = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_words = pd.DataFrame(word_frequencies_sorted[:5], columns=['Word', 'Frequency'])

        fig = plt.gcf()
        fig.savefig(filename)
        plt.close(fig)

        return top_words

    # Wordcloud with Negative tweets
    neg_top5 = generate_wordcloud('negative', tweets_df, topic)
    pos_top5 = generate_wordcloud('positive', tweets_df, topic)
    
    tweets_df['Date'] = tweets_df['Date'].dt.date

    top_positive_tweets = tweets_df.loc[tweets_df['Tweetsentiment'] == 'positive'].sort_values(by=['TweetProbability'], ascending=False).loc[:, ['Date', 'Text', 'Views']].head(5)

    top_negative_tweets = tweets_df.loc[tweets_df['Tweetsentiment'] == 'negative'].sort_values(by=['TweetProbability'], ascending=False).loc[:, ['Date', 'Text', 'Views']].head(5)

    topic = topic.title()
    location = location.title()
    max_tweets = len(tweets_df['Text'])
    max_tweets = "{:,}".format(max_tweets)
    fdate = tweets_df['Date'].iloc[0].strftime('%d %B %Y')
    ldate = tweets_df['Date'].iloc[-1].strftime('%d %B %Y')
    maxpositive = (tweet_counts['positive'].idxmax()).strftime('%d %B %Y')
    maxnegative = (tweet_counts['negative'].idxmax()).strftime('%d %B %Y')
    total_tweets = len(tweets_df)
    sentiment_counts = tweets_df['Tweetsentiment'].value_counts()
    percent_positive = round((sentiment_counts['positive'] / total_tweets) * 100, 2)
    percent_negative = round((sentiment_counts['negative'] / total_tweets) * 100, 2)
    users = str(tweets_df['Username'].nunique())
    tweets_df = tweets_df.drop(['Username', 'TextClean'], axis=1)
    tweets_df.to_csv('tweets.csv', index=False)
    plt.close('all')
    
    # Render the results template with the DataFrame as a parameter        
    return render_template('result.html', topic=topic, tweets_df=tweets_df, OverallSentiment=OverallSentiment, fdate=fdate, ldate=ldate, max_tweets=max_tweets, maxpositive=maxpositive, maxnegative=maxnegative, location = location, users=users, percent_negative=percent_negative, percent_positive=percent_positive, neg_top5=neg_top5.to_html(index=False), pos_top5=pos_top5.to_html(index=False), top_positive_tweets=top_positive_tweets.to_html(index=False), top_negative_tweets=top_negative_tweets.to_html(index=False))

@app.route('/tweets_table')
def tweets_table():
    # Render the tweets_table template
    tweets_df = pd.read_csv('tweets.csv',encoding= 'MacRoman')
    tweets_df['Date'] = pd.to_datetime(tweets_df['Date'])
    tweets_df = tweets_df.sort_values(by='Date', ascending=False)
    return render_template('tweets_table.html', tweets_df=tweets_df.to_html(index=False))

# Clear variables
clear_variables()

if __name__ == '__main__':
    app.run(debug=True)