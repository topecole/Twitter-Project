from flask import Flask, render_template, request, url_for
from PIL import Image
import pandas as pd
import numpy as np
import tweepy
import warnings
import snscrape.modules.twitter as sntwitter
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
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS

app = Flask(__name__)

@app.route('/')
def home():
    # Render the home template
    return render_template('home.html')

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

    # Create query for snscrape
    query = f'({topic}) near:"{location}" lang:en until:{end_date} since:{start_date} -filter:links -filter:retweet'

    # Create empty list to store tweets
    tweets_list = []

    # Use snscrape to scrape tweets
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i > max_tweets:
            break
        tweets_list.append([tweet.date, tweet.rawContent, tweet.user.username, tweet.viewCount])
        
    # Create a Pandas DataFrame from the list of tweets
    tweets_df = pd.DataFrame(tweets_list, columns=['Date', 'Text', 'Username', 'Views'])
    tweets_df['Date'] = tweets_df['Date'].dt.date
    tweets_df =tweets_df.sort_values(by=['Views'], ascending=False)
    
    # Check if there are any tweets
    if tweets_df.empty:
        # If there are no tweets, render the error template
        return render_template('error.html', message='No tweets found')

    tweets_df['TextClean'] = tweets_df['Text']

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

    # Removing stop words
    stop_words = set(stopwords.words('english'))
    tweets_df['TextClean'] = tweets_df['TextClean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Load the model
    model = tweetnlp.load_model('sentiment')

    # Define the sentiment analysis function
    def get_sentiment(text):
        return model.sentiment(text)['label']

    # Apply the function to each row of the 'filtertweet[textclean]' column and store the result in a new column called 'sentiment'
    tweets_df['Tweetsentiment'] = tweets_df['TextClean'].apply(get_sentiment)

    #Sentiment Classification Plot
    fig, ax = plt.subplots(figsize=(8,6))
    sns.countplot(y='Tweetsentiment', data=tweets_df, color='darkblue', ax=ax)
    ax.set_title('Sentiment Classification')
    # Save the figure as a variable
    SentimentClass = ax.get_figure()
    # save it as an image
    SentimentClass.savefig('static/SentimentClass.png')

    #Get the overall sentiment for the period
    OverallSentiment = tweets_df['Tweetsentiment'].mode()[0]

    # Group the dataframe by date and sentiment class and count the number of tweets in each group
    tweet_counts = tweets_df.groupby(['Date', 'Tweetsentiment']).size().unstack(fill_value=0)

    # Plot the line graph
    fig, ax = plt.subplots(figsize=(10, 6))
    tweet_counts.plot(ax=ax)

    # Add titles and labels
    ax.set_title('Tweet Sentiment over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Tweets')
    ax.legend(title='Sentiment Class', loc='upper left')
    Sentimentovertime = ax.get_figure()
    Sentimentovertime.savefig('static/Sentimentovertime.png')

    # Wordcloud with Negative tweets
    NegativeWC = plt.figure()
    plt.title("Negative Tweets - Wordcloud")
    plt.imshow(WordCloud(width=700, height=400,max_font_size=80, max_words=50, background_color="white", stopwords=([topic.lower()] + list(STOPWORDS))).generate(str(tweets_df['TextClean'][tweets_df['Tweetsentiment'] == 'negative'])), interpolation="bilinear")
    plt.axis("off")

    # Display the figure using plt.show()
    #plt.show()
    NegativeWC.savefig('static/Nwordcloud.png')
    plt.close()

    PositiveWC = plt.figure()
    plt.title("Positive Tweets - Wordcloud")
    plt.imshow(WordCloud(width=700, height=400,max_font_size=80, max_words=50, background_color="white", stopwords=([topic.lower()] + list(STOPWORDS))).generate(str(tweets_df['TextClean'][tweets_df['Tweetsentiment'] == 'positive'])), interpolation="bilinear")
    plt.axis("off")

    # Display the figure using plt.show()
    #plt.show()
    PositiveWC.savefig('static/Pwordcloud.png')
    plt.close()

    top_positive_tweets = tweets_df.loc[tweets_df['Tweetsentiment'] == 'positive'].sort_values(by=['Views'], ascending=False).loc[:, ['Date', 'Text', 'Views']].head(3)

    top_negative_tweets = tweets_df.loc[tweets_df['Tweetsentiment'] == 'negative'].sort_values(by=['Views'], ascending=False).loc[:, ['Date', 'Text', 'Views']].head(3)
    
    location = location.title()

    # Render the results template with the DataFrame as a parameter        
    return render_template('result.html', topic=topic, location=location, OverallSentiment=OverallSentiment, top_positive_tweets=top_positive_tweets.to_html(index=False), top_negative_tweets=top_negative_tweets.to_html(index=False))

if __name__ == '__main__':
    app.run(debug=True)