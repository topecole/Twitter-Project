from flask import Flask, render_template, request, url_for
import pandas as pd
import snscrape.modules.twitter as sntwitter
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re

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

    # Get the top 10 most viewed tweets
    tweets_df10 = tweets_df.head(10)
    
    # Generate word cloud
    w_tweets = tweets_df['Text'].str.replace('(\@\w+.*?)',"")
    w_tweets = w_tweets.str.replace('(\#\w+.*?)',"")
    stop_words = ["https", "co", "RT", topic.lower()] + list(STOPWORDS)
    wds_tweets = re.sub(r'\b\w{1,3}\b', '', str(w_tweets))
    tweets_wordcloud = WordCloud(width=700, height=400,max_font_size=40, max_words=50, background_color="white", stopwords = stop_words).generate(wds_tweets.lower())

    # Save word cloud as image file
    tweets_wordcloud.to_file('static/wordcloud.png')

    # Render the results template with the DataFrame as a parameter        
    return render_template('result.html', topic=topic, tweets=tweets_df10.to_html(index=False))

if __name__ == '__main__':
    app.run(debug=True)