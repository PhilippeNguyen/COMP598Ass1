from __future__ import division
__author__ = 'yaSh'

keyorder = ["url","n_tokens_title","n_tokens_content","n_unique_tokens","n_non_stop_words","n_non_stop_unique_tokens",	"num_hrefs", "num_self_hrefs", "num_imgs",	 "num_videos",	"average_token_length",	 "weekday_is_monday","weekday_is_tuesday","weekday_is_wednesday","weekday_is_thursday","weekday_is_friday", "weekday_is_saturday","weekday_is_sunday",	"is_weekend","global_subjectivity",	"global_sentiment_polarity", "title_subjectivity","title_sentiment_polarity","likes"]
import urllib2
import re
import numpy as np
import pandas as pd
from BeautifulSoup import BeautifulSoup

visited = set()
queue = []
#queue = Queue()
name = "huffingtonpost"
def get_articles(start_url, depth, feeds):
    queue = [start_url]
    while queue:
        if len(feeds) > 0: break
        for url in queue:
            queue.remove(url)
            request = urllib2.Request(url)
            response = urllib2.urlopen(request)
            soup = BeautifulSoup(response)
            articles = soup.findAll('a', href=re.compile(r'\d{4}/\d{2}/\d{2}/\w+'))
            for k,v in process_links(articles).iteritems():
                feeds.append({'url':k, 'date':v})
        for item in feeds: queue.append(item['url'])
    return feeds



def process_links(articles):
    url_list = {}
    for article in articles:
        try:
            url = article['href']
            if (name in url) & (str(url).endswith('.html')) & (url not in visited) & ('linkedin' not in url):
                visited.add(url)
                publish_date =re.search(r'(\d+/\d+/\d+)', url).group(1)
                url_list[article['href']] = publish_date
        except KeyError:
            continue
    return url_list

def get_num_links(url):
     # url_list = re.findall('"((http)s?://.*?)"', (urllib2.urlopen(url)).read())
     request = urllib2.Request(url)
     response = urllib2.urlopen(request)
     soup = BeautifulSoup(response)
     hrefs = soup.findAll('a')
     self_hrefs = soup.findAll('a', href=re.compile(r'\d{4}/\d{2}/\d{2}/\w+'))
     num_self_hrefs = len(self_hrefs)
     num_hrefs = len(hrefs)
     return  num_hrefs, num_self_hrefs

def get_title(url):
    soup = BeautifulSoup(urllib2.urlopen(url))
    return soup.title.string

def get_text(url):
    from boilerpipe.extract import Extractor
    try :
        extractor = Extractor(extractor='ArticleExtractor', url=url)
        return extractor.getText()
    except:
        return ""

def get_html(url):
    from boilerpipe.extract import Extractor
    try :
        extractor = Extractor(extractor='DefaultExtractor', url=url)
        return extractor.getHTML()
    except:
        return ""


def get_num_images(url):
    #get div for main article first
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response)
    images = []
    for i in soup.findAll('img'):
        if ('<img src=' in str(i)):
             images.append(i)
    return len(images)


def get_num_non_stop_words(content):
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    return 1 - len([i for i in content.split() if i not in stop])/len(content)

def get_num_likes(url):
     import json
     try:
         share_link = "http://graph.facebook.com/?id="+url
         response = urllib2.urlopen(share_link)
         data = json.loads(response.read())
         shares =  dict(data)['shares']
     except:
         shares = 0
     print shares
     return shares



corpus = {}
def compute_features(url,date):
    #omitting articles with no text
    content = (get_text(url).encode('ascii', 'ignore'))
    title = (get_title(url).encode('ascii', 'ignore'))
    if (len(content)) & (len(title)):
        corpus[title] = content
        print "Computing Feature Vector for: " + title + " | Published: " + date
        feature_vector = {'url':url, 'title':title, 'content':content, 'publish_date':date}
        score_hash = {}
        n_tokens_title = len(title.split())
        n_tokens_content = len(content.split())
        average_token_length = sum(len(list(word)) for word in content.split())/n_tokens_content
        num_images = get_num_images(url)
        score_hash['n_tokens_title'] = n_tokens_title
        score_hash['n_tokens_content'] = n_tokens_content
        score_hash['n_non_stop_words'] = get_num_non_stop_words(content)
        score_hash['average_token_length'] = average_token_length
        score_hash['num_imgs'] = num_images
        score_hash['num_videos'] = 0
        num_hrefs, num_self_hrefs = get_num_links(url)
        score_hash['num_hrefs'] = num_hrefs
        score_hash['num_self_hrefs'] = num_self_hrefs
        score_hash['likes'] = get_num_likes(url)
        date_hash = get_day(date)
        feature_vector['scores'] = dict(score_hash,**date_hash)
        # print feature_vector['scores']
        return feature_vector
    return {}

from datetime import datetime
def get_day(date):
    weekend = ['friday', 'saturday', 'sunday']
    date_time = datetime.strptime(date, "%Y/%m/%d")
    day = date_time.strftime("%A")
    is_weekend = 0
    if (day.lower() in weekend): is_weekend = 1
    day_hash = {'monday':0, 'tuesday':0, "wednesday":0,'thursday':0,'friday':0,'saturday':0, 'sunday':0}
    for (category,value) in day_hash.iteritems():
        if day.lower() == category:
            day_hash[category] = 1
            break
    for k in day_hash.iterkeys():
        if 'weekday' in str(k) : continue
        new = "weekday_is_"+str(k)
        day_hash[new] = day_hash.pop(k)
    day_hash['is_weekend'] = is_weekend
    return day_hash

def get_word_list(stop_words):
    unique_words = []
    from sklearn.feature_extraction.text import CountVectorizer
    if stop_words :
        from nltk.corpus import stopwords
        stop = stopwords.words('english')
        vectorizer = CountVectorizer(analyzer = "word", stop_words=stop, max_features = 5000)
    else:
        vectorizer = CountVectorizer(analyzer = "word", stop_words=None, max_features = 5000)
    words = vectorizer.fit_transform([i for i in corpus.itervalues()])
    dist = np.sum(words.toarray(),axis=0)
    median = np.median(dist)
    vocab = vectorizer.get_feature_names()
    for (feature, value) in zip(vocab,dist):
        if float(value) <= (median): unique_words.append(feature)
    return unique_words

from textblob import TextBlob
def textual_analysis(data):
    #need to stem words
    unique_words= get_word_list(stop_words=False)
    non_stop_unique = get_word_list(stop_words=True)
    for dict in data:
        freq_1 = 0
        freq_2 = 0
        for word in dict['content'].split():
            if word in unique_words:
                freq_1+=1
            if word in non_stop_unique:
                freq_2+=1
        blob_content = TextBlob(dict['content'])
        blob_title = TextBlob(dict['title'])
        scores = dict['scores']
        scores['n_unique_tokens'] = float(freq_1)/float(scores['n_tokens_content'])
        scores['n_non_stop_unique_tokens'] = float(freq_2)/float(scores['n_tokens_content'])
        scores['global_sentiment_polarity'] = blob_content.sentiment.polarity
        scores['global_subjectivity'] = blob_content.sentiment.subjectivity
        scores['title_sentiment_polarity'] = blob_title.sentiment.polarity
        scores['title_subjectivity'] = blob_title.sentiment.subjectivity
    return data

def run_crawler():
    url = 'http://www.huffingtonpost.ca/'
    feeds_by_date = get_articles(url, depth=2,feeds=[])
    print str(len(feeds_by_date))  +  " Articles Found"
    from operator import itemgetter
    sorted_feeds = sorted(feeds_by_date, key=itemgetter('date'))
    sorted_feeds = sorted(sorted_feeds, key=itemgetter('url'))
    data = []
    for feed in sorted_feeds:
            feature_vector = compute_features(feed['url'], feed['date'])
            if feature_vector: data.append(feature_vector)
    data = textual_analysis(data)
    scores_data = []
    for dict in data:
        scores = dict['scores']
        scores['url'] =dict['url']
        scores_data.append(scores)
    scores_data = pd.DataFrame(scores_data)[keyorder]
    scores_data.to_csv("data.csv")


def main():
    run_crawler()

if __name__ == '__main__':
    main()