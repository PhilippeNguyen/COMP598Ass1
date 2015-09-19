__author__ = 'yaSh'

import urllib2
from BeautifulSoup import BeautifulSoup

def get_links(url):
     import re
     return  re.findall('"((http)s?://.*?)"', (urllib2.urlopen(url)).read())

def get_articles(url, name):
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response)
    url_list = []
    for article in soup.findAll('a') :
        try:
            if name in article['href']:
                url_list.append(article['href'])
        except KeyError:
            continue
    return url_list

def get_title(url):
    soup = BeautifulSoup(urllib2.urlopen(url))
    return soup.title.string

def get_text(url):
    from boilerpipe.extract import Extractor
    try:
        extractor = Extractor(extractor='ArticleExtractor', url=url)
        return extractor.getText()
    except:
         return []

def get_num_images(url):
    from boilerpipe.extract import Extractor
    try:
        extractor = Extractor(extractor='ArticleExtractor', url=url)
        images = extractor.getImages()
        if images: return len(images)
        else : return 0
    except:
        return 0

def get_publish_date(url):
    from newspaper import Article
    print url
    article = Article(url)
    article.download()
    return article.publish_date


def compute_features(url):
    feature_vector = {'url':url, 'title':get_title(url), 'content':get_text(url)}
    score_hash = {}
    n_tokens_title = len(feature_vector['title'])
    n_tokens_content = len(feature_vector['content'])
    average_token_length = sum(len(word) for word in feature_vector['content'].split())/float(n_tokens_content)
    num_images = get_num_images(url)
    score_hash['n_tokens_title'] = n_tokens_title
    score_hash['n_tokens_content'] = n_tokens_content
    score_hash['average_token_length'] = average_token_length
    score_hash['num_images'] = num_images
    score_hash['num_hrefs'] = len(get_links(url))
    score_hash['num_self_hrefs'] = len(get_articles(url,'huffingtonpost'))
    feature_vector['scores'] = score_hash
    print feature_vector['scores']
    print get_publish_date(url)
    return feature_vector

def main():
    url = 'http://www.huffingtonpost.ca/'
    feeds = get_articles(url, "huffingtonpost")
    data = [compute_features(url) for url in feeds]
    print data

if __name__ == '__main__':
    main()