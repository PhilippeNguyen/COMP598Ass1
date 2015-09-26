from __future__ import division
__author__ = 'yaSh'

import urllib2
import re
from BeautifulSoup import BeautifulSoup


# import scrapy
#
# from scrapy.item import Item, Field
#
# class Article(Item):
#     title = Field()
#     author = Field()
#     tag = Field()
#     date = Field()
#     link = Field()
#
# from scrapy.contrib.spiders import CrawlSpider, Rule
# from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
# from scrapy.selector import HtmlXPathSelector
#
# def article_scraper(CrawlSpider, *args, **kwargs):
#     start_url = [kwargs.get('start_url') if kwargs.get('start_url ')]
#     if not start_url : start_url = ['http://www.huffingtonpost.ca/']
#     rules = [Rule(SgmlLinkExtractor(allow=[r'\d{4}/\d{2}/\w+']), callback='parser')]
#     def parser(self, response):
#          hxs = HtmlXPathSelector(response)
#          item = Article()
#          item['title'] = hxs.select('//header/h1/text()').extract()
#          item['tag'] = hxs.select("//header/div[@class='post-data']/p/a/text()").extract()
#          return item




visited = set()
queue = []
name = "huffingtonpost"
def get_articles(url):
    print url
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response)
    url_list = {}
    articles = soup.findAll('a', href=re.compile(r'\d{4}/\d{2}/\d{2}/\w+'))
    for article in articles:
        try:
            url = article['href']
            if (name in url) & (str(url).endswith('.html')) & (url not in visited):
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
     for link in self_hrefs: queue.append(link['href'])
     num_self_hrefs = len(self_hrefs)
     num_hrefs = len(hrefs)
     return num_hrefs, num_self_hrefs

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

def get_num_non_stop_words(content):
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    return 1 - len([i for i in content.split() if i not in stop])/len(content)

def get_num_shares(url):
     url = "https://www.sharedcount.com/#url="+url
     request = urllib2.Request(url)
     response = urllib2.urlopen(request)
     soup = BeautifulSoup(response)
     print

# int GetLikes(string url) {
#
#     string jsonString = new System.Net.WebClient().DownloadString("http://graph.facebook.com/?ids=" + url);
#
#     var json = new System.Web.Script.Serialization.JavaScriptSerializer().Deserialize(jsonString);
#     int count = json[url]["shares"];
#
#     return count;
# }

def compute_features(url,date):
    #omitting articles with no text
    content = (get_text(url).encode('ascii', 'ignore'))
    title = (get_title(url).encode('ascii', 'ignore'))
    if (len(content)) & (len(title)):
        print
        print "Computing Feature Vector for: " + title + " | Published: " + date
        feature_vector = {'url':url, 'title':title, 'content':content, 'publish_date':date}
        score_hash = {}
        n_tokens_title = len(feature_vector['title'])
        n_tokens_content = len(feature_vector['content'])
        average_token_length = sum(len(word) for word in feature_vector['content'].split())/float(n_tokens_content)
        num_images = get_num_images(url)
        print get_num_shares(url)
        score_hash['n_tokens_title'] = n_tokens_title
        score_hash['n_tokens_content'] = n_tokens_content
        score_hash['average_token_length'] = average_token_length
        score_hash['num_images'] = num_images
        score_hash['num_non_stop_words'] = get_num_non_stop_words(content)
        num_hrefs, num_self_hrefs = get_num_links(url)
        score_hash['num_hrefs'] = num_hrefs
        score_hash['num_self_hrefs'] = num_self_hrefs
        feature_vector['scores'] = score_hash
        print
        print feature_vector['scores']
        return feature_vector

def main():
    url = 'http://www.huffingtonpost.ca/'
    feeds_by_date = get_articles(url)
    depth = 0
    print str(len(feeds_by_date)) + " Articles Found at Depth Level : " + str(depth)
    data = [compute_features(url,date) for (url,date) in feeds_by_date.iteritems()]
    # while depth < 2:
    #     depth += 1
    #     print str(len(queue)) + " Articles Found at Depth Level : " + str(depth)
    #     for url in queue:
    #         feeds_by_date = get_articles(url)
    #         data = [compute_features(url,date) for (url,date) in feeds_by_date.iteritems()]
    #     del queue[:]


if __name__ == '__main__':
    main()