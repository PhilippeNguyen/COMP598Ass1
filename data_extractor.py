__author__ = 'yaSh'

url_list = []
corpus = {}
import re
import urllib2
from BeautifulSoup import BeautifulSoup
from Queue import Queue, Empty
from threading import Thread

visited = set()
queue = Queue()

def get_text(url):
    from boilerpipe.extract import Extractor
    try :
        extractor = Extractor(extractor='DefaultExtractor', url=url)
        return extractor.getText(), extractor.getHTML()
    except:
        return "",""

def get_title(html):
    try:
        titleRE = re.compile(r"<H1 class=\"title\">(.+?)</H1>")
        title = titleRE.search(html).group(1)
    except: title = ""
    return title

def get_topic(html):
    try:
        topicRE = re.compile(r"<LABEL for=\"vertical-brief\">(.+?)</LABEL>")
        topic = topicRE.search(html).group(1)
    except: topic = "Other"
    return topic

import json
def get_num_likes(url):
     try:
         share_link = "http://graph.facebook.com/?id="+url
         response = urllib2.urlopen(share_link)
         data = json.loads(response.read())
         shares =  dict(data)['shares']
     except:
         shares = 0
     return shares


def link_crawl(root, article_limit):
    name = root.rsplit(".")[1]
    def parse():
        try:
            while True:
                if len(url_list) > article_limit: break
                url = queue.get_nowait()
                try:
                    request = urllib2.Request(url)
                    response = urllib2.urlopen(request)
                    soup = BeautifulSoup(response)
                except UnicodeDecodeError:
                    continue
                for link in soup.findAll('a', href=re.compile(r'\d{4}/\d{2}/\d{2}/\w+')):
                    try:
                        href = link['href']
                    except KeyError:
                        continue
                    if href not in visited:
                        visited.add(href)
                        if (name in href) & (str(href).endswith('.html')) & ('linkedin' not in href):
                            queue.put(href)
                            date =re.search(r'(\d+/\d+/\d+)', href).group(1)
                            text, html = get_text(href)
                            title = get_title(html)
                            topic = get_topic(html)
                            likes = get_num_likes(href)
                            if len(title) & len(text):
                                feature_vector = {'url':href, 'title':title, 'content': text, 'date':date, 'topic':topic, 'likes':likes}
                                corpus[title] = text
                                print (title, date, topic, likes)
                                if date:
                                    url_list.append(feature_vector)
        except Empty:
            pass
    return parse

def get_data(limit):
    corpus = []
    import time
    start = time.time()
    root = "http://www.huffingtonpost.ca/"
    parser = link_crawl(root, article_limit=limit)
    queue.put(root)
    workers = []
    for i in range(5):
        worker = Thread(target=parser)
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()
    end = time.time()
    print
    print "Time Taken to Extract Data : " + (str((end - start)/60.0)) + " minutes"
    print
    return url_list

