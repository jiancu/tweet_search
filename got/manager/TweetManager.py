import urllib,urllib2,json,re,datetime,sys,cookielib
from .. import models
from pyquery import PyQuery
import random
random.seed(1)

class TweetManager:
	
	def __init__(self):
		pass
		
	@staticmethod
	def getTweetsById(tweet_id):
		url = 'https://twitter.com/xxx/status/%s'%(tweet_id)
		tweets = PyQuery(url)('div.js-original-tweet')
		for tweetHTML in tweets:
			tweetPQ = PyQuery(tweetHTML)
			tweet = models.Tweet()

			usernameTweet = tweetPQ("span:first.username.u-dir b").text();
			txt = re.sub(r"\s+", " ", tweetPQ("p.js-tweet-text").text().replace('# ', '#').replace('@ ', '@'));
			retweets = int(tweetPQ("span.ProfileTweet-action--retweet span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""));
			favorites = int(tweetPQ("span.ProfileTweet-action--favorite span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""));
			dateSec = int(tweetPQ("small.time span.js-short-timestamp").attr("data-time"));
			id = tweetPQ.attr("data-tweet-id");
			permalink = tweetPQ.attr("data-permalink-path");

			geo = ''
			geoSpan = tweetPQ('span.Tweet-geo')
			if len(geoSpan) > 0:
				geo = geoSpan.attr('title')

			tweet.id = id
			tweet.permalink = 'https://twitter.com' + permalink
			tweet.username = usernameTweet
			tweet.text = txt
			tweet.date = datetime.datetime.fromtimestamp(dateSec)
			tweet.retweets = retweets
			tweet.favorites = favorites
			tweet.mentions = " ".join(re.compile('(@\\w*)').findall(tweet.text))
			tweet.hashtags = " ".join(re.compile('(#\\w*)').findall(tweet.text))
			tweet.geo = geo
		return tweet
		
	@staticmethod
	def getTweets(tweetCriteria, refreshCursor='', receiveBuffer=None, bufferLength=100, proxy=None):
		results = []
		resultsAux = []
		cookieJar = cookielib.CookieJar()
		
		if hasattr(tweetCriteria, 'username') and (tweetCriteria.username.startswith("\'") or tweetCriteria.username.startswith("\"")) and (tweetCriteria.username.endswith("\'") or tweetCriteria.username.endswith("\"")):
			tweetCriteria.username = tweetCriteria.username[1:-1]

		active = True

		while active:
			json = TweetManager.getJsonReponse(tweetCriteria, refreshCursor, cookieJar, proxy)
			if len(json['items_html'].strip()) == 0:
				break

			refreshCursor = json['min_position']
			tweets = PyQuery(json['items_html'])('div.js-stream-tweet')
			
			if len(tweets) == 0:
				break
			
			for tweetHTML in tweets:
				tweetPQ = PyQuery(tweetHTML)
				tweet = models.Tweet()
				
				usernameTweet = tweetPQ("span:first.username.u-dir b").text();
				txt = re.sub(r"\s+", " ", tweetPQ("p.js-tweet-text").text().replace('# ', '#').replace('@ ', '@'));
				retweets = int(tweetPQ("span.ProfileTweet-action--retweet span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""));
				favorites = int(tweetPQ("span.ProfileTweet-action--favorite span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""));
				dateSec = int(tweetPQ("small.time span.js-short-timestamp").attr("data-time"));
				id = tweetPQ.attr("data-tweet-id");
				permalink = tweetPQ.attr("data-permalink-path");
				
				geo = ''
				geoSpan = tweetPQ('span.Tweet-geo')
				if len(geoSpan) > 0:
					geo = geoSpan.attr('title')
				
				tweet.id = id
				tweet.permalink = 'https://twitter.com' + permalink
				tweet.username = usernameTweet
				tweet.text = txt
				#tweet.clean_text =   TO DO
				tweet.date = datetime.datetime.fromtimestamp(dateSec)
				#tweet.reply = reply   TO DO
				tweet.retweets = retweets
				tweet.favorites = favorites
				tweet.mentions = " ".join(re.compile('(@\\w*)').findall(tweet.text))
				tweet.hashtags = " ".join(re.compile('(#\\w*)').findall(tweet.text))
				#tweet.href =          TO DO
				tweet.geo = geo
				
				if hasattr(tweetCriteria, 'sinceTimeStamp'):
					if tweet.date < tweetCriteria.sinceTimeStamp:
						active = False
						break
				
				results.append(tweet)
				resultsAux.append(tweet)
				
				if receiveBuffer and len(resultsAux) >= bufferLength:
					receiveBuffer(resultsAux)
					resultsAux = []
				
				if tweetCriteria.maxTweets > 0 and len(results) >= tweetCriteria.maxTweets:
					active = False
					break
					
		
		if receiveBuffer and len(resultsAux) > 0:
			receiveBuffer(resultsAux)
		
		return results
	
	@staticmethod
	def getJsonReponse(tweetCriteria, refreshCursor, cookieJar, proxy):
		url = "https://twitter.com/i/search/timeline?q=%s&src=typd&max_position=%s"
		
		urlGetData = ''
		
		if hasattr(tweetCriteria, 'username'):
			urlGetData += ' from:' + tweetCriteria.username
		
		if hasattr(tweetCriteria, 'querySearch'):
			urlGetData += ' ' + tweetCriteria.querySearch
		
		if hasattr(tweetCriteria, 'near'):
			urlGetData += "&near:" + tweetCriteria.near + " within:" + tweetCriteria.within
		
		if hasattr(tweetCriteria, 'since'):
			urlGetData += ' since:' + tweetCriteria.since
			
		if hasattr(tweetCriteria, 'until'):
			urlGetData += ' until:' + tweetCriteria.until
		

		if hasattr(tweetCriteria, 'topTweets'):
			if tweetCriteria.topTweets:
				url = "https://twitter.com/i/search/timeline?q=%s&src=typd&max_position=%s"
		
		if hasattr(tweetCriteria, 'tweetType'):
			url = url + tweetCriteria.tweetType
		
		url = url % (urllib.quote(urlGetData), refreshCursor)
		ua = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.%s'%(random.randint(0,999))

		headers = [
			('Host', "twitter.com"),
			('User-Agent', ua), 
			# Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.81 Safari/537.36 
			#Mozilla/5.0 (Windows NT 6.1; Win64; x64)
			('Accept', "application/json, text/javascript, */*; q=0.01"),
			('Accept-Language', "de,en-US;q=0.7,en;q=0.3"),
			('X-Requested-With', "XMLHttpRequest"),
			('Referer', url),
			('Connection', "keep-alive")
		]

		if proxy:
			opener = urllib2.build_opener(urllib2.ProxyHandler({'http': proxy, 'https': proxy}), urllib2.HTTPCookieProcessor(cookieJar))
		else:
			opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookieJar))
		opener.addheaders = headers

		try:
			response = opener.open(url)
			jsonResponse = response.read()
		except Exception,e:
			print "Twitter weird response. Try to see on browser: https://twitter.com/search?q=%s&src=typd" % urllib.quote(urlGetData)
			raise Exception(e.message)
			#sys.exit()
			#return None
		
		dataJson = json.loads(jsonResponse)
		
		return dataJson		
