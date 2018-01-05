import re
import time
from datetime import datetime,timedelta
from collections import Counter,defaultdict

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import metrics

from pprint import pprint
import pymongo
from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
client = pymongo.MongoClient('54.161.160.206:29017')
db = client.tweet

import redis
r = redis.StrictRedis(host='54.161.160.206', port=6479, db=0)

from tqdm import tqdm

import spacy
nlp = spacy.load('en_core_web_md')

locs=["North Korea"]
triggers=["test","launch","fire"]
targets = ["messile","satellite","rocket","nuclear"]
keywords = locs+triggers+targets
doc_keywords = [nlp(key.decode('utf-8')) for key in keywords]

def Doc2VecTransformer(X):
		vectors = []
		keyword_similarity = {}
		for i in X:
			doc = nlp(i)
			for token in doc:
				keyword_similarity[token.text] = np.mean([i.similarity(token) for i in doc_keywords])
				vectors.append(doc.vector)   
		return np.asmatrix(vectors),keyword_similarity

def tweet_cluster(X,max_cluster_num=8):
		count = CountVectorizer(ngram_range=(1, 3),min_df=2,stop_words='english')
		X_train_count = count.fit_transform(X)
		tfidf = TfidfTransformer(use_idf=True)
		X_train_tfidf = tfidf.fit_transform(X_train_count)
		# km,score = kmeans_best(X,X_train_tfidf,max_cluster_num)
		X_train_vector,keyword_similarity = Doc2VecTransformer(X)
		km,score = kmeans_best(X,X_train_vector,max_cluster_num)
		#print 'clusters:',km.get_params()['n_clusters']
		lda = LatentDirichletAllocation(n_topics=km.get_params()['n_clusters'],
									max_iter=5,
									learning_method='online',
									learning_offset=50.,
									random_state=0)
		lda.fit(X_train_tfidf)
		tf_feature_names = count.get_feature_names()
		lda_words = get_topics_top_words(lda, tf_feature_names)
		#for k,v in lda_words.iteritems():
			#lda_words[k] = [v[ix] for ix in np.argsort([keyword_similarity[i] for i in v])[:-11:-1]]
		return km,lda_words


def kmeans_best(X,X_train_tfidf=None,max_cluster_num=8):
	text_clusters = []
	for i in range(max_cluster_num)[2:]:
		#print i
		mbkm = MiniBatchKMeans(n_clusters=i)
		mbkm.fit(X_train_tfidf)
		socre = metrics.silhouette_score(X_train_tfidf, mbkm.labels_)
		text_clusters.append((mbkm,socre))
	return sorted(text_clusters,key=lambda i:i[1])[0]
    
def get_topics_top_words(model, feature_names, n_top_words=10):
	topics_top_words = {}
	for topic_idx, topic in enumerate(model.components_):
		topics_top_words[str(topic_idx)] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
	return topics_top_words
	
def get_ner(text):
	doc = nlp(text)
	who,where,when,what =([],[],[],[])
	for ner in doc.ents:
		if ner.label_ in ['PERSON','NORP','ORG','PRODUCT','WORK_OF_ART','LAW']:
			who.append(ner.text)
		elif ner.label_ in ['FACILITY','GPE','LOC',]:
			where.append(ner.text)
		elif ner.label_ in ['DATE','TIME']:
			when.append(ner.text)
		elif ner.label_ in ['EVENT']:
			what.append(ner.text)
		else:
			pass
	return who,where,when,what
	

def clustering():
	query = db.korea.find({'class.1':{'$gte':0.5},'cluster':None},{'_id':1,'tweet.standard_text':1,'tweet.hashtags':1,'tweet.created_at':1}).limit(1000)
	ids = []
	texts = []
	hashtags = []
	dates = []
	ners = []
	for i in query:
		ids.append(i['_id'])
		texts.append(i['tweet']['standard_text'])
		hashtags.append(i['tweet']['hashtags'])
		dates.append(i['tweet']['created_at'])
		ners.append(get_ner(i['tweet']['standard_text']))
	if len(ids) == 0:
		return None
	cluster_hash  = ids[0]+ids[-1]
	km,lda_words = tweet_cluster(texts)
	clusters = km.labels_
	clusters = [int(i) for i in clusters]
	requests = [UpdateOne({'_id': _id,'cluster':None}, 
		{'$set': {'cluster':{'cluster_label':clusters[index],'cluster_hash':cluster_hash}}}) for index,_id in tqdm(enumerate(ids))]
	result = db.korea.bulk_write(requests)
	pprint(result.bulk_api_result)
	
	whos,wheres,whens,whats = (defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list))
	for index,_id in tqdm(enumerate(ids)):
		who,where,when,what = ners[index]
		whos[clusters[index]].extend(who)
		wheres[clusters[index]].extend(where)
		whens[clusters[index]].extend(when)
		whats[clusters[index]].extend(what)
	clusters_counter = dict(Counter(clusters))
	clusters_counter_ = defaultdict()
	for k,v in clusters_counter.iteritems():
		clusters_counter_[str(k)] = v
	
	cluster_entities = defaultdict()
	for k,v in clusters_counter.iteritems():
		cluster_entities[str(k)] = {'whos':[{i[0]:i[1]} for i in Counter(whos[k]).most_common(3)],
									'wheres':[{i[0]:i[1]} for i in Counter(wheres[k]).most_common(3)],
									'whens':[{i[0]:i[1]} for i in Counter(whens[k]).most_common(3)],
									'whats':[{i[0]:i[1]} for i in Counter(whats[k]).most_common(3)]
		}
	
	clusters_hashtags = defaultdict(list)
	for index,hashtag in enumerate(hashtags):
		if len(hashtag) > 0:
			clusters_hashtags[str(clusters[index])].extend(hashtag)
	for k,v in clusters_hashtags.iteritems():
		clusters_hashtags[k] = [{i[0]:i[1]} for i in Counter(v).most_common(3)]

	db.cluster_metadata.insert_one({'_id':cluster_hash,'start_time':dates[0],'end_time':dates[-1],
		'texts_num':len(texts),'clusters_size':clusters_counter_,'clusters_hashtags':clusters_hashtags,
		'cluster_entities':cluster_entities,'topics':lda_words})

if __name__ == '__main__':
	print 'clustering_worker start!'
	while True:
		queue = r.lpop('task:clustering')
		if queue:
			print 'clustering_worker process!'
			clustering()
		print 'clustering_worker wait!'
		time.sleep(1)