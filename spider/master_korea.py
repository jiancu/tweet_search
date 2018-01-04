import re
import json
import fire
import time
from tqdm import tqdm
from datetime import datetime,timedelta

from Config import get_spider_config
_,db,r = get_spider_config()


freq_users = set([i['tweet']['user']['screen_name'] for i in db.dataset_korea_m_1.find({},{"tweet.user.screen_name":1})])
            
def get_query_str(loc,triggers,target,user):
	return '('+loc+')' + + ' '+'('+' OR '.join(triggers)+')'+' '+'('+target+')'+"from "+user

def get_task():
    locs=["North Korea"]
    triggers=["test","launch","fire"]
    target = ["messile","satellite","rocket","nuclear"]
    while True:
        now = datetime.now()
        for user in freq_users:
            for loc in locs:
                for target in targets:
                    q = get_query_str(loc,triggers,target,user)
                    message = {'q':q,'f':['&f=news','','&f=tweets'],'num':1000,
                    "sinceTimeStamp":(now - timedelta(minutes=60)).strftime("%Y-%m-%d %H:%M:%S"),
                    "untilTimeStamp":now.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    r.rpush("task:korea",json.dumps(message))
        time.sleep(60*60)
if __name__ == '__main__':
	get_task()
