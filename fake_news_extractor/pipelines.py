# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import pymongo
from scrapy.conf import settings
from scrapy.exceptions import DropItem
import logging
import datetime

class FakeNewsExtractorPipeline(object):
    def process_item(self, item, spider):
        return item


class MongoDBPipeline(object):
    def __init__(self):
        connection = pymongo.MongoClient(
            settings['MONGODB_SERVER'],
            settings['MONGODB_PORT']
        )
        db = connection[settings["MONGODB_DB"]]
        self.collection = db[settings["MONGODB_CNN_COLLECTION"]]
        self.log = logging.getLogger(__name__)

    def process_item(self,item,spider):

        for data in item:
            if not data:
                raise DropItem("Missing {0}!".format(data))
        if self.collection.find_one({'url' : item["url"]}):
            self.log.info("News already in MongoDB. {0}".format(item["url"]))
            return item
        else:
            try:
                item["text"] = ''.join(item["text"])
            except:
                self.log.error("Cannot convert to text")
            try:
                item["published"] = datetime.datetime.strptime(item["published"],'%Y-%m-%dT%H:%M:%SZ')
            except:
                self.log.info("Cannot convert to Datetime")
            self.collection.update({'url' : item["url"]},dict(item),upsert=True)
            self.log.info("News added to MongoDB. {0}".format(item["url"]))
            return item
