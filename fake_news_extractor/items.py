# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class FakeNewsExtractorItem(scrapy.Item):
    title = scrapy.Field()
    url = scrapy.Field()
    imgs = scrapy.Field()
    links = scrapy.Field()
    linked_tweets = scrapy.Field()
    related_links = scrapy.Field()
    text = scrapy.Field()
    authors = scrapy.Field()
    published = scrapy.Field()
    crawled_date = scrapy.Field()
    # define the fields for your item here like:
    # name = scrapy.Field()

