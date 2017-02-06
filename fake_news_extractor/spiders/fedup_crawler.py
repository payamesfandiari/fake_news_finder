# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from ..items import FakeNewsExtractorItem as newsItem
from datetime import datetime


class FedupCrawlerSpider(CrawlSpider):
    name = 'fedup_crawler'
    allowed_domains = ['http://100percentfedup.com/','100percentfedup.com']
    start_urls = ['http://www.100percentfedup.com/']

    rules = (
        Rule(LinkExtractor(allow_domains=allowed_domains), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        content = response.xpath('//article[contains(@class,"post")]')
        if content is not None and len(content) is not 0:
            content = content[0]
            story = newsItem()
            story["published"] = content.xpath('.//time/@datetime').extract_first()
            story['title'] = content.xpath('.//header/h1/text()').extract_first()
            story['url'] = response.url
            story['linked_tweets'] = len(content.xpath('.//*[contains(@class,"tweet")]'))
            story['related_links'] = len(content.xpath('.//p//a[not(contains(@href,"twitter"))] | .//h6//a |.//strong//a'))
            story['imgs'] = content.xpath('.//img/@src').extract()
            story['text'] = content.xpath('.//p/text() | .//h5/text() | .//strong/text()').extract()
            story['authors'] = "100FEDUP"
            story['crawled_date'] = datetime.now()
            return story
        else:
            self.log("Not Story found")
