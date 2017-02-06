# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from ..items import FakeNewsExtractorItem as newsItem
from datetime import datetime


class BbcCrawlerSpider(CrawlSpider):
    name = 'bbc_crawler'
    allowed_domains = ['bbc.com']
    start_urls = ['http://www.bbc.com/']

    rules = (
        Rule(LinkExtractor(allow=['news/.*'],allow_domains=allowed_domains), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        content = response.xpath('//div[@class="story-body"]')
        if content is not None and len(content) is not 0:
            content = content[0]
            story = newsItem()
            story["published"] = content.xpath('//*[contains(@class,"date")]/@data-datetime').extract_first()
            story['title'] = content.xpath('//*[contains(@class,"__h1")]/text()').extract_first()
            story['url'] = response.url
            story['linked_tweets'] = len(content.xpath('.//*[contains(@class,"story-body__link-external")]'))
            story['related_links'] = len(content.xpath('.//a[@class="story-body__link"]'))
            story['imgs'] = content.xpath('.//img/@src').extract()
            story['text'] = content.xpath('.//p[not(contains(@class,"twite"))]/text()').extract()

            if len(content.xpath('.//span[@class="byline__name"]')) == 0:
                story['authors'] = "BBC"
            else:
                story['authors'] = content.xpath('.//span[@class="byline__name"]/text()').extract_first()
            story['crawled_date'] = datetime.now()
            return story
        else:
            self.log("Not Story found")
