# -*- coding: utf-8 -*-
from datetime import datetime

import scrapy
from scrapy.spiders import Rule
from scrapy.linkextractors import LinkExtractor
from scrapy_splash import SplashRequest
from scrapy.shell import inspect_response
from ..items import FakeNewsExtractorItem as newsItem


class CnnSpider(scrapy.spiders.CrawlSpider):
    name = "cnn"
    allowed_domains = ["www.cnn.com"]
    start_urls = [
        'http://www.cnn.com/US/archive/',
        'http://www.cnn.com/specials/politics/national-politics',
        'http://www.cnn.com/specials/politics/world-politics',
    ]
    rules = [
        Rule(LinkExtractor(
            allow_domains=allowed_domains
        ), follow=True, callback='parse_item')
    ]

    # def start_requests(self):
    #     for url in self.start_urls:
    #         yield SplashRequest(url,self.parse_start_url,args={'wait': 1})

    # def parse_start_url(self, response):
    #     urls = response.xpath('//div[@class="cd__content"]/h3/a/@href').extract()
    #     for url in urls:
    #         page = response.urljoin(url)
    #         yield scrapy.Request(page,callback=self.parse_item)

    def parse_item(self,response):
        content = response.xpath('//article[contains(@class,"pg-rail")]')
        if content is not None and len(content) is not 0:
            content = content[0]
            story = newsItem()
            story["published"] = content.xpath('./meta[contains(@itemprop,"datePublished")]/@content').extract_first()
            story['title'] = content.xpath('./meta[contains(@itemprop,"headline")]/@content').extract_first()
            story['url'] = response.url
            story['linked_tweets'] = len(content.xpath('.//*[contains(@class,"twitter-")]'))
            story['related_links'] = len(content.xpath('.//*[contains(@class,"el__embedded")]'))
            story['imgs'] = content.xpath('./meta[contains(@itemprop,"image")]/@content').extract()
            story['text'] = content.xpath('.//*[@class="zn-body__paragraph"]/text()').extract()
            story['authors'] = content.xpath('./meta[contains(@itemprop,"author")]/@content').extract_first()
            story['crawled_date'] = datetime.now()
            return story
        else:
            self.log("Not Story found")




