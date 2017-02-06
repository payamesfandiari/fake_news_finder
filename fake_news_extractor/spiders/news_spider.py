from scrapy.spiders import CrawlSpider,Rule
from scrapy.linkextractors import LinkExtractor
from ..items import FakeNewsExtractorItem
from newspaper import Article
from datetime import datetime


class NewsSpider(CrawlSpider):
    name = "news"
    allowed_domains = ['bbc.com']
    start_urls = ['http://www.bbc.com/news']
    rules = [
        Rule(LinkExtractor(
            allow=['/world-[\w-]+'],
            allow_domains=allowed_domains
        ), follow=True, callback='parse_item')
    ]

    def parse_item(self, response):
        self.log("Scraping : "+ response.url)
        try:
            article = Article(url=response.url)
            article.download()
            article.parse()
        except Exception:
            self.log("Something happened")
            return
        story = FakeNewsExtractorItem()
        story['title'] = article.title
        story['url'] = response.url
        story['imgs'] = list(article.imgs)
        story['text'] = article.text
        story['authors'] = article.authors
        story['sources'] = []
        story['published'] = article.publish_date
        story['crawled_date'] = datetime.now()
        yield story
