# How Fake is it ? 
This is the background work behind the website [How Fake Is it?](https://howfakeisit.herokuapp.com/). In this project there are two things.
Data and Crawlers.

## Features of How Fake Is It?

- Can give you a measurement of how fake a news article is
- Have collaborative filtering. Users can give scores to News Articles
- Based on the latest approaches in Machine Learning.

## Data
Our data comes from variety of sources. Most of the data have been mined and scrapped from different websites.
The other two datasets are from Kaggle and from [George McIntire Fake News Dataset.](https://github.com/GeorgeMcIntire/fake_real_news_dataset) 
### Crawling CNN,BBC,100FEDUP,FOX,Breitbart News Network
The big part of this Dataset comes from Crawling several websites and using their data for Training the model. The crawlers are included in this package and you can run them yourself.
By Using [Scrapy](https://scrapy.org/), We have downloaded around 50,000 News Article. We have saved the following fields :

1. Authors
2. Title
3. Text of the News
4. Number of Related Articles
5. Number of Sources
6. Number of images
7. Published Date

These fields has been cleared for usage. The bulk of analysis happens on the **Text of the News** but **Number of Related Articles,
Number of Sources and Number of images** are also considered.

### Kaggle Dataset
This Data is consists of different parts. The first part is [Kaggle's Fake News Dataset](https://www.kaggle.com/mrisdal/fake-news)
which is available online. From the explanation of data in the original website : 

> The dataset contains text and metadata from 244 websites and represents 12,999 posts in total from the past 30 days. The data was 
> pulled using the [webhose.io][4] API; because it's coming from their crawler, not all websites identified by the BS Detector are 
> present in this dataset. Each website was labeled according to the [BS Detector as documented here][5]. Data sources that were missing > a label were simply assigned a label of "bs". There are (ostensibly) no genuine, reliable, or trustworthy news sources represented in > this dataset (so far), so don't trust anything you read.
> 

### Fake or Real dataset 
This dataset obtained from [George McIntire Fake News Dataset.](https://github.com/GeorgeMcIntire/fake_real_news_dataset)
The dataset contains around 11,000 news article classified into two major classes.


