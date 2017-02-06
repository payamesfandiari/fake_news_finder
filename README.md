# How Fake is it ? 
This is the background work behind the website [How Fake Is it?](https://howfakeisit.herokuapp.com/). In this project there are two things.
Data and Crawlers.

## Features of How Fake Is It?

- Can give you a measurement of how fake a news article is
- Have collaborative filtering. Users can give scores to News Articles
- Based on the latest approaches in Machine Learning.

## Data
### Kaggle Dataset
This Data is consists of different parts. The first part is [Kaggle's Fake News Dataset](https://www.kaggle.com/mrisdal/fake-news)
which is available online. From the explanation of data in the original website : 

> The dataset contains text and metadata from 244 websites and represents 12,999 posts in total from the past 30 days. The data was 
> pulled using the [webhose.io][4] API; because it's coming from their crawler, not all websites identified by the BS Detector are 
> present in this dataset. Each website was labeled according to the [BS Detector as documented here][5]. Data sources that were missing > a label were simply assigned a label of "bs". There are (ostensibly) no genuine, reliable, or trustworthy news sources represented in > this dataset (so far), so don't trust anything you read.
> 
> ## Fake news in the news
> 
> For inspiration, I've included some (presumably non-fake) recent stories covering fake news in the news. This is a sensitive, nuanced > topic and if there are other resources you'd like to see included here, please leave a suggestion. From defining fake, biased, and 
> misleading news in the first place to deciding how to take action (a blacklist is not a good answer), there's a lot of information to > consider beyond what can be neatly arranged in a CSV file.
### Crawling CNN,BBC,100FEDUP,FOX,Breitbart News Network
The big part of this Dataset comes from Crawling several websites and using their data for Training the model. The crawlers are included in this package and you can run them yourself.
