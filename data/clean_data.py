

import string
from os.path import isfile
import pandas as pd
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn

from nltk import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator,TransformerMixin



class NLTKPreprocesor(BaseEstimator,TransformerMixin):
    def __init__(self,stopwords = None,punct = None,lower = True,strip=True):
        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords or set(sw.words('english'))
        self.punct = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self,X,y=None):
        return self

    def inverse_transform(self,X):
        pass

    def transform(self,X):
        return [list(self.tokenize(doc)) for doc in X]

    def tokenize(self,document):

        for sent in sent_tokenize(document):
            for token,tag in pos_tag(wordpunct_tokenize(sent)):
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token
                token = token.strip('#') if self.strip else token

                if token in self.stopwords:
                    continue

                if all(char in self.punct for char in token):
                    continue

                if len(token) <= 0:
                    continue

                lemma = self.lemmatize(token,tag)
                yield lemma

    def lemmatize(self,token,tag):
        tag ={
            'N' : wn.NOUN,
            'V' : wn.VERB,
            'R' : wn.ADV,
            'J' : wn.ADJ
        }.get(tag[0],wn.NOUN)

        return self.lemmatizer.lemmatize(token,tag)



def extract_txt(excel_file,out_file,delim=';'):
    '''
    Extract Text File from Excel File
    :param excel_file: name of the File
    :return:
    '''
    if isfile(out_file):
        data = pd.read_csv(out_file,index_col=None,delimiter=delim)
        return data.values
    if '.csv' in excel_file:
        data = pd.read_csv(excel_file, index_col=None, delimiter=delim)
        new_data = data.loc[:,['SR_Service_RecID', 'SR_Detail_Notes']]
        new_data.to_csv(out_file, index=False,sep=delim)
        return data.loc[:,['SR_Service_RecID', 'SR_Detail_Notes']].values
    else:

        data = pd.read_excel(excel_file, sheetname='Sheet1')
        new_data = pd.DataFrame([],columns=['SR_Service_RecID','SR_Detail_Notes'],index=data.SR_Service_RecID.unique())
        for ID,group in data.groupby('SR_Service_RecID'):
            temp = ''
            if group.shape[0] == 1 :
                new_data.loc[ID]['SR_Service_RecID'] = group['SR_Service_RecID'].values[0]
                new_data.loc[ID]['SR_Detail_Notes'] = group['SR_Detail_Notes'].values[0]
            else:
                new_data.loc[ID]['SR_Service_RecID'] = group['SR_Service_RecID'].values[0]
                for s in group['SR_Detail_Notes'].values:
                    temp += str(s) + '\n'
                new_data.loc[ID]['SR_Detail_Notes'] = temp

        new_data.to_csv(out_file,index=False)
        return new_data.values