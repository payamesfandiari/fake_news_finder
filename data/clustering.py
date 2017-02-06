
import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import calinski_harabaz_score,silhouette_samples,silhouette_score
from sklearn.cluster import MiniBatchKMeans
from clean_data import NLTKPreprocesor
from clean_data import extract_txt
import pickle
from os.path import isfile
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        return result,te-ts

    return timed


@timeit
def build_and_clean(X,y=None,outpath=None,verbose = True):

    if outpath:
        if isfile(outpath):
            if verbose:
                print("A CLEAN DATA IS DETECTED...")
                print("Loading...")
            return pickle.load(open(outpath,'rb'))
        else:
            preProcess = Pipeline([
                ('NLTKpreprocess',NLTKPreprocesor()),
                ('vectorizer',TfidfVectorizer(encoding='latin1',tokenizer=lambda x: x,
                                              preprocessor=None,lowercase=False))
            ])
            if verbose:
                print("Transforming data...")
            new_data = preProcess.fit_transform(X,y)
            with open(outpath,'wb') as f:
                pickle.dump(new_data,f)

            return new_data
    else:
        preProcess = Pipeline([
            ('NLTKpreprocess', NLTKPreprocesor()),
            ('vectorizer', TfidfVectorizer(encoding='latin1', tokenizer=lambda x: x,
                                           preprocessor=None, lowercase=False))
        ])
        if verbose:
            print("Transforming data...")


        return preProcess.fit_transform(X, y)

@timeit
def feature_reduction(X,output_dims = None,verbose=True):
    if output_dims is None:
        output_dims = int(np.sqrt(X.shape[1]))
    if verbose:
        print("Performing Feature Reduction")
        print("Data before Reduction : {}".format(X.shape))

    lsa = Pipeline([
        ('SVD',TruncatedSVD(output_dims,algorithm='arpack')),
        ('normalize',Normalizer(copy=False))
    ])


    return lsa.fit_transform(X),int(lsa.named_steps['SVD'].explained_variance_ratio_.sum()*100)


def main(data,clustering,reduce_dims = True,outpath=None,verbose=True):
    if verbose:
        print("Reading the data...")
        print("Creating the indexes...")

    ind  = data[:,0]
    data = data[:,1]

    if verbose:
        print('Read {0} rows of data...'.format(ind.shape[0]))

    X,secs = build_and_clean(data,outpath=outpath,verbose=verbose)
    if verbose:
        print("Complete data build in {:0.3f} seconds".format(secs))
        print("New data has {0},{1} dimension".format(X.shape[0],X.shape[1]))
        print("Starting Clustering...")

    if reduce_dims:
        (X,var),secs = feature_reduction(X,5000,verbose=verbose)
        if verbose:
            print("Complete data build in {:0.3f} seconds".format(secs))
            print("Explained variance of the SVD : {}".format(var))
            print("New data has {0},{1} dimension".format(X.shape[0], X.shape[1]))


    clustering.fit(X)
    pred_lbl = clustering.predict(X)
    if verbose:
        print("Finished Clustering with score {:0.3f}".format(clustering.inertia_))
        print("Computing Calinski_Harabaz Score")
    # print(calinski_harabaz_score(X.toarray(),pred_lbl))
    if verbose:
        print("Computing Silhouette Score")
    print(silhouette_score(X,pred_lbl,sample_size=10000))

if __name__ == '__main__':
    data = extract_txt('./data/data_clean.csv','./data/processed_data.csv')
    main(data,MiniBatchKMeans(n_clusters=100, verbose=0),reduce_dims=True,outpath = 'proc_data',verbose = True)


