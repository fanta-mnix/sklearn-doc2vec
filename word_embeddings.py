import nltk
import numpy as np
import scipy.sparse as sp
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import to_unicode
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


def custom_transformer(tokens):
    return tokens


class TaggedLineDocument(object):

    def __init__(self, corpus, tokenizer=nltk.RegexpTokenizer(r'(?u)\b(?:\d+?(?:[\.\-/_:,]\d+)*|\w\w+)\b')):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.transformer = custom_transformer
        self.documents = None

    def __iter__(self):
        """Iterate through the lines in the source."""
        if self.documents is None:
            documents = []
            for item_no, document in enumerate(self.corpus):
                tokens = self.tokenizer.tokenize(to_unicode(document))
                documents.append(TaggedDocument(self.transformer(tokens), [item_no]))

            self.documents = documents

        return self.documents.__iter__()

    def shuffle(self):
        if self.documents is None:
            raise ValueError

        np.random.shuffle(self.documents)
        return self.documents

    def reorder(self):
        self.documents = sorted(self.documents, key=lambda x: x.tags[0])


class DocumentTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=True):
        return TaggedLineDocument(X)


class Doc2VecTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, size=300, window=8, min_count=5, sample=1e-3, negative=5, epochs=20):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.sample = sample
        self.negative = negative
        self.epochs = epochs
        self._model = None

    def fit(self, X, y=None):
        model = Doc2Vec(X, size=self.size, window=self.window, min_count=self.min_count, sample=self.sample, negative=self.negative)

        try:
            for epoch in range(self.epochs):
                print('Epoch: {}'.format(epoch))
                model.train(X.shuffle())

            self._model = model
            return self
        finally:
            X.reorder()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self._model.docvecs

    def transform(self, X, copy=True):
        assert self._model is not None, 'model is not fitted'
        return np.asmatrix(np.array([self._model.infer_vector(document.words) for document in X]))

