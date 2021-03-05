""" LogisticRegression with preprocessor to extract text length as only feature

>>> X = pd.DataFrame([dict(text='Lorem ipsum article text')]*3)
>>> Y = pd.DataFrame([
...     dict(
...         score_accountability=0.0,
...         score_unobjectivity=0.2,
...         score_inaccuracy=0.4,
...         score_fact_basedness=0.6,
...         score_influential=0.8,
...         score_opinionated=1.0,
...         is_accountable=1,
...         is_objective=2,
...         is_innacurate=3,
...         is_fact_based='?',
...         is_influential=1,
...         is_opinionated=2,
...         ),
...     ]*3)
>>> Y = Y[[c for c in Y.columns if c.startswith('is_')]]
>>> model = build_random_guesser_pipeline()
>>> model.fit(X, Y)
>>> model.predict(X)
"""
import numpy as np
import pandas as pd  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class TextStatsTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, functions=['len']):
        self.functions = functions

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)

        Inputs:
          X (sequence of str): array of text strings
        """
        # raise NotImplementedError("TextStatsTransformer does not require fitting")
        return self

    def transform(self, X, copy=True, columns=None):
        X = pd.DataFrame(data=X)
        if columns is not None:
            X.columns = columns
        for fun in self.functions:
            for c in X.columns:
                if isinstance(fun, str):
                    try:
                        X[f"{c}_{fun.__name__}"] = getattr(X[c].str, fun)()
                    except AttributeError:
                        print(f'unable to find StringMethod {fun}')
                elif isinstance(fun, callable):
                    X[f"{c}_{fun.__name__}"] = X[c].apply(fun)
        return X


class DropTextTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, functions=['len']):
        pass

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)

        Inputs:
          X (sequence of str): array of text strings
        """
        # raise NotImplementedError("TextStatsTransformer does not require fitting")
        return self

    def transform(self, X, copy=True, columns=None):
        droppable_columns = []
        for c in X.columns:
            if X[c].dtype == 'O':
                droppable_columns.append(c)
        X.drop(columns=droppable_columns, inplace=True)
        return X


class RandomEstimator(BaseEstimator):
    def __init__(self, functions=['len']):
        pass

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)

        Inputs:
          X (sequence of str): array of text strings
        """
        # raise NotImplementedError("TextStatsTransformer does not require fitting")
        self.shape = y.shape
        return self

    def predict(self, X, copy=True, columns=None):
        return np.random.randn(*self.shape)


def build_random_guesser_pipeline():
    return Pipeline([
        ('text_len_transformer', TextStatsTransformer()),
        ('drop_text_transformer', DropTextTransformer()),
        ('random_guesser', RandomEstimator())])


def build_logistic_regression_pipeline():
    return Pipeline([
        ('text_len_transformer', TextStatsTransformer()),
        ('drop_text_transformer', DropTextTransformer()),
        ('logistic_regression', LogisticRegression())])
