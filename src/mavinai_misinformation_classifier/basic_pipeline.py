""" LogisticRegression with preprocessor to extract text length as only feature

>>> X = pd.DataFrame([
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
...     ])
>>> target_names = list(trainingset[0].keys())[1:]
>>> model = BaselinePipeline()
>>> model.fit(training_set['text'], training_set[target_names])
>>> model.predict()
"""
import pandas as pd  # noqa
from sklearn.pipeline import Pipeline


class TextStatsTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, functions=[len]):
        self.functions = functions

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)

        Inputs:
          X (sequence of str): array of text strings
        """
        # raise NotImplementedError("TextStatsTransformer does not require fitting")
        return self

    def transform(self, X, copy=True, columns=None, index=None):
        X = pd.DataFrame(X)
        if columns is not None:
            X.columns = columns
        for fun in functions:
            for c in columns:
                try:
                    X[f"{c}_{fun.__name__}"] = X[c].str.getattr(fun.__name__)()
                except:


        return X

    @property
    def idf_(self):
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))

    @idf_.setter
    def idf_(self, value):
        value = np.asarray(value, dtype=np.float64)
        n_features = value.shape[0]
        self._idf_diag = sp.spdiags(value, diags=0, m=n_features,
                                    n=n_features, format='csr')

    def _more_tags(self):
        return {'X_types': 'sparse'}

    pass


from sklearn.feature_extraction import text


def build_pipeline():
    return Pipeline([])
