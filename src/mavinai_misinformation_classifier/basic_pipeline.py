""" LogisticRegression with preprocessor to extract text length as only feature

>>> training_set = pd.DataFrame([
...     dict(
...         text='Lorem ipsum article text',
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


class SimpleLogisticRegressionPipeline(Pipeline):
    pass
