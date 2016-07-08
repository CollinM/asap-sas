from asap.core import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.linear_model import LogisticRegression as LogReg
import numpy as np


class SklearnModel(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None

    def process(self, instance):
        features = np.array(self.collect_features(instance)).reshape((1, -1))
        prdxn = self._model.predict(features)[0]
        instance.add_feature(self.key, prdxn)
        return instance

    def train(self, instances):
        # Collect features and target label
        inputs = []
        targets = []
        for inst in instances:
            inputs.append(self.collect_features(inst))
            targets.append(self.get_target(inst))

        # Train model
        X = np.array(inputs)
        y = np.array(targets)
        self._model.fit(X, y)


class RandomForest(SklearnModel):

    def __init__(self, model=None, num_trees=100, max_depth=None, threads=1, **kwargs):
        super().__init__(**kwargs)
        self._num_trees = num_trees
        self._max_depth = max_depth
        self._threads = threads
        self._model = model

    def train(self, instances):
        # Create model
        self._model = RandomForestClassifier(n_estimators=self._num_trees, max_depth=self._max_depth,
                                             n_jobs=self._threads)
        super().train(instances)


class RidgeRegression(SklearnModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, instances):
        # Create model
        self._model = RidgeClassifier()
        super().train(instances)


class LogisticRegression(SklearnModel):

    def __init__(self, penalty='l2', **kwargs):
        super().__init__(**kwargs)
        self._penalty = penalty

    def train(self, instances):
        self._model = LogReg(penalty=self._penalty, solver='lbfgs', multi_class='multinomial')
        super().train(instances)


class ElasticNetSVM(SklearnModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, instances):
        self._model = SGDClassifier(penalty="elasticnet", l1_ratio=0.5)
        super().train(instances)