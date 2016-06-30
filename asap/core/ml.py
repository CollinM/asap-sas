from asap.core import Model
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class RandomForest(Model):

    def __init__(self, model=None, num_trees=100, max_depth=None, threads=1, **kwargs):
        super().__init__(**kwargs)
        self._trainable = True
        self._num_trees = num_trees
        self._max_depth = max_depth
        self._threads = threads
        self._model = model

    def process(self, instance):
        features = np.array(self.collect_features(instance)).reshape((1, -1))
        prdxn = self._model.predict(features)[0]
        instance.add_feature(self.key, prdxn)
        return instance

    def train(self, instances):
        # Create model
        self._model = RandomForestClassifier(n_estimators=self._num_trees, max_depth=self._max_depth, n_jobs=self._threads)
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
