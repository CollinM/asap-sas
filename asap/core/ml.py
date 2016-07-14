from asap.core import Model
from asap.metrics.nn import quadratic_weighted_kappa_loss

import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.linear_model import LogisticRegression as LogReg

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, GRU


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


class LSTM_Arch1(Model):

    def __init__(self, lstm_output_size=512, batch_size=32, num_epochs=20, **kwargs):
        super().__init__(**kwargs)
        self._lstm_output_size = lstm_output_size
        self._batch_size = batch_size
        self._epochs = num_epochs
        self._model = None

    def process(self, instance):
        input_data = []
        for feat in instance.get_feature(self._features[0]):
            input_data.append(feat)

        prdxns = self._model.predict_classes(np.array(input_data))
        prdxn_counts = Counter(prdxns)

        ranked = prdxn_counts.most_common()
        if len(ranked) == 1:
            prediction = ranked[0][0]
        else:
            if ranked[0][1] > ranked[1][1]:
                prediction = ranked[0][0]
            else:  # counts are equal
                prediction = min(ranked[0][0], ranked[1][0])

        instance.add_feature(self.key, prediction)

        return instance

    def train(self, instances):
        # Convert the gold standard labels
        scores = set()
        for inst in instances:
            scores.add(self.get_target(inst))
        score_size = max(scores) + 1

        # Get the input data
        X = []
        y = []
        for inst in instances:
            target = np.zeros(score_size)
            target[self.get_target(inst)] = 1
            for feat in inst.get_feature(self._features[0]):
                X.append(feat)
                y.append(target)

        # Iteratively try to get the input sizes in case of degenerate input...
        for i in range(3):
            try:
                sample_input = instances[i].get_feature(self._features[0])[0]
                input_dim = len(sample_input[0])
                input_length = len(sample_input)
                break
            except KeyError:
                continue

        # Create model
        self._model = Sequential()
        self._model.add(LSTM(self._lstm_output_size, input_dim=input_dim, input_length=input_length))
        self._model.add(Dense(score_size))
        self._model.add(Activation('sigmoid'))
        self._model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # Train
        self._model.fit(X, np.array(y), batch_size=self._batch_size, nb_epoch=self._epochs, verbose=2)
