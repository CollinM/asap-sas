import numpy as np
from .quadratic_weighted_kappa import quadratic_weighted_kappa


def write_qwk(actuals, predxns, filepath):
    """Write quadratic weighted kappa based to `filepath`, and return the metric."""
    qwk = quadratic_weighted_kappa(actuals, predxns)
    with open(filepath, 'w') as f:
        f.write(str(qwk))
    return qwk


def write_results(result_triples, filepath):
    """Write ID-gold-prediction result triples to `filepath`."""
    with open(filepath, 'w') as f:
        for t in result_triples:
            f.write(','.join(t) + '\n')


class ConfusionMatrix(object):
    """First index is actual, second is predicted."""

    def __init__(self, class_list):
        self._classes = [str(c) for c in class_list]
        self._classMap = {name: i for i, name in enumerate(self._classes)}
        self._matrix = np.zeros((len(self._classes), len(self._classes)))

    def increment(self, actual, predicted):
        if isinstance(actual, str) and isinstance(predicted, str):
            self._increment_str(actual, predicted)
        elif isinstance(actual, int) and isinstance(predicted, int):
            self._increment_int(actual, predicted)
        else:
            raise TypeError("Arguments must both be str or int!")

    def _increment_str(self, actual, predicted):
        a = self._classMap[actual]
        p = self._classMap[predicted]
        self._increment_int(a, p)

    def _increment_int(self, actual, predicted):
        self._matrix[actual][predicted] += 1

    def precision(self, label=None):
        if label is None:
            return sum([self.precision(c) * self._class_ratio(c) for c in self._classes])
        else:
            index = self._classMap[label]
            tp = self._matrix[index][index]
            fp = (tp * -1) + sum([self._matrix[i][label] for i in len(self._classes)])
            return 0 if tp + fp == 0 else tp / (tp + fp)

    def recall(self, label=None):
        if label is None:
            return sum([self.recall(c) * self._class_ratio(c) for c in self._classes])
        else:
            index = self._classMap[label]
            tp = self._matrix[index][index]
            fn = (tp * -1) + sum(self._matrix[index])
            return 0 if tp + fn == 0 else tp / (tp + fn)

    def f1(self, label=None):
        if label is None:
            p = self.precision()
            r = self.recall()
        else:
            p = self.precision(label)
            r = self.recall(label)
        return 2 * ((p * r) / (p + r))

    def _class_ratio(self, label):
        label_count = sum(self._matrix[self._classMap[label]])
        all_count = sum([sum(self._matrix[i]) for i in len(self._classes)])
        return label_count / all_count

    def write_csv(self, filepath):
        with open(filepath, 'w') as f:
            f.write(self.to_csv())

    def to_csv(self):
        out = ',' + ','.join(self._classes) + '\n'
        for c in self._classes:
            class_index = self._classMap[c]
            out += c + ','
            out += ','.join([str(float(self._matrix[class_index][i])) for i in range(len(self._classes))]) + '\n'
        return out
