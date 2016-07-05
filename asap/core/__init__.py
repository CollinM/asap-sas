from collections import defaultdict
import random
from math import floor
import pickle
import os


class Instance(object):

    def __init__(self, identifier, text):
        self._id = identifier
        self._text = text
        self._features = defaultdict(lambda: SparseVector(0))

    @property
    def text(self):
        """

        :str: text of the instance
        """
        return self._text

    @property
    def id(self):
        """

        :str: unique identifier
        """
        return self._id

    def add_feature(self, name, value):
        assert isinstance(name, str)
        if not (isinstance(value, DenseVector) or isinstance(value, SparseVector)):
            if isinstance(value, list):
                value = DenseVector(contents=value)
            else:
                value = DenseVector(contents=[value])
        self._features[name] = value

    def get_feature(self, name):
        return self._features[name]


class Base(object):

    def __init__(self, key):
        self._trainable = False
        assert isinstance(key, str)
        self._key = key

    @property
    def is_trainable(self):
        return self._trainable

    @property
    def key(self):
        return self._key

    def process(self, instance):
        raise NotImplementedError("Run is not implemented!")


class Model(Base):

    def __init__(self, target, features, key="prediction"):
        super().__init__(key)
        self._trainable = True
        self._target = target
        self._features = features

    def collect_features(self, instance):
        x = []
        for feat in self._features:
            x.extend(instance.get_feature(feat).to_list())
        return x

    def get_target(self, instance):
        return instance.get_feature(self._target)[0]

    def train(self, instances):
        raise NotImplementedError("Train is not implemented!")


class Pipeline(object):

    def __init__(self):
        self._phases = []

    def add_phase(self, component):
        self._phases.append(component)

    def train(self, instances):
        for p in self._phases:
            if p.is_trainable:
                p.train(instances)
            instances = list(map(p.process, instances))
        return instances

    def run(self, instances):
        for p in self._phases:
            instances = list(map(p.process, instances))
        return instances

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class DenseVector(object):

    def __init__(self, size=None, contents=None):
        if size is not None and contents is None:
            self._vec = [0 for i in range(size)]
        else:
            assert isinstance(contents, list)
            self._vec = contents

    def __len__(self):
        return len(self._vec)

    def __getitem__(self, key):
        return self._vec[key]

    def __setitem__(self, key, value):
        self._vec[key] = value

    def __iter__(self):
        return iter(self._vec)

    def to_list(self):
        return self._vec[:]


class SparseVector(object):

    def __init__(self, size, default=lambda: 0):
        self._vec = defaultdict(default)
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, key):
        return self._vec[key]

    def __setitem__(self, key, value):
        self._vec[key] = value

    def __iter__(self):
        return (self._vec[i] for i in range(self._size))

    def to_list(self):
        return list(self.__iter__())


def load_instances(filename):
    """Load a list of Instances from 'filename'"""
    instances = []
    with open(filename) as f:
        for line in [l.strip() for l in f.readlines()]:
            ident, score1, score2, text = line.split("\t")
            inst = Instance(ident, text)
            inst.add_feature('score1', int(score1))
            inst.add_feature('score2', int(score2))
            instances.append(inst)
    return instances


def split_instances(instances, portion, seed=None):
    """Split a list of Instances into training and testing sets, respectively."""
    random.seed(seed)
    insts = instances[:]
    random.shuffle(insts)
    split_index = floor(len(insts) * portion) + 1
    return insts[:split_index], insts[split_index:]


def gather_input_files(input_path):
    """Return list of input tuples: (input number, full filename)"""
    inputs = []
    for fname in os.listdir(input_path):
        full_name = os.path.join(input_path, fname)
        dot_idx = fname.find('.')
        num = fname[dot_idx - 2:dot_idx]
        inputs.append((num, full_name))
    return inputs
