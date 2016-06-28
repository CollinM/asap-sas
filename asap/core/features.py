from collections import defaultdict, Counter
from math import log10
from asap.core import Base, SparseVector


class Tokenizer(Base):

    def __init__(self, sep=' '):
        super().__init__()
        self._sep = sep

    def process(self, instance):
        tokens = instance.text.split(self._sep)
        instance.add_feature('tokens', tokens)
        return instance


class TFIDF(Base):

    def __init__(self, topk=None, idf=None, token_map=None):
        super().__init__()
        self._trainable = True

        self._topk = topk
        self._idf = idf
        self._token_index_map = token_map

    def process(self, instance):
        term_counts = Counter(instance.get_feature('tokens'))
        vector = SparseVector(len(self._token_index_map))
        for tok in term_counts.elements():
            if tok in self._token_index_map:
                vector[self._token_index_map[tok]] = term_counts[tok] * self._idf[tok]
        instance.add_feature('tfidf', vector)

        return instance

    def train(self, instances):
        """

        :Instance instances: list of instances to train with
        """
        assert isinstance(instances, list)

        total_docs = float(len(instances))
        # Gather all token occurrences in documents
        term_docs = defaultdict(set)
        for inst in instances:
            for tok in inst.get_feature('tokens'):
                term_docs[tok].add(inst.id)

        # Calculate IDF and TFIDF value for each token and sort by TFIDF value (high to low)
        idf_vals = {}
        tfidf_vals = []
        for tok in term_docs.keys():
            idf_vals[tok] = log10(total_docs / (len(term_docs[tok]) + 1))
            tfidf_vals.append((tok, (len(term_docs[tok]) + 1) * idf_vals[tok]))
        tfidf_vals.sort(key=lambda x: x[1], reverse=True)

        # Assign cutoff point
        if self._topk is not None:
            limit = self._topk
        else:
            limit = len(idf_vals)

        # Save IDF values and vector indices for values that make the cut
        idf = {}
        index_map = {}
        for i, tup in enumerate(tfidf_vals[:limit]):
            idf[tup[0]] = idf_vals[tup[0]]
            index_map[tup[0]] = i

        self._idf = idf
        self._token_index_map = index_map
