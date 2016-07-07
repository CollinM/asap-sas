from collections import defaultdict, Counter
from math import log10
from asap.core import Base, SparseVector


class Tokenizer(Base):

    def __init__(self, sep=' ', key='tokens'):
        super().__init__(key)
        self._sep = sep

    def process(self, instance):
        tokens = instance.text.split(self._sep)
        instance.add_feature(self.key, tokens)
        return instance


class TFIDF(Base):

    def __init__(self, topk=None, idf=None, token_map=None, key='tfidf'):
        super().__init__(key, True)

        self._topk = topk
        self._idf = idf
        self._token_index_map = token_map

    def process(self, instance):
        term_counts = Counter(instance.get_feature('tokens'))
        vector = SparseVector(len(self._token_index_map))
        for tok in term_counts.elements():
            if tok in self._token_index_map:
                vector[self._token_index_map[tok]] = term_counts[tok] * self._idf[tok]
        instance.add_feature(self.key, vector)

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


class BagOfWords(Base):

    def __init__(self, min_occur=1, key='bow'):
        super().__init__(key, True)

        assert isinstance(min_occur, int)
        self._min_occur = min_occur
        self._bow_lookup = None

    def process(self, instance):
        vector = SparseVector(len(self._bow_lookup))
        for token in instance.get_feature('tokens'):
            if token in self._bow_lookup:
                vector[self._bow_lookup[token]] = 1

        instance.add_feature(self.key, vector)

        return instance

    def train(self, instances):
        assert isinstance(instances, list)

        # Count token occurrences
        counter = Counter()
        for inst in instances:
            counter.update(inst.get_feature('tokens'))

        self._bow_lookup = {}
        n = 0
        for token, count in counter.most_common():
            if count >= self._min_occur:
                self._bow_lookup[token] = n
                n += 1
            else:
                break


class UniqueWordCount(Base):
    """Count the unique words in an instance's text."""

    def __init__(self, key="unique-word-count"):
        super().__init__(key)

    def process(self, instance):
        instance.add_feature(self.key, len(set(instance.get_feature("tokens").to_list())))
        return instance


class WordCount(Base):
    """Coutn all of the words (tokens) in an instance's text."""

    def __init__(self, key="word-count"):
        super().__init__(key)

    def process(self, instance):
        instance.add_feature(self.key, len(instance.get_feature("tokens")))
        return instance


class CharacterCount(Base):
    """Count all of the characters in an instance's text."""

    def __init__(self, key="char-count"):
        super().__init__(key)

    def process(self, instance):
        instance.add_feature(self.key, len(instance.text))
        return instance


class NonWhitespaceCharacterCount(Base):
    """Count all of the non-whitespace characters in an instance's text."""

    def __init__(self, key="!white-char-count"):
        super().__init__(key)

    def process(self, instance):
        instance.add_feature(self.key, len(instance.text.replace(' ', '')))
        return instance


class ContainsWords(Base):
    """Create word presence vector based on a list of words."""

    def __init__(self, word_list_path, key='word-presence'):
        super().__init__(key)
        self._words = {}
        with open(word_list_path) as f:
            for i, line in enumerate(f.readlines()):
                self._words[line.strip().lower()] = i

    def process(self, instance):
        vec = SparseVector(len(self._words))
        for tok in instance.get_feature('tokens'):
            if tok in self._words:
                vec[self._words[tok]] = 1
        instance.add_feature(self.key, vec)
        return instance
