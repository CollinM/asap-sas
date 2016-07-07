from asap.core import Base
import re


class PunctuationStripper(Base):

    def __init__(self):
        super().__init__('')
        self._DOT = re.compile("[\\.]")
        self._STRIP_PUNC = re.compile("[\\,\\!\\?\\\"\\']")

    def process(self, instance):
        s = self._STRIP_PUNC.sub('', instance.text)
        instance._text = self._DOT.sub(' ', s)
        return instance


class WhitespaceNormalizer(Base):

    def __init__(self):
        super().__init__('')
        self._WS = re.compile("\\s+")

    def process(self, instance):
        instance._text = self._WS.sub(' ', instance.text).strip()
        return instance


class LowerCaser(Base):

    def __init__(self):
        super().__init__('')

    def process(self, instance):
        instance._text = instance.text.lower()
        return instance
