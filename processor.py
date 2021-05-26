import json
import glob
import collections
import configparser


class PoetryDataset:
    def __init__(self):
        pass

    def get_tang_poetries(self):
        poetries = []
        filenames = glob.glob('./data/*tang*.json')

        for filename in filenames:
            with open(filename) as f:
                poetries.extend([''.join(poetry['paragraphs'])
                                 for poetry in json.load(f)])

        return poetries


SPECIAL_TOKEN = {
    'SRT': '[SRT]',
    'END': '[END]',
    'UNK': '[UNK]',
    'PAD': '[PAD]'
}


class PoetryProcessor:
    def __init__(self, poetries):
        self.poetries = poetries

        config = configparser.ConfigParser()
        config.read('config.ini')

        self.filters = config['filter']

    def filter_by_poetry(self, poetries=None):
        poetries = poetries if poetries else self.poetries

        poetries = [poetry
                    for poetry in poetries
                    if len(poetry) > int(self.filters['MAX_POETRY_LEN']) ]
        poetries = [poetry
                    for poetry in poetries
                    if not any(symbol in poetry for symbol in json.loads(self.filters['DISALLOWED_WORDS'])) ]

        return poetries

    def filter_by_word_count(self, words_count):
        return {word: count
                for word, count in words_count.items()
                if count > int(self.filters['MIN_WORD_FREQUENCY'])}

    def convert_poetries_to_token2id(self, poetries=None):
        poetries = poetries if poetries else self.poetries
        poetries = self.filter_by_poetry(poetries)

        words_count = collections.Counter(''.join(poetries))
        words_count = self.filter_by_word_count(words_count)

        word_count_tuple = sorted([(word, count) for word, count in words_count.items()],
                                  key=lambda x: -x[1])

        tokens = [word for word, _ in word_count_tuple]
        tokens = [SPECIAL_TOKEN['PAD'], SPECIAL_TOKEN['UNK'],
                  SPECIAL_TOKEN['SRT'], SPECIAL_TOKEN['END']] + tokens

        return dict(zip(tokens, range(len(tokens))))


class Tokenizer:

    def __init__(self, token2id):
        self.token2id = token2id
        self.id2token = {id: token for token, id in self.token2id.items()}

        self.token_num = len(self.token2id)

    def id_to_token(self, token_id):
        return self.id2token[token_id]

    def token_to_id(self, token):
        return self.token2id.get(token, self.token2id[SPECIAL_TOKEN['UNK']])

    def encode(self, tokens):
        """
        给定一个字符串s，在头尾分别加上标记开始和结束的特殊字符，并将它转成对应的编号序列
        :param tokens: 待编码字符串
        :return: 编号序列
        """
        tokens = [SPECIAL_TOKEN['SRT'],] + list(tokens) + [SPECIAL_TOKEN['END'],]

        return [self.token_to_id(token) for token in tokens]

    def decode(self, token_ids):
        """
        给定一个编号序列，将它解码成字符串
        :param token_ids: 待解码的编号序列
        :return: 解码出的字符串
        """
        special_tokens = {SPECIAL_TOKEN['SRT'], SPECIAL_TOKEN['END']}

        tokens = [self.id_to_token(token_id) for token_id in token_ids]
        tokens = [token for token in tokens if token not in special_tokens]

        return ''.join(tokens)


if __name__ == "__main__":
    poetryDataset = PoetryDataset()
    tang_poetries = poetryDataset.get_tang_poetries()

    print(len(tang_poetries))

    poetryProcessor = PoetryProcessor(tang_poetries)
    token2id = poetryProcessor.convert_poetries_to_token2id()

    print(token2id)
