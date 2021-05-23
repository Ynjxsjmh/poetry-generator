import json
import glob
import collections
import configparser


class PoetryDataset:
    def __init__(self):
        pass

    def get_tang(self):
        poetries = []
        filenames = glob.glob('./data/*tang*.json')

        for filename in filenames:
            with open(filename) as f:
                poetries.extend([''.join(poetry['paragraphs'])
                                 for poetry in json.load(f)])

        return poetries


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
        tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + tokens

        return dict(zip(tokens, range(len(tokens))))


if __name__ == "__main__":
    poetryDataset = PoetryDataset()
    tang_potries = poetryDataset.get_tang()

    print(len(tang_potries))

    poetryProcessor = PoetryProcessor(tang_potries)
    token2id = poetryProcessor.convert_poetries_to_token2id()

    print(token2id)
