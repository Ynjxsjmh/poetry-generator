import json
import glob
import collections
import configparser
import numpy as np


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


class PoetryUtil:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # 除了 END 之外的特殊字符数量
        self.ignored_token_num = len(SPECIAL_TOKEN) - 1

    def predict(self, token_ids):
        """
        在概率值为前100的词中选取一个词(按概率分布的方式)
        :return: 一个词的编号(不包含[PAD][NONE][START])
        """

        # 预测各个词的概率分布，不包含 [PAD][NONE][START]
        probs = self.model.predict([token_ids, ])[0, -1, self.ignored_token_num:]
        probs = np.squeeze(probs)

        # 先按概率升序排序，取后 100，然后降序
        p_idxes = probs.argsort()[-100:]

        # 根据索引找到具体的概率值
        p = [probs[idx] for idx in p_idxes]
        p = p / sum(p) # 归一

        # 按概率抽取一个
        # 前面预测时删除了前几个标记符，因此编号要补上3位，才是实际在tokenizer词典中的编号
        return np.random.choice(p_idxes, p=p) + self.ignored_token_num

    def generate_random_poetry(self, text=""):
        """
        随机生成一首诗
        """
        # 将初始字符串转成 token_ids，并去掉结束标记
        token_ids = self.tokenizer.encode(text)[:-1]

        while True:
            # 预测词的编号
            target = self.predict(token_ids)

            # 保存结果
            token_ids.append(target)

            # 到达END
            if target == self.tokenizer.token_to_id(SPECIAL_TOKEN['END']):
                break

        return "".join(self.tokenizer.decode(token_ids))

    def generate_acrostic_poetry(self, heads):
        """
        生成一首藏头诗
        :param heads: 藏头诗的头
        :return: 一首古诗的字符串
        """
        # token_ids，只包含[START]编号
        token_ids = self.tokenizer.encode('')[:-1]
        punc_ids = [self.tokenizer.token_to_id(punc)
                    for punc in ['，', '。']]

        content = []
        for head in heads:
            # head转为编号id，放入列表，用于预测
            token_ids.append(self.tokenizer.token_to_id(head))

            pid = -1
            # 开始生成一句诗
            # 遇到逗号、句号，说明本句结束，开始下一句
            while pid not in punc_ids:
                pid = self.predict(token_ids)

                # 只有不是特殊字符时，才保存到poetry里面去
                if pid > self.ignored_token_num:
                    # 保存结果到token_ids中，下一次预测还要用
                    token_ids.append(pid)

        return "".join(self.tokenizer.decode(token_ids))


if __name__ == "__main__":
    poetryDataset = PoetryDataset()
    tang_poetries = poetryDataset.get_tang_poetries()

    print(len(tang_poetries))

    poetryProcessor = PoetryProcessor(tang_poetries)
    token2id = poetryProcessor.convert_poetries_to_token2id()

    print(token2id)
