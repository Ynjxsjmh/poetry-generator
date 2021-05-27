import numpy as np
import tensorflow as tf

from processor import SPECIAL_TOKEN


class PoetryGenerator(tf.keras.utils.Sequence):
    def __init__(self, poetries, tokenizer, batch_size=16, shuffle=True):
        self.poetries = poetries

        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def pad_token_ids_list(self, token_ids_list, padding=' '):
        max_len = max(map(len, token_ids_list))

        return np.array([token_ids + [padding]*(max_len-len(token_ids))
                         for token_ids in token_ids_list])

    def __len__(self):
        return int(np.floor(len(self.poetries) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of poetries'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_poetries = [self.poetries[idx] for idx in indexes]

        batch_data = [self.tokenizer.encode(poetry) for poetry in batch_poetries]
        batch_data = self.pad_token_ids_list(batch_data, self.tokenizer.token_to_id(SPECIAL_TOKEN.PAD))

        return batch_data[:, :-1], tf.one_hot(batch_data[:, 1:], self.tokenizer.token_num)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.poetries))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class PoetryDataGenerator:
    def __init__(self, poetries, tokenizer, batch_size=16, shuffle=True):
        self.poetries = poetries
        self.tokenizer = tokenizer

        # batch size
        self.batch_size = batch_size
        # 每个epoch迭代的步数
        self.steps = int(np.floor(len(self.poetries) / self.batch_size))
        # 每个epoch开始时是否随机混洗
        self.shuffle = shuffle

    def pad_token_ids_list(self, token_ids_list, padding=None):
        padding = padding if padding else self.tokenizer.token_to_id(SPECIAL_TOKEN['PAD'])

        max_len = max(map(len, token_ids_list))

        return np.array([token_ids + [padding]*(max_len-len(token_ids))
                         for token_ids in token_ids_list])

    def __len__(self):
        return self.steps

    def __iter__(self):
        if self.shuffle == True:
            np.random.shuffle(self.poetries)

        total = len(self.poetries)

        for start in range(0, total, self.batch_size):
            end = min(start+self.batch_size, total)

            batch_data = [self.tokenizer.encode(poetry) for poetry in self.poetries[start:end]]
            batch_data = self.pad_token_ids_list(batch_data, self.tokenizer.token_to_id(SPECIAL_TOKEN['PAD']))

            end = self.tokenizer.token_to_id(SPECIAL_TOKEN['END'])
            batch_label = [np.append(data, end) for data in batch_data[:, 1:]]
            yield batch_data, tf.one_hot(batch_label, self.tokenizer.token_num)

            del batch_data

    def for_fit(self):
        """
        创建一个生成器，用于训练
        """
        # 死循环，当数据训练一个epoch之后，重新迭代数据
        while True:
            # 委托生成器
            yield from self.__iter__()
