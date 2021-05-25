import numpy as np
import tensorflow as tf

from processor import PoetryDataset, PoetryProcessor, Tokenizer


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
        padding = padding if padding else self.tokenizer.token_to_id('[PAD]')

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
            batch_data = self.pad_token_ids_list(batch_data, self.tokenizer.token_to_id('[PAD]'))

            yield batch_data[:, 1:], tf.one_hot(batch_data[:, 1:] + [self.tokenizer.token_to_id('[END]'),], self.tokenizer.token_num)
            del batch_data

    def for_fit(self):
        """
        创建一个生成器，用于训练
        """
        # 死循环，当数据训练一个epoch之后，重新迭代数据
        while True:
            # 委托生成器
            yield from self.__iter__()


class PoetryModel:
    def __init__(self, poetries, tokenizer):
        self.poetries = poetries
        self.tokenizer = tokenizer

        self.model = self.init_model()

    def init_model(self):
        model = tf.keras.Sequential([
            # 不定长度的输入(optional)
            tf.keras.layers.Input((None,)),
            # 词嵌入层
            tf.keras.layers.Embedding(input_dim=self.tokenizer.token_num, output_dim=150),
            tf.keras.layers.LSTM(150, dropout=0.5, return_sequences=True),
            tf.keras.layers.LSTM(150, dropout=0.5, return_sequences=True),
            # 利用 TimeDistributed 对每个时间步的输出都做 Dense 操作(softmax 激活)
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.tokenizer.token_num, activation='softmax')),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.categorical_crossentropy
        )

        return model

    def train(self):
        training_generator = PoetryDataGenerator(self.poetries, self.tokenizer)

        self.model.fit(
            training_generator.for_fit(),
            steps_per_epoch=training_generator.steps,
            epochs=10,
        )

    def predict(self):
        pass


if __name__ == "__main__":
    poetryDataset = PoetryDataset()
    tang_poetries = poetryDataset.get_tang_poetries()

    poetryProcessor = PoetryProcessor(tang_poetries)
    token2id = poetryProcessor.convert_poetries_to_token2id()

    tokenizer = Tokenizer(token2id)

    poetryModel = PoetryModel(tang_poetries, tokenizer)
    poetryModel.train()
