import numpy as np


class PoetryUtil:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, token_ids):
        """
        在概率值为前100的词中选取一个词(按概率分布的方式)
        :return: 一个词的编号(不包含[PAD][NONE][START])
        """

        # 预测各个词的概率分布，不包含 [PAD][NONE][START]
        probs = self.model.predict([token_ids, ])[0, -1, 3:]
        probs = np.squeeze(probs)

        # 先按概率升序排序，取后 100，然后降序
        p_idxes = probs.argsort()[-100:]

        # 根据索引找到具体的概率值
        p = [probs[idx] for idx in p_idxes]
        p = p / sum(p) # 归一

        # 按概率抽取一个
        # 前面预测时删除了前几个标记符，因此编号要补上3位，才是实际在tokenizer词典中的编号
        return np.random.choice(p_idxes, p=p) + 3

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
            if target == self.tokenizer.token_to_id('[END]'):
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
                if pid > 3:
                    # 保存结果到token_ids中，下一次预测还要用
                    token_ids.append(pid)

        return "".join(self.tokenizer.decode(token_ids))
