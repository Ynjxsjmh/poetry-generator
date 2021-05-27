import tensorflow as tf

from processor import PoetryDataset, PoetryProcessor, PoetryUtil, Tokenizer


if __name__ == "__main__":
    poetryDataset = PoetryDataset()
    tang_poetries = poetryDataset.get_tang_poetries()

    poetryProcessor = PoetryProcessor(tang_poetries)
    token2id = poetryProcessor.convert_poetries_to_token2id()

    tokenizer = Tokenizer(token2id)

    model = tf.keras.models.load_model('./model/best_model.h5')

    poetryUtil = PoetryUtil(model, tokenizer)

    print('随机生成')
    for i in range(10):
        print(poetryUtil.generate_random_poetry())

    print('\n\n续写诗')
    for i in range(10):
        print(poetryUtil.generate_random_poetry('開窗放入大江來，'))

    # 生成藏头诗
    print('\n\n藏头诗')
    for i in range(10):
        print(poetryUtil.generate_acrostic_poetry('神經網絡'))
