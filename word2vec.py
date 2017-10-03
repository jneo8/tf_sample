"""Word2vec with tensorflow."""

import os
import pandas as pd
from utils import settings
import jieba

# logger
from logger import logconf
logger = logconf.Logger(__name__)


def read_data():
    """Read origin data use pandas.

    Return a list of sentence.
    """
    source_list = [
        '獸性老公吻上癮.txt',
        '惡魔總裁的小妻子.txt',
    ]
    source_list = [os.path.join(settings.SOURCE_DIR, s) for s in source_list]

    list_ = []
    for s in source_list:
        logger.debug(s)
        df_ = pd.read_csv(s, sep=" ", header=None, names=['text'])
        list_.append(df_)
    df = pd.concat(list_)
    logger.debug(df.shape)
    return df['text'].tolist()

def cut_sentence(x):
    """Cut sentence use jieba."""
    for row in x:
        row = [jieba.cut(x) ]
        row = [r for r in row if r is not ' ']
    return x

def main():
    """Main."""
    texts = read_data()
    texts = cut_sentence(x=texts)
    logger.debug(texts[:5])


if __name__ == '__main__':
    main()

