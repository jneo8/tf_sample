"""Word2vec with tensorflow."""

import os
import re
import collections
import pandas as pd
from utils import settings
from utils.jieba_init import jieba

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

###########################
#  Preprocess method
##########################
def cut_sentence(x):
    """Cut sentence use jieba."""
    list_ = []
    for row in x:
        row = list(jieba.cut(row))
        row = [r for r in row if r not in ['\u3000', ' ']]
        list_ += (['RowMark'] + row)
    return list_

def clean_doc(text):
    """Remove symbol."""
    text = str(text)
    # logger.debug(f'text before clean {text}')
    symbol_1 = r'https://[a-zA-Z0-9.?/&=:]*'
    symbol_2 = '[’"#$%&\'()*+,-./:;<=>@[\\]^_`{|}]+'
    symbol_3 = '[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕】【-]+'
    symbol_4 = r'\n'
    text_ = re.sub(symbol_1, ' ', text)
    text_ = re.sub(symbol_2, ' ', text_)
    text_ = re.sub(symbol_3, ' ', text_)
    # logger.debug(f'text after clean {text_}')
    return text_

def build_dataset(words, n_words):
    """Process raw input into a dateset.

    Filling 4 global variables:
        - data:
            List of codes (integers from 0 to n_words - 1).
            This is the original text but words are replaces by their codes
        - count
            Map of words to count of occurences.
        - dictionary
            Map of words to their codes(integers).
        - reverse_dictionary
            Map codes(integers) to words(strings)
    """
    # Count word use collections
    # https://docs.python.org/2/library/collections.html
    count = [['UNK', -1]]
    logger.debug(type(count))
    count.extend(collections.Counter(words).most_common(n_words -1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)  # The code of word is the idx in count.
        logger.debug(f'word: {word} count_num: {_} code: {dictionary[word]}')

    data = []
    unk_count = 0

    for word in words:
        idx = dictionary.get(word, 0)
        if idx == 0:  # word not in count -> unk word.
            unk_count += 1
        data.append(idx)
    count[0][1] = unk_count
    reversed_dictionary = dict((zip(dictionary.values(), dictionary.keys())))
    return data, count, dictionary, reverse_dictionary

def main():
    """Main."""
    ###########################
    #  Step 1 :
    #       - Read Data, original data, a list of sentences
    #       - clean sentence, remove special symbol.
    #       - Cut the sentence, example: 你真的好棒 -> 你/真的/好棒
    ##########################
    texts = read_data()
    texts = texts[:1000]
    for idx, text in enumerate(texts):
        text = clean_doc(text=text)
        texts[idx] = text

    texts = cut_sentence(x=texts)

    ##########################
    # Step 2 :
    #       - Build the dictionary and rare words with UNK token.
    #########################
    vocabulary_size = 10
    build_dataset(words=texts, n_words=vocabulary_size)





if __name__ == '__main__':
    main()

