"""Word2vec with tensorflow."""

import os
import re
import collections
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import settings
from utils.jieba_init import jieba

# logger
from logger import logconf
logger = logconf.Logger(__name__)


# Init data_index
data_index = 0


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
        - reversed_dictionary
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
        # logger.debug(f'word: {word} count_num: {_} code: {dictionary[word]}')

    data = []
    unk_count = 0

    for word in words:
        idx = dictionary.get(word, 0)
        if idx == 0:  # word not in count -> unk word.
            unk_count += 1
        data.append(idx)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

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
    vocabulary_size = 100000
    data, count, dictionary, reversed_dictionary = build_dataset(
            words=texts, n_words=vocabulary_size)
    del texts  # Hint to reduce memory
    logger.info(f'Most common words (+UNK) {count[:5]}')
    for x in range(0, 10):
        logger.debug(f'{data[x]} {reversed_dictionary[data[x]]}')


    ############################
    # Step 3: Skip-gram model
    ############################

    # Init data_index
    data_index = 0

    def generate_batch(batch_size, num_skips, skip_window):
        """Generate a train batch fpr the skip-gram model."""
        global data_index
        logger.debug(f'data_index {data_index}')
        # Check
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [skip_window target skip_windom]
        buffer = collections.deque(maxlen=span)
        if data_index + span > len(data):
            data_index += span
        buffer.extend(data[data_index:data_index + span])
        data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = collections.deque(context_words)
            logger.debug(words_to_use)
            for j in range(num_skips):
                batch[i * num_skips + j] = buffer[skip_window]
                context_word = words_to_use.pop()
                labels[i * num_skips + j, 0] = buffer[context_word]
            if data_index == len(data):
                buffer[:] = data[:span]
                data_index = span
            else:
                buffer.append(data[data_index])
                data_index += 1

        # Backtrack a little bit to avoid skipping words in the end of a batch

        data_index = (data_index + len(data) - span % len(data))
        logger.debug(f'batch: {batch} labels: {labels}')
        return batch, labels

    #########################
    # Step 4: Build and train a skip-gram model.
    #########################

    batch_size = 128
    embedding_size = 128    # Dimension of the embedding vector.
    skip_window = 1         # How many word to consider left and right.
    num_skips = 2           # How many times to reuse an input to generate a label.
    num_sampled = 64        # Number of negative examples to sample.


    # We pick a random validation set to sample nearest neighbors.
    # Here we limit the validation samples to the words that have a low numeric ID,
    # which by construction are also the most frequnt.
    # These 3 variables are used to only for displaying model accuracy,
    # They don't affect calculation.

    valid_size = 16
    valid_window = 100
    valid_examples = np.random.choice(valid_window, valid_size, replace=True)

    graph = tf.Graph()

    with graph.as_default():
        # Input data
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and valid pinned to the cpu because of missing GPU implemention

        with tf.device('/cpu:0'):
            # Look up embedding for input.
            embedding = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
            )





if __name__ == '__main__':
    main()

