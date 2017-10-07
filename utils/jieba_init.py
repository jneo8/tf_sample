"""Jieba Init."""
import os
import json
import jieba

from .settings import SOURCE_DIR
from logger import logconf

logger = logconf.Logger(__name__)


def jieba_init():
    """Read dict from source_dir."""
    try:
        # Load userdict
        jieba_dict_paths = [
            os.path.join(SOURCE_DIR, 'dictionary', filename)
            for filename in os.listdir(os.path.join(SOURCE_DIR, 'dictionary'))
            if filename.endswith('.txt')
        ]
        for path in jieba_dict_paths:
            jieba.load_userdict(path)
            logger.info(f'Jieba load userdict {path}')

        # Load stopwords
        stop_word_path = os.path.join(SOURCE_DIR, 'dictionary', 'stopwords.json')

        stopword_set = set()
        with open(stop_word_path, 'r') as csv:
            stop_words = json.load(csv)
            for word in stop_words:
                stopword_set.add(word)

        logger.info(f'Jieba load stop word {stop_word_path}')
        jieba.initialize()
        logger.info('Init Jieba')
    except:
        import traceback
        logger.error(traceback.format_exc())

jieba_init()

