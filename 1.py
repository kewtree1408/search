#! /usr/bin/python
# coding: utf-8

import requests
import html5lib
import string
import logging
from nltk import wordpunct_tokenize
from nltk.stem.snowball import RussianStemmer
import collections

rus_stemmer = RussianStemmer()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('stats.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

def is_punctuation(token):
    for punct in string.punctuation:
        if punct in token:
            return True
    return False

# получаем уникальные токены для текущего документа
def get_tokens(text):
    tokens = set()
    sum_length = 0.0
    for t in wordpunct_tokenize(text):
        if not is_punctuation(t):
            tokens.add(t) # check

    sum_length = sum(len(t) for t in tokens)
    count = len(tokens)
    return tokens, count, sum_length

# получаем уникальные термы для текущего документа
def get_terms(tokens):
    terms = set()
    sum_length = 0.0
    for t in tokens:
        term = rus_stemmer.stem(t)
        terms.add(term)

    sum_length = sum(len(t) for t in terms)
    count = len(terms)
    return terms, count, sum_length

def get_reverse_index(html, path_value):
    doc_id = 0
    pairs_seq = set()
    non_uniq_tokens = list()
    count_tokens, count_terms = 0, 0
    sum_len_tokens, sum_len_terms = 0.0, 0.0
    for dd in html.xpath(path_value):
        # поиск параграфов
        for p in dd.xpath('.//p/*'):
            doc_id += 1

        # токенизация
        tokens, tok_len, sum_len_token = get_tokens(dd.text)
        count_tokens += tok_len
        sum_len_tokens += sum_len_token
        non_uniq_tokens.extend(tokens)

        # терминизация
        terms, term_len, sum_len_term = get_terms(tokens)
        count_terms += term_len
        sum_len_terms += sum_len_term

        # получаем последовательность пар (термин, doc_id)
        for term in terms:
            pairs_seq.add((term, doc_id))

    # слияние термов, получение словаря
    index_dict = collections.defaultdict(lambda: (0, list()))
    for term, docid in pairs_seq:
        freq, docids = index_dict[term]
        # запоминаем частоту
        index_dict[term] = freq + 1, sorted(docids + [docid])

    # обратный индекс
    rindex = sorted(index_dict.items())

    # собираем характеристики
    uniq_tokens = set(non_uniq_tokens)
    count_uniq_tokens = len(uniq_tokens)
    count_uniq_terms = len(rindex)

    # 1. сам индекс
    for t, (f, d) in rindex:
        logger.debug("term = %s, freq = %d, docId = %s", t, f, repr(d))

    # 2. количество и средняя длина токенов и термов во всем тексте
    logger.info("in text: count_tokens = %d, avg_len_tokens = %f", count_tokens, sum_len_tokens/count_tokens)
    logger.info("in text: count_terms = %d, avg_len_terms = %f", count_terms, sum_len_terms/count_terms)

    # 3. количество и средняя длина уникальных токенов во всем тексте и уникальных термов в индексе
    logger.info("in index: count_tokens = %d, average_length_uniq_tokens = %f",
        count_uniq_tokens, sum(len(t) for t in uniq_tokens)*1.0/count_uniq_tokens)
    logger.info("in index: count_terms_in_rindex = %d, average_length_of_rindex-terms = %f",
        count_uniq_terms, sum(len(t) for t, _ in rindex)*1.0/count_uniq_terms)

    return rindex


def main():
    url = 'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0080.shtml' # Анна
    # url = 'http://az.lib.ru/p/pushkin_a_s/text_0427.shtml' # Кирджали

    session = requests.session()
    html = html5lib.parse(
        session.get(url).text,
        treebuilder="lxml",
        namespaceHTMLElements=False,
    ).getroot()

    get_reverse_index(html, '//body/div[1]/xxx7/dd') # для Анны
    # get_reverse_index(html, '//body/div[1]/div/div/xxx7/dd') # для Кирджали

if __name__ == '__main__':
    main()
