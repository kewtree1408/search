#! /usr/bin/python
# coding: utf-8

import requests
import html5lib
import subprocess
import codecs
import cPickle
import string
import sys
import os

from math import log
from nltk.tag import pos_tag
from nltk import wordpunct_tokenize
from collections import Counter
from math import log
from os.path import join
from html2text import html2text


# является ли токен пунктуацией?
def is_punctuation(token):
    for punct in string.punctuation:
        if punct in token:
            return True
    return False


# берем страницу html
def get_html(url):
    session = requests.session()
    html = html5lib.parse(
        session.get(url).text,
        treebuilder="lxml",
        namespaceHTMLElements=False,
    ).getroot()
    return html


def get_content(url):
    html = requests.get(url).content
    return html2text(html.decode('cp1251'))


# либо берем данные из файла, либо получаем сами и потом сохраняем в файл
def save_and_get_data(fname, callback):
    try:
        with open(fname, 'rb') as bf:
            data = cPickle.load(bf)
    except IOError as ex:
        data = callback()
        with open(fname, 'wb') as bf:
            cPickle.dump(data, bf)
    return data


# сохраняем результаты(типа collections.Counter) в файл
def save_result(cnt_obj, fname):
    with codecs.open(fname, encoding='utf-8', mode='w') as f:
        for elem in cnt_obj.most_common():
            f.write(u"%s %s\n" % (elem[0], elem[1]))


# Сохраняем просто текст в unicode
def save_text(text, fname):
    with codecs.open(fname, encoding='utf-8', mode='w') as f:
        f.write(text)


# первый раз вытаскиваем текст и сохраняем в файл
# в качестве побочного результата - сам текст в урле
def get_text(urls_xpaths, fname):
    def parse():
        text = ''
        for url in urls_xpaths:
            try:
                t = get_content(url)
            except Exception as ex:
                print ex
            text += t
        return text

    data = save_and_get_data(fname, parse)
    return data


# получаем словарь: токен - частота
def get_tokens(text, fname):
    def pick_tokens():
        # токинизируем и приводим все токены к нижнему регистру
        all_tokens = [token.lower()
            for token in wordpunct_tokenize(text) if not is_punctuation(token)]
        tokens_freq = Counter() # {'еда': 4, 'идти': 6}
        for t in all_tokens:
            tokens_freq[t] += 1
        return tokens_freq

    tokens_freq = save_and_get_data(fname, pick_tokens)
    return tokens_freq


# посмотрим на биграммы
def get_bigrams(text, fname):
    def pick_bigrams():
        tokens = [t.lower() for t in wordpunct_tokenize(text) if not is_punctuation(t)]
        count = len(tokens)
        bigrams = Counter() # {('Анна', 'Каренина'): 10}
        for i in xrange(0,count-1):
            t1, t2 = tokens[i], tokens[i+1]
            bigrams[(t1, t2)] += 1
        return bigrams

    bigrams = save_and_get_data(fname, pick_bigrams)
    return bigrams

def get_results(urls, name_dir):
    # сохраняем текст в бинарном виде или смотрим существующий
    text = get_text(urls, join(name_dir, 'text.pkl'))
    save_text(text, join(name_dir, 'text.txt'))

    # токенизация, подсчет общего кол-ва слов, частота встречаемого слова
    # сохраняем информацию в файле
    tokens = get_tokens(text, join(name_dir,'tokens.pkl'))
    save_result(tokens, join(name_dir,'tokens.txt'))

    # если еще не делали, то ищем биграммы и коллокации
    bigrams = get_bigrams(text, join(name_dir,'bigrams.pkl'))
    save_result(bigrams, join(name_dir,'bigrams.txt'))

    return text, tokens, bigrams

def get_probability_train_gramm(tr_tokens, tr_bigrams, test_tokens, test_bigrams):
    N = len(tr_tokens)
    B = len(tr_bigrams)
    lbd = 0.00000000001
    P_ngram = 1.0
    P_test_ngram = 1.0
    log_P_ngram = 0.0
    log_P_test_ngram = 0.0

    for b in tr_bigrams:
        w1, c_w1w2 = b[0], tr_bigrams[b]
        p_w1w2 = (c_w1w2+lbd)/(N+B*lbd)
        c_w1 = tr_tokens[w1]
        # main form: P(t2|t1) = P(t1, t2)/P(t1)
        p_w1 = (c_w1+lbd)/(N+B*lbd)
        p_w2w1 = p_w1w2/p_w1
        P_ngram *= p_w2w1
        log_P_ngram += -log(p_w2w1, 2)

    for b in test_bigrams:
        w1, c_w1w2 = b[0], test_bigrams[b]
        p_w1w2 = (c_w1w2+lbd)/(N+B*lbd)
        c_w1 = tr_tokens[w1]
        # main form: P(t2|t1) = P(t1, t2)/P(t1)
        p_w1 = (c_w1+lbd)/(N+B*lbd)
        p_w2w1 = p_w1w2/p_w1
        P_test_ngram *= p_w2w1
        log_P_test_ngram += -log(p_w2w1, 2)


        print w1, "\t", (P_ngram, log_P_ngram), (P_test_ngram, log_P_test_ngram)
    print "\n\n\n"
    return (P_ngram, log_P_ngram), (P_test_ngram, log_P_test_ngram)


def tolstoy():
    name_dir_train = 'train_results'
    name_dir_test = 'test_results'
    try:
        print u"Данные хранятся в папках: %s и %s" % (name_dir_train, name_dir_test)
        os.mkdir(name_dir_train)
        os.mkdir(name_dir_test)
    except OSError as ex:
        pass

    # обучающее множество
    urls_tolstoy = (
        # 4 тома "Война и Мир"
        'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0040.shtml',
        'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0050.shtml',
        'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0060.shtml',
        'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0070.shtml',
        # Анна Каренина
        'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0080.shtml',
        # Воскресение
        'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0090.shtml',
        # Пушкин - Кирджали
        # 'http://az.lib.ru/p/pushkin_a_s/text_0427.shtml',
    )

    # тестовое множество
    urls_pushkin = (
        # Пушкин - Кирджали
        'http://az.lib.ru/p/pushkin_a_s/text_0427.shtml',
    )

    train_text, train_token, train_bigrams = get_results(urls_tolstoy, name_dir_train)
    test_text, test_token, test_bigrams = get_results(urls_pushkin, name_dir_test)

    n_gramm, test_n_gramm = get_probability_train_gramm(train_token, train_bigrams, test_token, test_bigrams)
    print n_gramm, "\t", test_n_gramm

    # (0.0, 2888582.773172026)    (0.0, 6614.040213921463) lbd = 0.000001
    # (0.0, 2534344.1284884815)   (0.0, 10500.746690466585) lbd = 0.999999
    # (0.0, 2888587.872329521)    (0.0, 4368.441697310411) lbd = 0.0000001
    # (0.0, 2888587.918687691)    (0.0, 3245.63022738086) lbd = 0.00000001
    # (0.0, 2888587.9238303555)   (9.28836275339043e+36, -122.80483573314021) lbd = 0.00000000001

    return

    # real    0m37.991s
    # real    0m16.438s


if __name__ == '__main__':
    tolstoy()
