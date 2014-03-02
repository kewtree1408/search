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

from nltk.tag import pos_tag
from nltk import wordpunct_tokenize
from collections import Counter
from pymorphy import get_morph
from math import log
from os.path import join
from html2text import html2text


# для ошибки первого рода, {alpha: threshold}
chi_critical_values = {
    0.1: 2.71,
    0.05: 3.84,
    0.01: 6.63,
    0.005: 7.88,
    0.001: 10.83,
}


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
            f.write(u"%s %s %s\n" % (elem[0][0], elem[0][1], elem[1]))


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


# в morph_dicts хранятся все словари
def download_morph():
    # скачиваем и используем словарь для получения грамматической информации о слове (часть речи)
    path_to_dictionary = os.path.realpath(os.path.curdir)
    morph_path = join(path_to_dictionary, 'morph_dicts')
    if not os.path.exists(morph_path):
        subprocess.call(['wget', 'https://bitbucket.org/kmike/pymorphy/downloads/ru.sqlite-json.zip'])
        subprocess.call(['unzip', 'ru.sqlite-json.zip', '-d', 'morph_dicts'])
    morph = get_morph(morph_path)
    return morph


# POS - parst of speech - опредление части речи для русского языка
# с помощью библиотеки: http://pythonhosted.org//pymorphy/intro.html#id1
def custom_pos_tag(token, morph):
    info = morph.get_graminfo(token.upper())
    if info:
        return info[0]['class']
    return u'Неизвестно'


# получаем информацию о тексте: про каждый токен - часть речи и частота
def get_info_tokens(text, fname):
    def pick_info():
        morph = download_morph()
        tokens = dict()
        # токинизируем и определяем часть речи
        tagging_tokens = [(token.lower(), custom_pos_tag(token, morph))
            for token in wordpunct_tokenize(text) if not is_punctuation(token)]
        info_tokens = Counter() #{('еда', 'С'): 4, ('идти', 'Г'): 6}
        for t in tagging_tokens:
            info_tokens[t] += 1
        return info_tokens

    info_tokens = save_and_get_data(fname, pick_info)
    return info_tokens


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


# ainfo == apriori informations about tokens
# Правила для выбора коллокаций из биграмм:
# Прилаготельное Существительное
# Существительное Существителное
def get_collocs(text, ainfo, fname):
    def pick_collocs():
        tokens = [t.lower() for t in wordpunct_tokenize(text) if not is_punctuation(t)]
        count = len(tokens)
        collocations = Counter()
        for i in xrange(0,count-1):
            t1, t2 = tokens[i], tokens[i+1]
            if ((t1, u'П') in ainfo and (t2, u'С') in ainfo):
                collocations[(t1,t2)] += 1
            elif ((t1, u'С') in ainfo and (t2, u'С') in ainfo):
                collocations[(t1,t2)] += 1
        return collocations

    collocs = save_and_get_data(fname, pick_collocs)
    return collocs


# отношение правдоподобия
# ainfo == apriori informations about tokens
# c == one collocation
def likelihood_ratio(colloc, ainfo, N, alpha=0.05):
    w1, w2 = colloc[0][0], colloc[0][1]
    token_freq = dict([(info[0][0], info[1]) for info in ainfo.most_common()])
    # c1, c2, c12 for the number of occurrences of w1, w2, w1w2 in the corpus
    c1, c2 = token_freq[w1], token_freq[w2]
    c12 = colloc[1]

    p = 1.0*c2/N
    p1 = 1.0*c12/c1
    p2 = 1.0*(c2-c12)/(N-c1)

    def L(k,n,x):
        l = x**k * (1.0-x)**(n-k)
        if l == 0:
            return sys.float_info.epsilon
        return l

    log_L = log(L(c12,c1,p)) + log(L(c2-c12,N-c1,p)) - log(L(c12,c1,p1)) - log(L(c2-c12,N-c1,p2))
    if -2*log_L < chi_critical_values[alpha]:
        # не коллокация - между словами есть зависимость
        return False
    else:
        # колокация - между словами нет зависимости
        return True


# критерий Пирсона
def pirson_test(c, bigrams, N, alpha=0.05):
    o11 = c[1] # new companies
    o12 = 0 # old companies
    o21 = 0 # new machines
    o22 = 0 # old machines
    w1, w2 = c[0][0], c[0][1]
    for b in bigrams.most_common():
        if w2 == b[0][1] and w1 != b[0][0]:
            o12 += b[1]
        elif w1 == b[0][0] and w2 != b[0][1]:
            o21 += b[1]
    o22 = N-o11
    chi_square = (1.0*N *(o11*o22 - o12*o21)**2)/((o11+o12)*(o11+o21)*(o12+o22)*(o21+o22))
    # Для уровня значимости alpha=0,05 порог 3,841, основная гипотеза не может быть отвергнута.
    if chi_square < chi_critical_values[alpha]:
        # не коллокация - между словами есть зависимость
        return False
    else:
        # колокация - между словами нет зависимости
        return True


def check_results(collocs, bigrams, info_tokens, fname, check_count=100):
    # начальная гипотеза: H0 – между словами нет зависимости.
    N = sum(collocs.values())
    with codecs.open(fname, encoding='utf-8', mode='w') as f:
        f.write(u"Колокация | Частота вхождения | Критерий Пирсона | Критерий максимального правдоподобия \n")
        for c in collocs.most_common()[:int(check_count)]:
            res_p = pirson_test(c, bigrams, N)
            res_l = likelihood_ratio(c, info_tokens, N)
            f.write(u"(%s %s %d): p=%s l=%s\n" % (c[0][0], c[0][1], c[1], res_p, res_l))


def tolstoy():
    name_dir = 'tolstoy_results'
    try:
        print u"Все результаты хранятся в папке", name_dir
        os.mkdir(name_dir)
    except OSError as ex:
        pass

    url_xpath_tolstoy = (
        # 4 тома "Война и Мир"
        'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0040.shtml',
        'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0050.shtml',
        'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0060.shtml',
        'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0070.shtml',
        # Анна Каренина
        'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0080.shtml',
        # Воскресение
        'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0090.shtml',
    )

    # сохраняем текст в бинарном виде или смотрим существующий
    text = get_text(url_xpath_tolstoy, join(name_dir, 'text.pkl'))
    save_text(text, join(name_dir, 'text.txt'))

    # токенизация, подсчет общего кол-ва слов, частота встречаемого слова
    # сохраняем информацию в файле
    info_tokens = get_info_tokens(text, join(name_dir,'tokens.pkl'))
    save_result(info_tokens, join(name_dir,'tokens.txt'))

    # если еще не делали, то ищем биграммы и коллокации
    bigrams = get_bigrams(text, join(name_dir,'bigrams.pkl'))
    save_result(bigrams, join(name_dir,'bigrams.txt'))

    collocations = get_collocs(text, info_tokens, join(name_dir,'collocations.pkl'))
    save_result(collocations, join(name_dir,'collocations.txt'))

    # проверка результатов по критерию Пирсона и отношению правдоподобия
    check_count = 100
    check_results(collocations, bigrams, info_tokens, join(name_dir,'check_res.txt'), check_count)
    FP_p = 38 # штук не являются коллокацими (посчитано вручную), по критерию Пирсона
    TP_p = 62 # штук являются коллокацими (посчитано вручную), по критерию Пирсона
    FP_l = 34 # штук не являются коллокацими (посчитано вручную), по методу максимального правдоподобия
    TP_l = 62 # штук являются коллокацими (посчитано вручную), по методу максимального правдоподобия
    P_p = 1.0*TP_p/(TP_p + FP_p)
    P_l = 1.0*TP_l/(TP_l + FP_l)
    print "Точность по Критерию Пирсона: ", P_p
    print "Точность по методу максимального правдоподобия: ", P_l

    return


if __name__ == '__main__':
    tolstoy()
