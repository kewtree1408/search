#! /usr/bin/python
# coding: utf-8

import requests
import html5lib
import string
import logging
import cPickle
import codecs
import collections
import re

from nltk import wordpunct_tokenize
from nltk.stem.snowball import RussianStemmer
from pprint import pprint
from optparse import OptionParser
from termcolor import colored


rus_stemmer = RussianStemmer()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('stats_rindex.txt')
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
            tokens.add(t)

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

    # все что ниже - для сбора статистики
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

    return dict(index_dict)


def get_html(url):
    session = requests.session()
    html = html5lib.parse(
        session.get(url).text,
        treebuilder="lxml",
        namespaceHTMLElements=False,
    ).getroot()
    return html


def create_rindex(url, xpath):
    html = get_html(url)
    return get_reverse_index(html, xpath)


# соответствие между параграфами и содержащимся в нем текстом
def get_docs_contexts(url, path_value):
    html = get_html(url)
    doc_context = collections.defaultdict(lambda: "")
    doc_id = 0
    for dd in html.xpath(path_value):
        if dd.text: doc_context[doc_id] += dd.text
        for p in dd.xpath('.//p/*'):
            doc_id += 1
    return doc_context


# фильтрация: замена списков токенов на термы
def filter_query(query):
    terms_query = set()
    for q in query:
        tok = get_tokens(q)[0]
        terms_query |= get_terms(tok)[0]
    return terms_query


# поиск параграфов с запросами
def finder(rindex, query):
    terms_and_query = set()

    # сначала разбиваем по ИЛИ
    or_queries = query.split('|')
    if len(or_queries) > 1:
        result = set()
        for or_query in or_queries:
            result |= finder(rindex, or_query)
        # print "or_query", result
        return result

    # в поддзапросах из ИЛИ вынимаем запросы с И
    and_query = query.split('&')
    terms_and_query = filter_query(and_query)
    # получение списка документов, сортировка по частоте
    # for t in terms_and_query:
        # print t
        # print rindex.get(t)
    freq_docids = sorted([rindex.get(t, [0, tuple()]) for t in terms_and_query], key=lambda freq: freq[0])
    # пересечение координатных блоков через intersection
    res = reduce(set.intersection, [set(s[1]) for s in freq_docids])
    return res


# есть ли термы в предложении (сниппете)
def is_terms_in_seq(terms, seq):
    # разбиваем предложение на термы
    seq_terms = get_terms(get_tokens(seq)[0])[0]
    for term in terms:
        if term in seq_terms:
            return True
    return False


# сплиттим запрос по | и & и фильтруем токены, заменяя на термы
def get_query_terms(query):
    query_word = [q for and_q in query.split('|') for q in and_q.split('&')]
    query_terms = filter_query(query_word)
    return query_terms


# получаем сниппеты (предложения разделенные '.') из текста параграфа
def get_snippet(query, text, nsnippet):
    query_terms = get_query_terms(query)
    sequences = text.split('.')
    snippet = []
    n = 0
    # print text, len(sequences)
    for seq in sequences:
        seq = seq.strip('\n\t\r')
        if is_terms_in_seq(query_terms, seq) and n < nsnippet:
            snippet.append(seq+'.')
            n += 1
        elif n > nsnippet:
            break

    return snippet


# подсветка слов из запроса в результате
def light_text(terms, snippet):
    if isinstance(snippet, unicode):
        snippet = snippet.encode('utf-8')
    for t in terms:
        for word in re.findall(
                (u'[^A-Za-zЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЁЯЧСМИТЬБЮйцукенгшщзхъфывапролджэёячсмитьбю]('.encode('utf-8')
                    + t.encode('utf-8')
                    + u'[a-zйцукенгшщзхъфывапролджэёячсмитьбю]{0,8})'.encode('utf-8')
                    + u'[^A-Za-zЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЁЯЧСМИТЬБЮйцукенгшщзхъфывапролджэёячсмитьбю]'.encode('utf-8')),
                snippet, flags=re.IGNORECASE):
            snippet = snippet.replace(word, colored(word, 'yellow'))
        for word in re.findall(
                (u'[^A-Za-zЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЁЯЧСМИТЬБЮйцукенгшщзхъфывапролджэёячсмитьбю]('.encode('utf-8')
                    + t.capitalize().encode('utf-8')
                    + u'[a-zйцукенгшщзхъфывапролджэёячсмитьбю]{0,8})'.encode('utf-8')
                    + u'[^A-Za-zЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЁЯЧСМИТЬБЮйцукенгшщзхъфывапролджэёячсмитьбю]'.encode('utf-8')),
                snippet, flags=re.IGNORECASE):
            snippet = snippet.replace(word, colored(word, 'yellow'))
    return snippet


def get_rindex(url, xpath, fname):
    try:
        with open(fname, 'rb') as bf:
            rindex = cPickle.load(bf)
    except IOError as ex:
        rindex = create_rindex(url, xpath)
        with open(fname, 'wb') as bf:
            cPickle.dump(rindex, bf)
    return rindex


def parse_synonims(fname):
    synon = collections.defaultdict(list)
    with codecs.open(fname, encoding='cp1251', mode='r') as f:
        for line in f:
            parts = line.split('|')
            main_word = parts[0]
            if len(main_word.split()) == 1:
                synon[main_word] += [word for word in parts[1].split(',') if len(word.split())==1]
    
    return dict(synon)


def get_synonims(fname, name_obj_syn):
    try:
        with open(name_obj_syn, 'rb') as bf:
            synonims = cPickle.load(bf)
    except IOError as ex:
        synonims = parse_synonims(fname)
        with open(name_obj_syn, 'wb') as bf:
            cPickle.dump(synonims, bf)
    return synonims


def expand_query(query, synonims, rindex):
    ex_query = query[:]
    for q in query.split('|'):
        for qq in q.split('&'):
            word = qq.strip()
            ex_query = ex_query.replace(word, u' | '.join([word] + synonims.get(word, list())), 1)
    return ex_query

def main():
    # добавили аргументы командной строки
    usage = "Usage: ./%prog -q 'this & query | should | be | in & russian' -p 100 -s 100"
    parser = OptionParser(usage=usage)
    parser.add_option("-q", "--query", dest="query", type="string",
                help="query for binary search (support just | and &).")
    parser.add_option("-p", "--nparagraph", dest="nparagraph", type="int", default=100,
                help="count of output paragraphs. 100 by default.")
    parser.add_option("-s", "--nsnippet", dest="nsnippet", type="int", default=100,
                help="count of output snippets for one paragraph. 100 by default.")

    (options, args) = parser.parse_args()
    query = options.query
    nparagraph = options.nparagraph
    nsnippet = options.nsnippet

    url1 = 'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0080.shtml' # Анна
    xpath1 = '//body/div[1]/xxx7/dd' # для Анны

    url2 = 'http://az.lib.ru/p/pushkin_a_s/text_0427.shtml' # Кирджали
    xpath2 = '//body/div[1]/div/div/xxx7/dd' # для Кирджали

    if not query:
        print "Query is None."
        parser.print_help()
        return -1

    # получить синонимы 
    synonims = get_synonims('synonims.txt', 'synonims.pkl')
    
    # построили обратный индекс
    rindex = get_rindex(url1, xpath1, 'ridx_anna.pkl')
    
    # построили словарь типа "{номер параграфа: текст}"
    doc_context = get_docs_contexts(url1, xpath1)
    
    # расширение запроса
    ex_query = expand_query(query.decode('utf8'), synonims, rindex)
    print ex_query
    
    # нашли запросы и отсортировали параграфы
    docids = sorted(list(finder(rindex, ex_query)))
    print "Query in ", docids

    # выводим результаты: номер параграфа и сниппет с искомыми словами
    for docid in docids[:nparagraph]:
        snippet = get_snippet(ex_query, doc_context[docid], nsnippet)
        print "%d: " % docid
        for s in snippet:
            print "\t", light_text(get_query_terms(ex_query), s)


if __name__ == '__main__':
    main()
