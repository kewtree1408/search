#! /usr/bin/python
# code: utf-8
import requests
from pprint import pprint

def inverse_index(text):
    pairs_seq = []
    for doc in parag:
        doc_id = get_doc_id() # хеш или уникальный id
        tokens = get_tokens() # токены
        for token in tokens:
            count_tokens += 1 #количество всех токенов или только уникальных токенов
            len_tokens += len(token)
            term = get_term(token) # нормализация
            pairs_seq.append((term, doc_id))

    sorted(pair_seq) # sort: сначала по термам, потом по doc_id
    # reduce, подсчет частоты + doc_id в списки, map-reduce???
    index = {}
    # for _ in pair_seq:
        # ...
    # ... подсчет кол-ва термов и длина каждого терма


    return index


# 2. Частота считается только для разных документах

def main():
    # url = 'http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0080.shtml' #Анна
    url = 'http://az.lib.ru/p/pushkin_a_s/text_0427.shtml' #Кирджали
    r = requests.get(url)
    ii = create_inverse_index(r.text)
    # write-to-file or return:
    # 1) index
    # 2) count of tokens & terms
    # 3) average length of tokens & terms


if __name__ == '__main__':
    main()
