Search
======

Наработки по курсу "Информационный поиск"

-----------------------------------------
1) Задание:
Построить индекс по тексту Л.Н. Толстого "Анна Каренина". Вывести статистику:
 - количество токенов и терминов
 - средняя длина токена и термина


Запуск: ./1.py
Результаты и логи в stats.txt.

------------------------------------------
2) Задание: Реализовать поиск по обратному индексу.
Ввод: булев запрос и (опционально) количество выводимых результатов.
Вывод: список найденных результатов:
    - на отдельной строке номер параграфа и снипет с найденными словами из этого параграфа
    - результаты сортируются по номеру параграфа
    - (*) результаты сортируются по частоте найденных слов запроса в параграфе (самые частотные результаты сверху)


Перед запуском нужно сделать pip install -r requirements.txt
Запуск можно сделать так:
    ./2.py -q 'я & яичница | этот & ядовитый & язык ' -p 1000 -s 1000
или так:
    ./2.py -q 'Анна & любовь | смерть & боль | еда & печенье' -p 100 -s 1000
или любой другой запрос, содержащий слова с & или | (без скобок и отрицания)

Поддерживаемые опции:
 - запрос (-q)
 - количество выводимых параграфов (-p)
 - количество снипеттов в найденном параграфе (в качестве снипеттов - предложение с найденными словами) (-s)
Более подробно -- в хелпе: (./2.py -h)
Результаты выводятся на стандартный вывод в формате:
    Query in [номера всех параграфов, соответствующих запросу]
    номер параграфа:
        предложение с запросом (слова из запроса выделены желтым цветом)

Баги:
1. Код поиска по обратному индексу соединен с построением обратного индекса. Соответственно, при каждом запросе каждый раз строится обратный индекс.
2. В зависимости от запроса, поиск занимает от 0.5 до 1.5 минут.
