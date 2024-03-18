|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Идентификация взаимосвязи между метками с использованием алгоритма, основанного на собственном внимании к задаче классификации с несколькими метками, обосновывающая связь с процессами Хоукса.
    :Тип научной работы: M1P/НИР/CoIS
    :Автор: Галина Леонидовна Боева
    :Научный руководитель: к.ф.-м.н. Зайцев Алексей 
    :Научный консультант(при наличии): к.ф.-м.н. Грабовой Андрей

Abstract
========

Большая часть доступной пользовательской информации может быть представлена в виде последовательности событий с временными метками. Каждому событию присваивается набор категориальных меток, будущая структура которых представляет большой интерес. Это задача прогнозирования временных наборов для последовательных данных. Современные подходы фокусируются на архитектуре преобразования последовательных данных, используя собственного внимания("self-attention") к элементам в последовательности. В этом случае мы учитываем временные взаимодействия событий, но теряем информацию о взаимозависимостях меток. Мотивированные этим недостатком, мы предлагаем использовать механизм собственного внимания("self-attention") к меткам, предшествующим прогнозируемому шагу. Поскольку наш подход представляет собой сеть внимания к меткам, мы называем ее LANET.  Мы также обосновываем этот метод агрегирования, он положительно влияет на интенсивность события, предполагая, что мы используем стандартный вид интенсивности, предполагая работу с базовым процессом Хоукса.

Setting the environment
========================
```
HYDRA_FULL_ERROR=1 python3 train.py --config-name=train.yaml trainer.gpus=0 
```
Внутри конфига datamodule.yaml можно менять параметр исторической информации, датасет или размер векторного пр-ва.

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.
