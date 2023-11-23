# HSE Sber ML Hack
## мисисково

Команда:
* Гуреева Ирэна
* Минина Полина
* Окунев Даниил
* Скрипкин Матвей
* Югай Александр

Структура репозитория:
* [`data_analysis.ipynb`](data_analysis.ipynb) - разведочный анализ данных.
* [`aggregation_model.ipynb`](aggregation_model.ipynb) - часть топ-1 решения, финальное решение состоит из блендинга нескольких подобных моделей.
* [`ptls_notebook.ipynb`](ptls_notebook.ipynb) - эксперименты с нейросетевыми моделями  с использованием библиотеки `pytorch-lifestream`.
* [`checkpoints`](checkpoints) - папка с сохраненными весами нейросетевых моделей.
* [`models.py`](models.py) - нейросетевые модели.
* [`dataset.py`](dataset.py) - специальный датасет для подгрузки разных подпоследовательностей каждого пользователя в стиле [CoLES](https://arxiv.org/abs/2002.08232).

Окружение для воспроизводимости

```
conda env create -f environment.yml
conda activate sber
```
