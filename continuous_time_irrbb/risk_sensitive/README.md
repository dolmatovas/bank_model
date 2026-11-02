# Risk-sensitive IRRBB

Материалы исследования динамического управления долей fixed/float в банковской книге.

- `STATIONARY.md` — самодостаточное описание стационарной постановки.
- `environment.py`, `bellman.py`, `stationary_solver.py` — среда и численное решение.
- `train_stationary.py`, `sweep_stationary.py` — обучение и перебор параметров.
- `policy_scenarios.ipynb`, `risk_metrics_interpretation.ipynb`, `gamma_risk_interpretation.ipynb` — сценарии и интерпретация риск-метрик.
- `policy_benchmark.ipynb` — сравнение с эвристиками.
- `RISK_METRIC_PROXIES.md`, `value_proxy_experiments.ipynb` — прокси для ΔV.
- `stationary_formulation/presentation.tex` — презентация; изображения и шрифты включены.

## Запуск

Нужны Python, PyTorch и NumPy; для ноутбуков также Jupyter и Matplotlib. Обучение поддерживает CPU и CUDA.
Из корня репозитория:

```bash
python -m continuous_time_irrbb.risk_sensitive.train_stationary --device cuda --output continuous_time_irrbb/risk_sensitive/runs/new_run
```

Или из этой папки:

```bash
python train_stationary.py --device cuda --output runs/new_run
```

Полный список параметров: `python train_stationary.py --help`.

Папка `runs/` с чекпоинтами и кешами оценок намеренно исключена из Git. Для повторного выполнения ноутбуков добавьте свои чекпоинты в `runs/` и укажите соответствующий `RUN_DIR` или `SWEEP_DIR`. Сохранённые в исходных ноутбуках результаты и графики оставлены без изменений. Они относятся к различным экспериментам; сходимость нужно проверять отдельно для каждого запуска.

Тесты из корня репозитория:

```bash
python -m unittest continuous_time_irrbb.risk_sensitive.test_model continuous_time_irrbb.risk_sensitive.test_solver continuous_time_irrbb.risk_sensitive.test_stationary
```
