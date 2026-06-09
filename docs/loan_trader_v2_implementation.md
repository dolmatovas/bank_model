# Toy-модель банка с процентным риском и капиталом: план реализации на Python

## 1. Цель

Реализовать минимальную stochastic control модель банка:

- кредиты полностью фондируются депозитами: `A_t = D_t`;
- капитал лежит как cash buffer: `C_t = K_t`;
- депозиты стоят short-rate: `R_D = r_t`;
- кредиты fixed-rate, новая выдача идёт под `r_t + spread(g_t)`;
- процентный риск проедает капитал через `NII`;
- если `CAR < c_star`, дивиденды запрещены;
- если `CAR < c0`, происходит default/resolution;
- акционер максимизирует дисконтированные дивиденды + terminal value или recovery.

---

## 2. Состояние

Минимальное состояние:

```python
state = (A, K, c, r)
```

где:

| Переменная | Смысл |
|---|---|
| `A` | объём fixed-rate кредитов |
| `K` | капитал / cash buffer |
| `c` | средний фиксированный купон портфеля |
| `r` | short rate |

Депозиты и cash выводятся из нормализации:

```python
D = A
C = K
```

---

## 3. Управления

На каждом шаге выбираются:

```python
action = (g, p)
```

где:

| Переменная | Смысл |
|---|---|
| `g` | новая выдача как доля текущего портфеля |
| `p` | payout ratio от положительной прибыли |

Для первой реализации удобно взять дискретную сетку:

```python
g_grid = np.array([0.00, 0.005, 0.010, 0.015, 0.020])
p_grid = np.array([0.0, 0.3, 0.6, 1.0])
```

Если `alpha = 0.01`, то `g = alpha` означает constant balance.

---

## 4. Параметры

Пример структуры параметров:

```python
from dataclasses import dataclass

@dataclass
class Params:
    T: int = 60
    gamma: float = 0.995

    alpha: float = 0.01
    s0: float = 0.03 / 12
    kappa: float = 1.0

    mu: float = 0.04 / 12
    rho: float = 0.95
    sigma: float = 0.01 / np.sqrt(12)
    r_floor: float = 0.0

    c_star: float = 0.12
    c0: float = 0.06
    delta: float = 0.70
    theta: float = 0.02
```

Важное замечание по единицам: лучше сразу работать в месячных ставках. Если годовая ставка 4%, то месячная приближённо `0.04 / 12`.

---

## 5. Функция спреда

Линейный спред:

```python
def spread(g: float, p: Params) -> float:
    return p.s0 - p.kappa * (g - p.alpha)
```

Опционально можно добавить floor:

```python
def spread(g: float, p: Params) -> float:
    return max(p.s_min, p.s0 - p.kappa * (g - p.alpha))
```

В первой версии floor лучше не ставить: пусть оптимизатор сам видит, что агрессивный рост может быть невыгоден.

---

## 6. One-step transition

Основная функция должна принимать состояние, действие и шок ставки.

```python
def step(state, action, eps, params):
    A, K, c, r = state
    g, payout = action

    # 1. Новый баланс
    A_next = A * (1.0 - params.alpha + g)

    # 2. Текущий NII / PnL
    profit = A * (c - r) + r * K

    # 3. Преддивидендный капитал
    K_pre = K + profit

    # 4. Проверка default/resolution
    car_pre_next_balance = K_pre / A_next
    if car_pre_next_balance < params.c0:
        recovery = (1.0 - params.delta) * max(K_pre, 0.0)
        next_state = None
        cashflow = recovery
        done = True
        info = {
            "profit": profit,
            "dividend": 0.0,
            "recovery": recovery,
            "default": True,
            "car": car_pre_next_balance,
        }
        return next_state, cashflow, done, info

    # 5. Dividend capacity
    desired_dividend = payout * max(profit, 0.0)
    max_dividend = max(K_pre - params.c_star * A_next, 0.0)
    dividend = min(desired_dividend, max_dividend)

    # 6. Капитал после дивидендов
    K_next = K_pre - dividend

    # 7. Новый средний купон
    denom = 1.0 - params.alpha + g
    new_loan_rate = r + spread(g, params)
    c_next = ((1.0 - params.alpha) * c + g * new_loan_rate) / denom

    # 8. Новый short rate
    r_next = params.mu + params.rho * (r - params.mu) + params.sigma * eps
    r_next = max(params.r_floor, r_next)

    next_state = (A_next, K_next, c_next, r_next)
    cashflow = dividend
    done = False
    info = {
        "profit": profit,
        "dividend": dividend,
        "recovery": 0.0,
        "default": False,
        "car": K_next / A_next,
        "spread": spread(g, params),
        "new_loan_rate": new_loan_rate,
    }
    return next_state, cashflow, done, info
```

---

## 7. Симуляция заданной политики

Сначала реализовать несколько простых политик без оптимизации.

### 7.1 Constant policy

```python
def constant_policy(state):
    return (0.01, 0.3)
```

### 7.2 Capital-based policy

```python
def capital_based_policy(state, params):
    A, K, c, r = state
    car = K / A

    if car < params.c_star:
        g = 0.0
        payout = 0.0
    elif car < params.c_star + 0.03:
        g = params.alpha
        payout = 0.3
    else:
        g = 0.02
        payout = 0.6

    return (g, payout)
```

### 7.3 Rate-and-capital policy

Идея:

- если ставки высокие, новые fixed-rate кредиты привлекательнее;
- если капитал низкий, рост и дивиденды надо ограничивать.

```python
def rate_capital_policy(state, params):
    A, K, c, r = state
    car = K / A

    if car < params.c_star:
        return (0.0, 0.0)

    if r > params.mu:
        g = 0.02
    else:
        g = params.alpha

    payout = 0.3 if car < params.c_star + 0.05 else 0.6
    return (g, payout)
```

---

## 8. Monte Carlo оценка политики

```python
def simulate_policy(policy, init_state, params, n_paths=10000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.zeros(n_paths)
    paths_info = []

    for i in range(n_paths):
        state = init_state
        total_value = 0.0
        defaulted = False
        info_path = []

        for t in range(params.T):
            action = policy(state, params)
            eps = rng.normal()
            next_state, cashflow, done, info = step(state, action, eps, params)

            total_value += (params.gamma ** t) * cashflow
            info_path.append(info)

            if done:
                defaulted = True
                break

            state = next_state

        if not defaulted:
            A, K, c, r = state
            terminal_value = K + params.theta * A
            total_value += (params.gamma ** params.T) * terminal_value

        values[i] = total_value
        paths_info.append(info_path)

    return values, paths_info
```

Метрики:

```python
values, paths = simulate_policy(policy, init_state, params)

mean_value = values.mean()
p05 = np.quantile(values, 0.05)
p01 = np.quantile(values, 0.01)
```

---

## 9. Brute-force policy search

Самый простой optimisation layer — подобрать параметры rule-based политики.

Например политика:

```python
def param_policy_factory(g_low, g_mid, g_high, p_low, p_high, car_buffer, r_threshold):
    def policy(state, params):
        A, K, c, r = state
        car = K / A

        if car < params.c_star:
            return (g_low, 0.0)

        if r > r_threshold and car > params.c_star + car_buffer:
            return (g_high, p_high)

        return (g_mid, p_low)

    return policy
```

Дальше перебор:

```python
best_score = -np.inf
best_params = None

for g_low in [0.0, 0.005, 0.01]:
    for g_mid in [0.005, 0.01, 0.015]:
        for g_high in [0.015, 0.02, 0.025]:
            for p_low in [0.0, 0.3]:
                for p_high in [0.3, 0.6, 1.0]:
                    policy = param_policy_factory(
                        g_low, g_mid, g_high,
                        p_low, p_high,
                        car_buffer=0.03,
                        r_threshold=params.mu,
                    )
                    values, _ = simulate_policy(policy, init_state, params, n_paths=2000)
                    score = values.mean()

                    if score > best_score:
                        best_score = score
                        best_params = (g_low, g_mid, g_high, p_low, p_high)
```

Можно заменить score на risk-adjusted objective:

```python
score = values.mean() - lam * (values.mean() - np.quantile(values, 0.01))
```

или использовать CVaR нижнего хвоста.

---

## 10. Dynamic programming на сетке

Для более строгой постановки можно решить finite-horizon dynamic programming.

### 10.1 Дискретизировать состояние

Пример сеток:

```python
A_grid = np.linspace(50, 200, 76)
K_grid = np.linspace(0, 40, 81)
c_grid = np.linspace(0.00, 0.12 / 12, 61)
r_grid = np.linspace(0.00, 0.12 / 12, 61)
```

Но полная 4D-сетка быстро становится большой.

### 10.2 Упростить состояние

Можно нормировать на `A`:

```text
x_t = K_t / A_t      # CAR
m_t = c_t - r_t      # current margin gap
r_t                  # rate level
```

Тогда состояние:

```python
state = (car, margin, r)
```

Это сильно уменьшит размерность.

### 10.3 Bellman recursion


default/recovery учитываются в transition.

```python
V[T, state] = K + theta * A

V[t, state] = max_action E[ cashflow + gamma * V[t+1, next_state] ]
```

Для ожидания по `eps` можно использовать Gauss-Hermite quadrature или дискретные shock nodes:

```python
eps_nodes = np.array([-2, -1, 0, 1, 2])
eps_probs = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
```

---

## 11. Reinforcement learning / approximate DP

Если не хочется строить сетку:

1. Сделать `gymnasium.Env` с состоянием `(A,K,c,r)`.
2. Actions — дискретные пары `(g,p)`.
3. Reward — дивиденд, recovery при default, terminal value в конце.
4. Обучить DQN/PPO.

Но для этой toy-модели лучше сначала сделать:

- Monte Carlo evaluation;
- grid search по rule-based политикам;
- потом finite-horizon DP.

---

## 12. Какие графики строить

Для интерпретации модели полезны:

1. Средняя траектория `A_t`.
2. Средняя траектория `K_t/A_t`.
3. Вероятность default по времени.
4. Распределение shareholder value.
5. Распределение terminal `CAR`.
6. Средний `g_t` как функция `CAR` и `r_t`.
7. Dividend stop frequency: доля месяцев, где `CAR < c_star` и дивиденды запрещены.
8. Сравнение политик:
   - aggressive growth;
   - conservative growth;
   - capital-based;
   - rate-and-capital;
   - optimised policy.

---

## 13. Sanity checks

Перед оптимизацией проверить:

### 13.1 Рост ставок ухудшает старый fixed book

Если `r_t` резко вырос, то:

```python
profit = A * (c - r) + r * K
```

должен падать, если `A >> K`.

### 13.2 Высокие ставки улучшают купон новой выдачи

Если `r_t` высокий, то:

```python
new_loan_rate = r + spread(g)
```

выше, и будущий `c_next` постепенно растёт.

### 13.3 Агрессивный рост снижает spread

Если `g` растёт, то:

```python
spread(g) = s0 - kappa * (g - alpha)
```

падает.

### 13.4 Dividend lock-up работает

Если:

```python
K_pre < c_star * A_next
```

то `dividend == 0`.

### 13.5 Default работает

Если:

```python
K_pre / A_next < c0
```

то transition должен вернуть `done=True` и `cashflow=recovery`.

---

## 14. Минимальная структура проекта

```text
toy_bank_ir/
  params.py          # dataclass Params
  model.py           # spread(), step(), rate process
  policies.py        # baseline policies
  simulate.py        # Monte Carlo simulator
  optimize.py        # grid search / policy search
  plots.py           # diagnostics
  notebook.ipynb     # experiments
```

---

## 15. Первый MVP

1. Реализовать `Params`.
2. Реализовать `spread()`.
3. Реализовать `step()`.
4. Реализовать `simulate_policy()`.
5. Сравнить 4 политики:
   - zero growth, no dividends;
   - constant balance, fixed payout;
   - aggressive growth;
   - capital-based policy.
6. Построить распределения shareholder value.
7. Добавить grid search по rule-based политикам.

---

## 16. Дальнейшие расширения

После MVP можно добавить:

1. 2D-gap вместо среднего купона `c_t`.
2. Множественные maturity buckets.
3. Floating/fixed mix активов.
4. Deposit beta вместо `R_D=r_t`.
5. Докапитализацию вместо default.
6. Recovery actions: stop growth, forced deleveraging.
7. CVaR objective.
8. Risk appetite constraints:
   - probability of default `< eps`;
   - probability of dividend stop `< eps`;
   - expected recovery loss `< threshold`.
