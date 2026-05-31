# 📘 Назва проєекту
> Моделювання та аналіз інтелектуальних агентів для розвідки у невідомому середовищі

---

## 👤 Автор

- **ПІБ:** Шуляк Ілля Володимирович 
- **Група:** ФеІ-43
- **Керівник:** Катеринчук Іван, кандидат фізико-математичних наук, доцент кафедри оптоелектроніки та інформаційних технологій
- **Дата виконання:** [30.05.2026]

---

## 📌 Загальна інформація


Тип проєкту:  Симуляційне середовище 
Мова програмування: Python 3.11 
Бібліотеки: `pygame`, `numpy`, `matplotlib`, `pandas`, `seaborn` |

---

## 🧠 Опис функціоналу

- 🗺️ **Генерація лабіринтів** — алгоритм рекурсивного повернення, лабіринти з петлями, болото, 5 готових заготовок
- 🤖 **7 алгоритмів-агентів** — Random, DFS, BFS, Greedy Best-First, LRTA\*, FSA-A\*, D\* Lite
- 👁️ **Туман війни** — агент бачить лише безпосередніх сусідів, права панель показує повну карту
- 📊 **Пакетний режим** — збір метрик по N запусків, генерація графіків та CSV-файлів
- 🖥️ **Інтерактивний режим** — меню вибору лабіринту та агента, перегляд симуляції в реальному часі

---

## 🧱 Опис основних файлів


- `main.py` - інтерактивний режим
- `agents.py` - клас `Agent` — обгортка над об'єктом політики
- `policies.py` - Реалізація 7 алгоритмів: Random, DFS, BFS, Greedy, LRTA\*, FSA-A\*, D\* Lite
-`simulations.py` - містить класи  `SimpleSimulation` і `BatchSimulation`, які відповідають за запуск емуляцій
- `worlds.py` - Генератори лабіринтів та реєстр 5 пресетів (`MAZE_PRESETS`)
- `analysis.py` - візуальний інтерфейс для налаштування та запуску пакетних експериментів 
- `env/gridworld.py` - містить клас `GridEnv` — середовище (крок агента, FOV, термінальний стан)
- `experiments/` - папка з результатами запусків (CSV, рафіки) 

---

## ▶️ Запуск проекту

### 1. Вимоги

- Python 3.11
- pip

### 2. Клонування репозиторію

```bash
git clone https://github.com/wisdan31/bachelors_thesis.git
cd bachelors_thesis
```

### 3. Створення та активація віртуального середовища

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 5. Запуск симулятора (візуальний режим)

```bash
python main.py
```

### 6. Запуск аналізу в пакетному режимі (з UI)

```bash
python analysis.py
```

---

## 📊 Метрики, що збираються

### Якість рішення
- **Коефіцієнт субоптимальності** — відношення вартості шляху агента до оптимального 
- **Відсоток успішності** — частка запусків, де агент досяг цілі

### Обчислювальні витрати
- **Затримка прийняття рішення** - мс\крок
- **Пікова пам'ять** — кількість записів у структурах даних агента
- **Повторні відвідування** — кількість разів, коли агент повертається на вже відвідану клітину

### Поведінка дослідження
- **Ефективність відкриття карти** — відсоток відкритих клітин від загальної площі
- **Кроки бектрекінгу** — кроки на вже відвідані клітини
- **Середній інформаційний приріст** — нових клітин відкрито за крок
- **Теплова карта відвідувань** — просторовий розподіл переміщень агента

---

## 🖱️ Інструкція для користувача

### Інтерактивний режим (`main.py`)

1. **Ліва панель — SELECT MAZE**: виберіть один із 5 пресетів або `Random Maze`
2. **Права панель**: попередній перегляд обраного лабіринту
3. **SELECT AGENT**: оберіть один із 7 алгоритмів
4. Натисніть **START SIMULATION**

### Під час симуляції

- **Ліва панель** — вид агента
- **Права панель** — повна карта 

### Пакетний режим (`analysis.py`)

1. Виберіть агентів 
2. Виберіть розмір сітки (11–51)
3. Виберіть кількість запусків (5–100)
4. Натисніть **RUN ANALYSIS** — результати зберігаються в `experiments/<timestamp>/`

---

## 🧪 Відомі проблеми та рішення

- **`ModuleNotFoundError: No module named 'pygame'`** — Запустіть `pip install -r requirements.txt` у активованому venv

---

## 🧾 Використані джерела / література

1. Choset, H., Lynch, K. M., Hutchinson, S., Kantor, G., Burgard, W., Kavraki, L.
E., & Thrun, S. (2005). Principles of Robot Motion: Theory, Algorithms, and
Implementations. MIT Press.
2. Choset, H., Lynch, K. M., Hutchinson, S., Kantor, G., Burgard, W., Kavraki, L.
E., & Thrun, S. (2005). Principles of Robot Motion: Theory, Algorithms, and
Implementations. MIT Press.
3. Russell, S. J., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach
(4th ed.). Pearson.
4. Elfes, A. (1989). Using occupancy grids for mobile robot perception and
navigation. Computer, 22(6), 46-57.
5. Buro M. Real-time strategy games: A new AI research challenge. Proceedings
of the 18th International Joint Conference on Artificial Intelligence (IJCAI).
Acapulco, Mexico, 2003, 1534–1535.
6. Korf, R. E. (1990). Real-time heuristic search. Artificial Intelligence
7. Koenig, S., & Likhachev, M. (2002). D* Lite. AAAI/IAAI, 15, 476-483.
8. Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein
"Introduction to Algorithms"
9. Pearl, J. (1984). Heuristics: Intelligent Search Strategies for Computer Problem
Solving. Addison-Wesley.
10. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A Formal Basis for the
Heuristic Determination of Minimum Cost Paths. IEEE Transactions on
Systems Science and Cybernetics, 4(2), 100-107.
11. Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
Numerische Mathematik, 1(1), 269-271.
12. Koenig, S., & Likhachev, M. (2005). Fast Replanning for Navigation in
Unknown Terrain. IEEE Transactions on Robotics, 21(3), 354-363.
13.Yamauchi, B. (1997). A frontier-based approach for autonomous exploration.
84
14. LaValle, S. M. (2006). Planning Algorithms. Cambridge University Press
