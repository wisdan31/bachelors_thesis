# Modeling and Analysis of Intelligent Agents for Exploration in Unknown Environments

A simulation-based benchmarking framework designed for analyzing the performance of different autonomous pathfinding algorithms navigating through unknown, partially observable environments. 

This project serves as the codebase for the Bachelor's Thesis: *"Modeling and Analysis of Intelligent Agents for Exploration in Unknown Environments"*.

---

## 📌 Project Overview
The primary objective of this project is to model and analyze how different intelligent search strategies perform when dropped into an entirely unknown environment (fog-of-war). Instead of having a full global map, the agents are provided with local, step-by-step visibility of their immediately adjacent surroundings. 

The core challenge revolves around finding an optimal path from the **Start (0,0)** to the **Goal (N-1, N-1)** while balancing state memory, exploration steps, and real-time computation.

## 🧠 Algorithms Tested
The simulation tests four core paradigms of agent reasoning:
1. **Random Agent (`RandomPolicy`)**: A baseline agent that randomly picks an unvisited adjacent node. Used to measure pure chance.
2. **DFS Agent (`DFSPolicy`)**: Explores deeply into one branch of the map. Backtracks when hitting a dead end. Highly effective in narrow, winding mazes.
3. **BFS Agent (`BFSPolicy`)**: Guarantees optimal path length from an omniscient perspective, but requires an incredible amount of physical step tracking and backtracking when navigating unknown grids physically.
4. **A* Agent (`AStarPolicy`)**: Incorporates space heuristics (Manhattan Distance). Evaluates the priority of unknown frontiers based on proximity to the objective.

## 🛠️ Environment & Maze Generation
- **Dynamic Maze Generation (`worlds.py`)**: Employs a Randomized Depth-First Search (Recursive Backtracking) to create perfect, highly branching 2D grids (solvability guaranteed).
- **Fog of War Observation (`env/gridworld.py`)**: Agents do not see the whole map. Every step, the environment reveals only the coordinates of valid, unblocked neighboring cells.

---

## 📊 Evaluation Metrics
The framework compiles statistics across multiple evaluation parameters to compare algorithm viability:
- **Steps**: The overall distance walked by the agent (including backtracking).
- **Unique Nodes Explored**: The efficiency of mapping. Shows how much of the grid had to be unveiled.
- **Time (seconds)**: Processing overhead for decision making.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.10+ installed.

### Installation
1. Clone or download the repository.
2. Open a terminal in the root folder and install the required numerical/graphing dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Simulations

#### 1. Interactive Demo
Watch a single agent dynamically generate a map and traverse step-by-step directly in your console:
```bash
python main.py
```

#### 2. Statistical Analysis
Run multiple variations of randomly generated mazes across all distinct policies simultaneously. This calculates long-term means and exports statistical plots to the `/experiments` directory:
```bash
python analysis.py
```

---

## 📂 Project Architecture
```text
├── env/
│   └── gridworld.py     # Fog-of-war movement mechanics
├── experiments/         # Generated benchmarking visual charts
├── agents.py            # Standard agent actor module
├── policies.py          # State/Logic handlers for algorithms
├── worlds.py            # Perfect maze generation (Recursive Backtracking)
├── simulations.py       # Standalone and Batch wrappers
├── analysis.py          # High-level data aggregation & matplotlib graphing
└── main.py              # Visual entry point
```