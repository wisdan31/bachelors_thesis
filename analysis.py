import os
import pandas as pd
import matplotlib.pyplot as plt
from env.gridworld import GridEnv
from worlds import maze_grid
from agents import Agent
from policies import RandomPolicy, DFSPolicy, BFSPolicy, AStarPolicy
from simulations import BatchSimulation

def run_analysis(maze_size=15, num_runs=10):
    print(f"Running analysis: {num_runs} mazes of size {maze_size}x{maze_size}")
    
    def env_factory(run_idx):
        # Generate a new maze for each run_idx
        grid = maze_grid(maze_size, seed=run_idx)
        start = (0, 0)
        goal = (maze_size-1, maze_size-1)
        return GridEnv(grid, start, goal, maze_size)
        
    goal_pos = (maze_size-1, maze_size-1)
    
    agent_factories = {
        "Random": lambda: Agent(RandomPolicy(goal_pos)),
        "DFS": lambda: Agent(DFSPolicy(goal_pos)),
        "BFS": lambda: Agent(BFSPolicy(goal_pos)),
        "A*": lambda: Agent(AStarPolicy(goal_pos))
    }
    
    batch_sim = BatchSimulation(env_factory, agent_factories, num_runs=num_runs)
    results = batch_sim.run()
    
    # Process results into a DataFrame
    records = []
    for agent_name, runs in results.items():
        for run_idx, metrics in enumerate(runs):
            records.append({
                "Agent": agent_name,
                "Run": run_idx,
                "Success": metrics["success"],
                "Steps": metrics["steps"],
                "Unique Nodes": metrics["unique_nodes"],
                "Time (s)": metrics["time_seconds"]
            })
            
    df = pd.DataFrame(records)
    
    print("\n--- Summary Statistics ---")
    summary = df.groupby("Agent")[["Steps", "Unique Nodes", "Time (s)"]].mean()
    print(summary)
    
    # Ensure experiments dir exists
    os.makedirs("experiments", exist_ok=True)
    
    # Plot 1: Average Steps
    plt.figure(figsize=(10, 6))
    summary["Steps"].plot(kind="bar", color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title(f"Average Steps Taken to Solve {maze_size}x{maze_size} Maze")
    plt.ylabel("Steps")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("experiments/average_steps.png")
    print("Saved plot: experiments/average_steps.png")
    
    # Plot 2: Average Unique Nodes Explored
    plt.figure(figsize=(10, 6))
    summary["Unique Nodes"].plot(kind="bar", color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title(f"Average Unique Nodes Explored in {maze_size}x{maze_size} Maze")
    plt.ylabel("Unique Nodes")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("experiments/unique_nodes.png")
    print("Saved plot: experiments/unique_nodes.png")

if __name__ == "__main__":
    run_analysis()
