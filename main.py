import sys
from env.gridworld import GridEnv
from agents import Agent 
from policies import BFSPolicy, AStarPolicy, DFSPolicy
from simulations import SimpleSimulation
from worlds import maze_grid
import time

def print_grid_with_agent(grid, agent_pos, visited):
    size = len(grid)
    for r in range(size):
        row_str = ""
        for c in range(size):
            if (r, c) == tuple(agent_pos):
                row_str += "A "
            elif grid[r][c] == 1:
                row_str += "# "
            elif (r, c) in visited:
                row_str += "x "
            else:
                row_str += ". "
        print(row_str)
    print()

def run_visual_demo():
    size = 15
    print(f"Generating {size}x{size} maze...")
    grid = maze_grid(size, seed=42)
    start = (0, 0)
    goal = (size-1, size-1)
    
    env = GridEnv(grid, start, goal, size)
    
    # Choose your agent here
    agent = Agent(AStarPolicy(goal_pos=goal))
    print("Using A* Policy Agent")
    
    sim = SimpleSimulation(agent, env)
    
    # Step by step simulation visualization
    print("Starting visual simulation (first 20 steps)...")
    steps = 0
    while not env.is_terminal() and steps < 20:
        observation = env.observe()
        current_node = observation[0]
        
        if not sim.visited_nodes or sim.visited_nodes[-1] != current_node:
            sim.visited_nodes.append(current_node)
            
        print(f"Step {steps}, Position: {current_node}")
        print_grid_with_agent(grid, env.agent_pos, set(sim.visited_nodes))
        
        action = agent.act(observation)
        if action is None:
            print("Agent gave up!")
            break
            
        env.step(action)
        steps += 1
        time.sleep(0.1) # Pause for visual effect
        
    print("\nVisual demonstration finished.")
    print("To run the full batch analysis and generate graphs, run `python analysis.py`")

if __name__ == "__main__":
    run_visual_demo()