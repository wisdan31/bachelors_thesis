import sys
import time
import pygame
from env.gridworld import GridEnv
from agents import Agent 
from policies import BFSPolicy, AStarPolicy, DFSPolicy
from simulations import SimpleSimulation
from worlds import imperfect_maze_grid

# Define Color Palette
WHITE = (240, 240, 240)    # Floor
GRAY = (70, 70, 70)        # Walls
BROWN = (139, 69, 19)      # Mud
BLACK = (10, 10, 10)       # Fog of War
AGENT_COLOR = (0, 200, 100)
GOAL_COLOR = (220, 50, 50)
GRID_LINE_COLOR = (40, 40, 40)

def draw_grid(screen, grid, agent_pos, goal_pos, revealed_cells, cell_size):
    size = len(grid)
    for r in range(size):
        for c in range(size):
            rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
            
            # Fog of War Check
            if (r, c) not in revealed_cells:
                pygame.draw.rect(screen, BLACK, rect)
            else:
                # Base Terrain
                cell_val = grid[r][c]
                if cell_val == 1:
                    pygame.draw.rect(screen, GRAY, rect)
                elif cell_val == 2:
                    pygame.draw.rect(screen, BROWN, rect)
                else:
                    pygame.draw.rect(screen, WHITE, rect)
                
                # Goal overlay
                if (r, c) == tuple(goal_pos):
                    pygame.draw.rect(screen, GOAL_COLOR, rect)
                
            # Draw grid cell borders
            pygame.draw.rect(screen, GRID_LINE_COLOR, rect, 1)

    # Draw Agent
    agent_r, agent_c = agent_pos
    agent_rect = pygame.Rect(agent_c * cell_size + 4, agent_r * cell_size + 4, cell_size - 8, cell_size - 8)
    pygame.draw.rect(screen, AGENT_COLOR, agent_rect, border_radius=4)

def run_visual_demo():
    # Pygame initialization
    pygame.init()
    
    # Grid setup
    size = 21
    cell_size = 30
    width = size * cell_size
    height = size * cell_size
    padding = 20
    total_width = width * 2 + padding
    
    # Information banner height
    banner_height = 40
    
    screen = pygame.display.set_mode((total_width, height + banner_height))
    pygame.display.set_caption("Agent Exploration in Unknown Environment (Side-by-Side Comparison)")
    
    print(f"Generating {size}x{size} imperfect maze...")
    grid = imperfect_maze_grid(size, seed=42)
    start = (0, 0)
    goal = (size-1, size-1)
    
    env = GridEnv(grid, start, goal, size)
    
    # Initialize the specific Agent (Change this to test DFS or BFS)
    agent = Agent(AStarPolicy(goal_pos=goal))
    print("Using A* Policy Agent")
    
    sim = SimpleSimulation(agent, env)
    
    # Fog of war state
    revealed_cells = set()
    all_cells = set((r, c) for r in range(size) for c in range(size))
    
    clock = pygame.time.Clock()
    running = True
    steps = 0
    total_cost = 0
    font = pygame.font.SysFont("consolas", 20, bold=True)
    small_font = pygame.font.SysFont("consolas", 16, bold=True)
    
    # Initial observation
    observation = env.observe()
    
    # Uncover initial visible cells (including walls, for visual appeal)
    visible_cells = env.get_visible_cells(tuple(env.agent_pos))
    for cell in visible_cells:
        revealed_cells.add(cell)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Agent Turn Logic
        if not env.is_terminal():
            current_node = observation[0]
            if not sim.visited_nodes or sim.visited_nodes[-1] != current_node:
                sim.visited_nodes.append(current_node)
                
            action = agent.act(observation)
            
            if action is None:
                print("Agent got trapped or gave up!")
                running = False
            else:
                observation, step_cost = env.step(action)
                
                # Uncover newly visible cells around the new position
                visible_cells = env.get_visible_cells(tuple(env.agent_pos))
                for cell in visible_cells:
                    revealed_cells.add(cell)
                    
                steps += 1
                total_cost += step_cost
        
        # Rendering Engine
        screen.fill(BLACK)
        
        # Left Panel: Fog of War
        maze_surface_left = pygame.Surface((width, height))
        draw_grid(maze_surface_left, grid, env.agent_pos, goal, revealed_cells, cell_size)
        screen.blit(maze_surface_left, (0, banner_height))
        
        # Right Panel: Omniscient View
        maze_surface_right = pygame.Surface((width, height))
        draw_grid(maze_surface_right, grid, env.agent_pos, goal, all_cells, cell_size)
        screen.blit(maze_surface_right, (width + padding, banner_height))
        
        # Render top banner info
        pygame.draw.rect(screen, (30, 30, 40), (0, 0, total_width, banner_height))
        
        info_text = f"Steps: {steps} | Cost: {total_cost}"
        if env.is_terminal():
            info_text += " [GOAL REACHED!]"
            
        info_surface = font.render(info_text, True, (255, 255, 150))
        screen.blit(info_surface, (10, 10))
        
        title_left = small_font.render("AGENT VIEW (FOG OF WAR)", True, WHITE)
        screen.blit(title_left, (width // 2 - title_left.get_width() // 2, 12))
        
        title_right = small_font.render("OMNISCIENT VIEW (FULL MAZE)", True, WHITE)
        screen.blit(title_right, (width + padding + width // 2 - title_right.get_width() // 2, 12))
        
        pygame.display.flip()
        
        # Speed control
        if env.is_terminal():
            clock.tick(5) # Slow down dramatically when finished
        else:
            clock.tick(20) # 20 actions per second
            
    pygame.quit()
    print(f"\nVisual demonstration finished. Total Cost: {total_cost}, Steps: {steps}")
    print("Run `python analysis.py` to generate statistical batch metrics.")

if __name__ == "__main__":
    run_visual_demo()