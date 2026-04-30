import numpy as np
import random
import heapq

def empty_grid(size):
    return np.zeros((size, size), dtype=int)

def random_grid(size, wall_prob=0.2, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    grid = np.random.rand(size, size) < wall_prob
    return grid.astype(int)

def maze_grid(size, seed=None):
    """
    Generates a guaranteed solvable maze using Recursive Backtracking.
    1 represents a wall, 0 represents an open path.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Start with a grid full of walls
    grid = np.ones((size, size), dtype=int)
    
    # Starting point for generation
    start_r, start_c = 0, 0
    grid[start_r, start_c] = 0
    
    stack = [(start_r, start_c)]
    
    while stack:
        r, c = stack[-1]
        
        # Directions: (dr, dc) for jumping 2 cells
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)
        
        moved = False
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check if neighbor is within bounds and is a wall (unvisited)
            if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 1:
                # Knock down the wall between current and neighbor
                wall_r, wall_c = r + dr // 2, c + dc // 2
                grid[wall_r, wall_c] = 0
                grid[nr, nc] = 0
                
                stack.append((nr, nc))
                moved = True
                break
                
        if not moved:
            stack.pop()
            
    # Guarantee that the goal (bottom-right) is accessible.
    grid[size-1, size-1] = 0
    if size > 1:
        if grid[size-2, size-1] == 1 and grid[size-1, size-2] == 1:
             grid[size-2, size-1] = 0

    return grid

def imperfect_maze_grid(size, wall_knockdown_prob=0.08, mud_prob=0.15, seed=None):
    """
    Generates an imperfect maze with multiple paths and weighted terrain.
    0 = Fast floor (cost 1)
    1 = Wall (impassable)
    2 = Mud / Slow terrain (cost 3)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    grid = maze_grid(size, seed)
    
    # Knock down some internal walls to create loops
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if grid[r, c] == 1:
                if random.random() < wall_knockdown_prob:
                    grid[r, c] = 0
                    
    # Add mud/slow terrain to open paths
    for r in range(size):
        for c in range(size):
            if grid[r, c] == 0:
                # Don't put mud on the start or goal
                if (r, c) == (0, 0) or (r, c) == (size-1, size-1):
                    continue
                if random.random() < mud_prob:
                    grid[r, c] = 2
                    
    return grid

def omniscient_dijkstra(grid, start, goal):
    """
    Finds the true optimal path cost seeing the whole map.
    Returns (optimal_cost, optimal_path_length).
    """
    size = len(grid)
    pq = [(0, start, [start])] # (cost, pos, path)
    visited = set()
    
    while pq:
        cost, pos, path = heapq.heappop(pq)
        
        if pos == goal:
            return cost, len(path) - 1
            
        if pos in visited:
            continue
        visited.add(pos)
        
        r, c = pos
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                cell_type = grid[nr, nc]
                if cell_type != 1: # Not a wall
                    step_cost = 3 if cell_type == 2 else 1
                    heapq.heappush(pq, (cost + step_cost, (nr, nc), path + [(nr, nc)]))
                    
    return float('inf'), 0