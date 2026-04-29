import numpy as np
import random

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