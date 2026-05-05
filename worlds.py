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


# ============================================================
# PRESET MAZES - Each tests a specific aspect of agent behavior
# ============================================================

def preset_narrow_corridors(size, seed=7):
    """
    Preset 1: Narrow Corridors (Perfect Maze)
    A pure recursive backtracking maze with NO walls knocked down.
    Only one path exists between any two points.
    Tests: DFS excels here (follows the single winding path).
           BFS wastes enormous effort exploring all dead ends.
    """
    return maze_grid(size, seed=seed)

def preset_open_arena(size, seed=12):
    """
    Preset 2: Open Arena
    A mostly open grid with scattered pillar-like obstacles.
    Tests: BFS finds the shortest path easily since the space is wide open.
           DFS may wander far from the goal in the open space.
           A* should perform best due to its heuristic in open terrain.
    """
    random.seed(seed)
    np.random.seed(seed)
    grid = np.zeros((size, size), dtype=int)
    
    # Place scattered pillar clusters
    for _ in range(size * size // 8):
        r = random.randint(1, size - 2)
        c = random.randint(1, size - 2)
        grid[r, c] = 1
        # Sometimes extend pillar into an L-shape
        if random.random() < 0.4:
            dr, dc = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            nr, nc = r + dr, c + dc
            if 1 <= nr < size - 1 and 1 <= nc < size - 1:
                grid[nr, nc] = 1
    
    # Always keep start and goal clear
    grid[0, 0] = 0
    grid[size-1, size-1] = 0
    # Clear a small area around start and goal
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            for pr, pc in [(0, 0), (size-1, size-1)]:
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < size and 0 <= nc < size:
                    grid[nr, nc] = 0
    return grid

def preset_heavy_mud(size, seed=42):
    """
    Preset 3: Heavy Mud Swamp
    A maze where the direct path is flooded with mud (cost 3),
    but longer alternative corridors are clean (cost 1).
    Tests: A* with cost-awareness should find the cheaper detour.
           Greedy agents (DFS) will walk straight through mud.
    """
    random.seed(seed)
    np.random.seed(seed)
    grid = maze_grid(size, seed=seed)
    
    # Knock down some walls to create alternative routes
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if grid[r, c] == 1 and random.random() < 0.12:
                grid[r, c] = 0
    
    # Flood a diagonal band with mud (the "direct" path area)
    for r in range(size):
        for c in range(size):
            if grid[r, c] == 0 and (r, c) != (0, 0) and (r, c) != (size-1, size-1):
                # Distance from the main diagonal
                dist_from_diag = abs(r - c)
                if dist_from_diag < size // 3:
                    if random.random() < 0.55:
                        grid[r, c] = 2
    
    return grid

def preset_spiral_trap(size, seed=99):
    """
    Preset 4: Spiral Trap
    A spiral maze that forces agents inward before letting them reach the goal.
    The goal appears close (Manhattan distance is small) but the actual path
    spirals around the entire grid.
    Tests: A* heuristic gets "tricked" by proximity — it keeps trying to go
           straight but hits walls. DFS may find the spiral path faster.
    """
    grid = np.ones((size, size), dtype=int)
    
    # Carve a spiral path
    r, c = 0, 0
    grid[r, c] = 0
    
    top, bottom, left, right = 0, size - 1, 0, size - 1
    
    while top <= bottom and left <= right:
        # Go right along top
        for c in range(left, right + 1):
            if 0 <= c < size and 0 <= top < size:
                grid[top, c] = 0
        top += 2
        
        # Go down along right
        for r in range(top - 1, bottom + 1):
            if 0 <= r < size and 0 <= right < size:
                grid[r, right] = 0
        right -= 2
        
        # Go left along bottom
        if top <= bottom:
            for c in range(right + 1, left - 1, -1):
                if 0 <= c < size and 0 <= bottom < size:
                    grid[bottom, c] = 0
            bottom -= 2
        
        # Go up along left
        if left <= right:
            for r in range(bottom + 1, top - 1, -1):
                if 0 <= r < size and 0 <= left < size:
                    grid[r, left] = 0
            left += 2
    
    grid[0, 0] = 0
    grid[size-1, size-1] = 0
    # Ensure goal is connected — open a cell next to it if needed
    if size > 1:
        if grid[size-2, size-1] == 1 and grid[size-1, size-2] == 1:
            grid[size-1, size-2] = 0
    
    return grid

def preset_dense_multipath(size, seed=55):
    """
    Preset 5: Dense Multi-Path Network
    A maze with many walls knocked down (25%), creating a dense network
    of interconnected corridors with many possible routes.
    Tests: Agents with smarter frontier prioritization (A*) should
           navigate efficiently. BFS explores too many branches.
           Random agent gets lost in the abundance of choices.
    """
    random.seed(seed)
    np.random.seed(seed)
    grid = maze_grid(size, seed=seed)
    
    # Aggressively knock down walls
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if grid[r, c] == 1 and random.random() < 0.25:
                grid[r, c] = 0
    
    # Sprinkle some mud patches for cost variety
    for r in range(size):
        for c in range(size):
            if grid[r, c] == 0 and (r, c) != (0, 0) and (r, c) != (size-1, size-1):
                if random.random() < 0.08:
                    grid[r, c] = 2
    
    return grid


# Registry of all presets for easy access
MAZE_PRESETS = {
    "Narrow Corridors": {
        "fn": preset_narrow_corridors,
        "desc": "Perfect maze, single path. Tests DFS vs BFS."
    },
    "Open Arena": {
        "fn": preset_open_arena,
        "desc": "Wide open space with pillars. Tests heuristic navigation."
    },
    "Heavy Mud Swamp": {
        "fn": preset_heavy_mud,
        "desc": "Diagonal mud band. Tests cost-aware pathfinding."
    },
    "Spiral Trap": {
        "fn": preset_spiral_trap,
        "desc": "Spiral path. Tricks proximity-based heuristics."
    },
    "Dense Multi-Path": {
        "fn": preset_dense_multipath,
        "desc": "Many routes available. Tests decision-making."
    },
}