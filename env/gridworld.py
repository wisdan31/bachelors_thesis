class GridEnv:
    def __init__(self, grid, start, goal, size):
        self.size = size
        self.grid = grid
        self.start = start
        self.goal = goal
        self.agent_pos = list(start)

    def get_valid_neighbors(self, pos):
        r, c = pos
        neighbors = {}
        moves = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }
        for action, (dr, dc) in moves.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                cell_val = self.grid[nr][nc]
                if cell_val != 1:  # 1 is wall
                    cost = 3 if cell_val == 2 else 1
                    neighbors[action] = {"pos": (nr, nc), "cost": cost}
        return neighbors

    def get_visible_cells(self, pos):
        r, c = pos
        visible = [(r, c)]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                visible.append((nr, nc))
        return visible

    def observe(self):
        # In an "unknown environment", the agent only sees what is immediately accessible
        current_pos = tuple(self.agent_pos)
        valid_neighbors = self.get_valid_neighbors(current_pos)
        return current_pos, valid_neighbors
    
    def is_terminal(self):
        return tuple(self.agent_pos) == self.goal

    def step(self, action):
        current_pos = tuple(self.agent_pos)
        valid_neighbors = self.get_valid_neighbors(current_pos)
        
        step_cost = 0
        if action in valid_neighbors:
            self.agent_pos = list(valid_neighbors[action]["pos"])
            step_cost = valid_neighbors[action]["cost"]
            
        return self.observe(), step_cost