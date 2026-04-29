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
                if self.grid[nr][nc] == 0:  # 0 is open path, 1 is wall
                    neighbors[action] = (nr, nc)
        return neighbors

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
        
        if action in valid_neighbors:
            self.agent_pos = list(valid_neighbors[action])
            
        return self.observe()