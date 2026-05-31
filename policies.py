import heapq
import random

class Policy:
    def select(self, observation):
        raise NotImplementedError

class ExplorePolicy(Policy):
    def __init__(self, goal_pos=None):
        self.goal_pos = goal_pos
        self.graph = {} # pos -> {action: {"pos": next_pos, "cost": cost}}
        self.visited = set()
        self.frontier = []
        self.frontier_set = set()
        self.path_to_execute = []
        
    def add_to_frontier(self, pos):
        raise NotImplementedError
        
    def pop_from_frontier(self):
        raise NotImplementedError
        
    def find_path_in_known_graph(self, start, target):
        # Use Dijkstra to find the lowest cost path in the currently known graph
        pq = [(0, start, [])] # (cost, current_pos, path_of_actions)
        visited = set()
        
        while pq:
            cost, curr, path = heapq.heappop(pq)
            if curr == target:
                return path
                
            if curr in visited:
                continue
            visited.add(curr)
            
            if curr in self.graph:
                for action, data in self.graph[curr].items():
                    next_pos = data["pos"]
                    step_cost = data["cost"]
                    if next_pos not in visited:
                        heapq.heappush(pq, (cost + step_cost, next_pos, path + [action]))
        return []

    def select(self, observation):
        current_pos, valid_neighbors = observation
        
        self.graph[current_pos] = valid_neighbors
        self.visited.add(current_pos)
        
        for action, data in valid_neighbors.items():
            next_pos = data["pos"]
            if next_pos not in self.visited and next_pos not in self.frontier_set:
                self.add_to_frontier(next_pos)
                self.frontier_set.add(next_pos)
                
        if self.path_to_execute:
            return self.path_to_execute.pop(0)
            
        while self.frontier:
            target = self.pop_from_frontier()
            if target in self.frontier_set:
                self.frontier_set.remove(target)
            if target in self.visited:
                continue
                
            path = self.find_path_in_known_graph(current_pos, target)
            if path:
                self.path_to_execute = path
                return self.path_to_execute.pop(0)
                
        return None

class RandomPolicy(ExplorePolicy):
    def add_to_frontier(self, pos):
        self.frontier.append(pos)
    def pop_from_frontier(self):
        idx = random.randint(0, len(self.frontier) - 1)
        return self.frontier.pop(idx)

class DFSPolicy(ExplorePolicy):
    def add_to_frontier(self, pos):
        self.frontier.append(pos)
    def pop_from_frontier(self):
        return self.frontier.pop()

class BFSPolicy(ExplorePolicy):
    def add_to_frontier(self, pos):
        self.frontier.append(pos)
    def pop_from_frontier(self):
        return self.frontier.pop(0)

class GreedyBestFirstPolicy(ExplorePolicy):
    def __init__(self, goal_pos=None, **kwargs):
        super().__init__(goal_pos)
        self.frontier = [] # list of (priority, pos)
        
    def heuristic(self, pos):
        if not self.goal_pos: return 0
        return abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])
        
    def add_to_frontier(self, pos):
        priority = self.heuristic(pos)
        heapq.heappush(self.frontier, (priority, pos))
        
    def pop_from_frontier(self):
        _, pos = heapq.heappop(self.frontier)
        return pos

class LRTAStarPolicy(Policy):
    def __init__(self, goal_pos=None, **kwargs):
        self.goal_pos = goal_pos
        self.H = {} 
        
    def get_h(self, pos):
        if pos == self.goal_pos:
            return 0
        if pos not in self.H:
            self.H[pos] = abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])
        return self.H[pos]
        
    def select(self, observation):
        current_pos, valid_neighbors = observation
        if current_pos == self.goal_pos:
            return None
            
        if not valid_neighbors:
            return None
            
        best_f = float('inf')
        best_action = None
        
        for action, data in valid_neighbors.items():
            next_pos = data["pos"]
            step_cost = data["cost"]
            f = step_cost + self.get_h(next_pos)
            
            if f < best_f:
                best_f = f
                best_action = action
                
        # LRTA* Update Rule
        self.H[current_pos] = max(self.get_h(current_pos), best_f)
        
        return best_action


class FSA_AStarPolicy(Policy):
    def __init__(self, goal_pos, grid_size, **kwargs):
        self.goal_pos = goal_pos
        self.grid_size = grid_size
        self.known_walls = set()
        self.known_mud = set()
        self.path_to_execute = []

    def heuristic(self, pos):
        return abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])

    def plan(self, start):
        pq = [(self.heuristic(start), 0, start, [])]
        visited = set()
        
        while pq:
            _, g, curr, path = heapq.heappop(pq)
            if curr == self.goal_pos:
                return path
            
            if curr in visited:
                continue
            visited.add(curr)
            
            r, c = curr
            moves = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
            for action, (dr, dc) in moves.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    next_pos = (nr, nc)
                    if next_pos in self.known_walls:
                        continue
                    
                    step_cost = 3 if next_pos in self.known_mud else 1
                    
                    if next_pos not in visited:
                        new_g = g + step_cost
                        f = new_g + self.heuristic(next_pos)
                        heapq.heappush(pq, (f, new_g, next_pos, path + [action]))
        return []

    def select(self, observation):
        current_pos, valid_neighbors = observation
        
        discrepancy = False
        r, c = current_pos
        moves = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
        for action, (dr, dc) in moves.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                adj_pos = (nr, nc)
                if action in valid_neighbors:
                    true_cost = valid_neighbors[action]["cost"]
                    if true_cost == 3 and adj_pos not in self.known_mud:
                        self.known_mud.add(adj_pos)
                        discrepancy = True
                    elif true_cost == 1 and adj_pos in self.known_mud:
                        self.known_mud.remove(adj_pos)
                        discrepancy = True
                else:
                    if adj_pos not in self.known_walls:
                        self.known_walls.add(adj_pos)
                        discrepancy = True
                        
        if discrepancy or not self.path_to_execute:
            self.path_to_execute = self.plan(current_pos)
            
        if self.path_to_execute:
            next_action = self.path_to_execute[0]
            if next_action in valid_neighbors:
                return self.path_to_execute.pop(0)
            else:
                self.path_to_execute = self.plan(current_pos)
                if self.path_to_execute:
                    return self.path_to_execute.pop(0)
                
        return None


class DStarLitePolicy(Policy):
    def __init__(self, goal_pos, grid_size, **kwargs):
        self.goal_pos = goal_pos
        self.start_pos = None
        self.grid_size = grid_size
        self.known_walls = set()
        self.known_mud = set()
        
        self.U = {}
        self.rhs = {}
        self.g = {}
        
        self.km = 0
        self.last_pos = None
        self.initialized = False
        
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
    def calculate_key(self, pos):
        min_g_rhs = min(self.g.get(pos, float('inf')), self.rhs.get(pos, float('inf')))
        return (min_g_rhs + self.heuristic(self.start_pos, pos) + self.km, min_g_rhs)
        
    def get_cost(self, u, v):
        if v in self.known_walls or u in self.known_walls:
            return float('inf')
        if v in self.known_mud:
            return 3
        return 1

    def get_neighbors(self, pos):
        r, c = pos
        res = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                res.append((nr, nc))
        return res

    def initialize(self, start):
        self.start_pos = start
        self.last_pos = start
        self.rhs[self.goal_pos] = 0
        self.U[self.goal_pos] = self.calculate_key(self.goal_pos)
        self.initialized = True
        self.compute_shortest_path()

    def update_vertex(self, u):
        if u != self.goal_pos:
            self.rhs[u] = min((self.get_cost(u, v) + self.g.get(v, float('inf')) for v in self.get_neighbors(u)), default=float('inf'))
        if u in self.U:
            del self.U[u]
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            self.U[u] = self.calculate_key(u)

    def top_key(self):
        if not self.U:
            return (float('inf'), float('inf'))
        return min(self.U.values())

    def compute_shortest_path(self):
        while True:
            k_old = self.top_key()
            if not self.U:
                break
            u = min(self.U.keys(), key=lambda k: self.U[k])
            
            if k_old >= self.calculate_key(self.start_pos) and self.rhs.get(self.start_pos, float('inf')) == self.g.get(self.start_pos, float('inf')):
                break
                
            k_new = self.calculate_key(u)
            if k_old < k_new:
                self.U[u] = k_new
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs.get(u, float('inf'))
                del self.U[u]
                for v in self.get_neighbors(u):
                    if v != self.goal_pos:
                        self.rhs[v] = min(self.rhs.get(v, float('inf')), self.get_cost(v, u) + self.g[u])
                    self.update_vertex(v)
            else:
                g_old = self.g.get(u, float('inf'))
                self.g[u] = float('inf')
                for v in self.get_neighbors(u) + [u]:
                    if self.rhs.get(v, float('inf')) == self.get_cost(v, u) + g_old:
                        if v != self.goal_pos:
                            self.rhs[v] = min((self.get_cost(v, s) + self.g.get(s, float('inf')) for s in self.get_neighbors(v)), default=float('inf'))
                    self.update_vertex(v)

    def select(self, observation):
        current_pos, valid_neighbors = observation
        if not self.initialized:
            self.initialize(current_pos)
            
        if self.start_pos != current_pos:
            self.start_pos = current_pos
            
        discrepancy = False
        r, c = current_pos
        moves = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
        for action, (dr, dc) in moves.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                adj_pos = (nr, nc)
                if action in valid_neighbors:
                    true_cost = valid_neighbors[action]["cost"]
                    if true_cost == 3 and adj_pos not in self.known_mud:
                        self.known_mud.add(adj_pos)
                        discrepancy = True
                        self.update_vertex(adj_pos)
                        for n in self.get_neighbors(adj_pos):
                            self.update_vertex(n)
                    elif true_cost == 1 and adj_pos in self.known_mud:
                        self.known_mud.remove(adj_pos)
                        discrepancy = True
                        self.update_vertex(adj_pos)
                        for n in self.get_neighbors(adj_pos):
                            self.update_vertex(n)
                else:
                    if adj_pos not in self.known_walls:
                        self.known_walls.add(adj_pos)
                        discrepancy = True
                        self.update_vertex(adj_pos)
                        for n in self.get_neighbors(adj_pos):
                            self.update_vertex(n)

        if discrepancy:
            self.km += self.heuristic(self.last_pos, self.start_pos)
            self.last_pos = self.start_pos
            self.compute_shortest_path()
            
        best_cost = float('inf')
        best_action = None
        for action, data in valid_neighbors.items():
            adj_pos = data["pos"]
            edge_cost = data["cost"]
            cost = edge_cost + self.g.get(adj_pos, float('inf'))
            if cost < best_cost:
                best_cost = cost
                best_action = action
                
        return best_action