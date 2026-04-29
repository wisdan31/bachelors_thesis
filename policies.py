import collections
import heapq
import random

class Policy:
    def select(self, observation):
        raise NotImplementedError

class ExplorePolicy(Policy):
    def __init__(self, goal_pos=None):
        self.goal_pos = goal_pos
        self.graph = {} # pos -> {action: next_pos}
        self.visited = set()
        self.frontier = []
        self.frontier_set = set()
        self.path_to_execute = []
        
    def add_to_frontier(self, pos):
        raise NotImplementedError
        
    def pop_from_frontier(self):
        raise NotImplementedError
        
    def find_path_in_known_graph(self, start, target):
        queue = collections.deque([(start, [])])
        visited = {start}
        while queue:
            curr, path = queue.popleft()
            if curr == target:
                return path
            if curr in self.graph:
                for action, next_pos in self.graph[curr].items():
                    if next_pos not in visited:
                        visited.add(next_pos)
                        queue.append((next_pos, path + [action]))
        return []

    def select(self, observation):
        current_pos, valid_neighbors = observation
        
        self.graph[current_pos] = valid_neighbors
        self.visited.add(current_pos)
        
        for action, next_pos in valid_neighbors.items():
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

class AStarPolicy(ExplorePolicy):
    def __init__(self, goal_pos=None):
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