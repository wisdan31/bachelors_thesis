import time
from worlds import omniscient_dijkstra

class Simulation:
    def __init__(self, agent, env, max_steps=10000):
        self.agent = agent
        self.env = env
        self.max_steps = max_steps
        self.visited_nodes = []
        self.metrics = {}

    def run(self):
        raise NotImplementedError
    
class SimpleSimulation(Simulation):
    def run(self):
        steps = 0
        total_cost = 0
        start_time = time.time()
        
        while not self.env.is_terminal() and steps < self.max_steps:
            observation = self.env.observe()
            current_node = observation[0]
            
            if not self.visited_nodes or self.visited_nodes[-1] != current_node:
                self.visited_nodes.append(current_node)
                
            action = self.agent.act(observation)
            if action is None:
                # Agent got stuck or gave up
                break
                
            _, step_cost = self.env.step(action)
            steps += 1
            total_cost += step_cost

        end_time = time.time()
        
        success = self.env.is_terminal()
        unique_nodes_visited = len(set(self.visited_nodes))
        
        self.metrics = {
            "success": success,
            "steps": steps,
            "total_cost": total_cost,
            "unique_nodes": unique_nodes_visited,
            "time_seconds": end_time - start_time
        }
        
        return self.metrics

class BatchSimulation:
    def __init__(self, env_factory, agent_factories, num_runs=10):
        """
        env_factory: callable that returns a new GridEnv
        agent_factories: dict of name -> callable that returns a new Agent
        """
        self.env_factory = env_factory
        self.agent_factories = agent_factories
        self.num_runs = num_runs
        self.results = {name: [] for name in agent_factories.keys()}
        
    def run(self):
        for run_idx in range(self.num_runs):
            # Create a shared environment configuration for this run so all agents face the same maze
            base_env = self.env_factory(run_idx)
            
            # Compute omniscient optimal path as a benchmark
            opt_cost, opt_len = omniscient_dijkstra(base_env.grid, base_env.start, base_env.goal)
            
            for name, agent_factory in self.agent_factories.items():
                import copy
                env = copy.deepcopy(base_env)
                agent = agent_factory()
                
                sim = SimpleSimulation(agent, env)
                metrics = sim.run()
                
                # Append benchmark comparison
                metrics["optimal_cost"] = opt_cost
                metrics["optimal_len"] = opt_len
                metrics["cost_ratio"] = metrics["total_cost"] / opt_cost if opt_cost > 0 else 1.0
                
                self.results[name].append(metrics)
                
        return self.results