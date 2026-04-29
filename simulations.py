import time

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
                
            self.env.step(action)
            steps += 1

        end_time = time.time()
        
        success = self.env.is_terminal()
        unique_nodes_visited = len(set(self.visited_nodes))
        
        self.metrics = {
            "success": success,
            "steps": steps,
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
            
            for name, agent_factory in self.agent_factories.items():
                import copy
                # Deepcopy grid to avoid agents modifying shared state if we ever allow it,
                # though our agents don't modify the env directly.
                env = copy.deepcopy(base_env)
                agent = agent_factory()
                
                sim = SimpleSimulation(agent, env)
                metrics = sim.run()
                self.results[name].append(metrics)
                
        return self.results