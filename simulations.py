import time
import copy
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

        step_latencies = []
        position_visit_counts = {}
        revealed_cells = set()
        cells_revealed_per_step = []
        backtrack_steps = 0
        previous_pos = None

        while not self.env.is_terminal() and steps < self.max_steps:
            observation = self.env.observe()
            current_pos = observation[0]

            if not self.visited_nodes or self.visited_nodes[-1] != current_pos:
                self.visited_nodes.append(current_pos)

            position_visit_counts[current_pos] = position_visit_counts.get(current_pos, 0) + 1

            visible_now = set(self.env.get_visible_cells(current_pos))
            new_reveals = visible_now - revealed_cells
            cells_revealed_per_step.append(len(new_reveals))
            revealed_cells.update(visible_now)

            if previous_pos is not None and position_visit_counts.get(current_pos, 0) > 1:
                backtrack_steps += 1

            t0 = time.perf_counter()
            action = self.agent.act(observation)
            t1 = time.perf_counter()
            step_latencies.append(t1 - t0)

            if action is None:
                break

            _, step_cost = self.env.step(action)
            previous_pos = current_pos
            steps += 1
            total_cost += step_cost

        end_time = time.time()

        success = self.env.is_terminal()
        unique_nodes_visited = len(set(self.visited_nodes))

        total_revisits = sum(max(0, v - 1) for v in position_visit_counts.values())
        total_cells_in_grid = self.env.size * self.env.size
        total_revealed = len(revealed_cells)

        discovery_efficiency = total_revealed / total_cells_in_grid if total_cells_in_grid > 0 else 0
        avg_info_gain = sum(cells_revealed_per_step) / len(cells_revealed_per_step) if cells_revealed_per_step else 0
        peak_memory = self._estimate_peak_memory()

        self.metrics = {
            "success": success,
            "steps": steps,
            "total_cost": total_cost,
            "unique_nodes": unique_nodes_visited,
            "time_seconds": end_time - start_time,
            "step_latencies": step_latencies,
            "avg_latency_ms": (sum(step_latencies) / len(step_latencies) * 1000) if step_latencies else 0,
            "max_latency_ms": (max(step_latencies) * 1000) if step_latencies else 0,
            "peak_memory_nodes": peak_memory,
            "total_revisits": total_revisits,
            "discovery_efficiency": discovery_efficiency,
            "backtrack_steps": backtrack_steps,
            "avg_info_gain": avg_info_gain,
            "cells_revealed": total_revealed,
            "position_visit_counts": position_visit_counts,
            "cells_revealed_per_step": cells_revealed_per_step,
        }

        return self.metrics

    def _estimate_peak_memory(self):
        policy = self.agent.policy
        count = 0
        if hasattr(policy, 'graph'):
            count += len(policy.graph)
        if hasattr(policy, 'frontier'):
            count += len(policy.frontier)
        if hasattr(policy, 'frontier_set'):
            count += len(policy.frontier_set)
        if hasattr(policy, 'visited'):
            count += len(policy.visited)
        if hasattr(policy, 'H'):
            count += len(policy.H)
        if hasattr(policy, 'known_walls'):
            count += len(policy.known_walls)
        if hasattr(policy, 'known_mud'):
            count += len(policy.known_mud)
        if hasattr(policy, 'U'):
            count += len(policy.U)
        if hasattr(policy, 'g') and isinstance(policy.g, dict):
            count += len(policy.g)
        if hasattr(policy, 'rhs') and isinstance(policy.rhs, dict):
            count += len(policy.rhs)
        return count


class BatchSimulation:
    def __init__(self, env_factory, agent_factories, num_runs=10):
        self.env_factory = env_factory
        self.agent_factories = agent_factories
        self.num_runs = num_runs
        self.results = {name: [] for name in agent_factories.keys()}

    def run(self, progress_callback=None):
        total_tasks = self.num_runs * len(self.agent_factories)
        completed = 0

        for run_idx in range(self.num_runs):
            base_env = self.env_factory(run_idx)
            opt_cost, opt_len = omniscient_dijkstra(base_env.grid, base_env.start, base_env.goal)

            for name, agent_factory in self.agent_factories.items():
                env = copy.deepcopy(base_env)
                agent = agent_factory()

                sim = SimpleSimulation(agent, env)
                metrics = sim.run()

                metrics["optimal_cost"] = opt_cost
                metrics["optimal_len"] = opt_len
                metrics["cost_ratio"] = metrics["total_cost"] / opt_cost if opt_cost > 0 else 1.0

                self.results[name].append(metrics)
                completed += 1

                if progress_callback:
                    progress_callback(completed, total_tasks, name, run_idx)

        return self.results