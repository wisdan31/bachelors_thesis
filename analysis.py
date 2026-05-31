import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import pygame

from env.gridworld import GridEnv
from worlds import imperfect_maze_grid, omniscient_dijkstra
from agents import Agent
from policies import (RandomPolicy, DFSPolicy, BFSPolicy,
                      GreedyBestFirstPolicy, LRTAStarPolicy,
                      FSA_AStarPolicy, DStarLitePolicy)
from simulations import BatchSimulation

BG_COLOR      = (25, 25, 30)
PANEL_BG      = (35, 35, 45)
WHITE         = (240, 240, 240)
LIGHT_GRAY    = (180, 180, 180)
ACCENT_BLUE   = (60, 130, 200)
ACCENT_ORANGE = (220, 150, 50)
ACCENT_GREEN  = (60, 180, 100)
BTN_NORMAL    = (55, 55, 70)
BTN_HOVER     = (75, 75, 95)
BTN_SELECTED  = (60, 130, 200)
BTN_START     = (40, 160, 80)
BTN_START_HVR = (50, 200, 100)
BTN_DANGER    = (160, 50, 50)
BTN_DANGER_H  = (200, 70, 70)

ALL_AGENTS = [
    "Random", "DFS", "BFS", "Greedy Best-First",
    "LRTA*", "FSA-A*", "D* Lite"
]

AGENT_PALETTE = {
    "Random":            "#b36432",
    "DFS":               "#328cb4",
    "BFS":               "#32b464",
    "Greedy Best-First": "#c8b432",
    "LRTA*":             "#9632c8",
    "FSA-A*":            "#32c896",
    "D* Lite":           "#c83296",
}


class Button:
    def __init__(self, x, y, w, h, text, font,
                 color=BTN_NORMAL, hover_color=BTN_HOVER,
                 selected_color=BTN_SELECTED, text_color=WHITE):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font = font
        self.color = color
        self.hover_color = hover_color
        self.selected_color = selected_color
        self.text_color = text_color
        self.selected = False
        self.enabled = True

    def draw(self, surface, mouse_pos):
        if not self.enabled:
            col = (40, 40, 50)
        elif self.selected:
            col = self.selected_color
        elif self.rect.collidepoint(mouse_pos):
            col = self.hover_color
        else:
            col = self.color
        pygame.draw.rect(surface, col, self.rect, border_radius=6)
        pygame.draw.rect(surface, (100, 100, 110), self.rect, 2, border_radius=6)
        txt_surf = self.font.render(self.text, True, self.text_color)
        surface.blit(txt_surf, (self.rect.centerx - txt_surf.get_width() // 2,
                                self.rect.centery - txt_surf.get_height() // 2))

    def clicked(self, mouse_pos):
        return self.enabled and self.rect.collidepoint(mouse_pos)


def build_agent_factories(selected_agents, maze_size):
    goal_pos = (maze_size - 1, maze_size - 1)
    factories = {}
    mapping = {
        "Random":            lambda: Agent(RandomPolicy(goal_pos)),
        "DFS":               lambda: Agent(DFSPolicy(goal_pos)),
        "BFS":               lambda: Agent(BFSPolicy(goal_pos)),
        "Greedy Best-First": lambda: Agent(GreedyBestFirstPolicy(goal_pos)),
        "LRTA*":             lambda: Agent(LRTAStarPolicy(goal_pos)),
        "FSA-A*":            lambda: Agent(FSA_AStarPolicy(goal_pos, grid_size=maze_size)),
        "D* Lite":           lambda: Agent(DStarLitePolicy(goal_pos, grid_size=maze_size)),
    }
    for name in selected_agents:
        factories[name] = mapping[name]
    return factories


def generate_all_plots(results, maze_size, num_runs, out_dir):
    sns.set_theme(style="darkgrid", palette="muted")

    records = []
    all_latencies = {}

    for agent_name, runs in results.items():
        lats = []
        for run_idx, m in enumerate(runs):
            records.append({
                "Agent": agent_name,
                "Run": run_idx,
                "Success": m["success"],
                "Steps": m["steps"],
                "Total Cost": m["total_cost"],
                "Suboptimality Ratio": m.get("cost_ratio", 1.0),
                "Unique Nodes": m["unique_nodes"],
                "Time (s)": m["time_seconds"],
                "Avg Latency (ms)": m["avg_latency_ms"],
                "Max Latency (ms)": m["max_latency_ms"],
                "Peak Memory (nodes)": m["peak_memory_nodes"],
                "Re-expansions": m["total_revisits"],
                "Discovery Efficiency": m["discovery_efficiency"],
                "Backtrack Steps": m["backtrack_steps"],
                "Avg Info Gain": m["avg_info_gain"],
                "Cells Revealed": m["cells_revealed"],
            })
            lats.extend(m.get("step_latencies", []))
        all_latencies[agent_name] = [l * 1000 for l in lats]

    df = pd.DataFrame(records)

    df.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)

    summary = df.groupby("Agent").agg({
        "Success": "mean",
        "Steps": "mean",
        "Total Cost": "mean",
        "Suboptimality Ratio": "mean",
        "Avg Latency (ms)": "mean",
        "Max Latency (ms)": "mean",
        "Peak Memory (nodes)": "mean",
        "Re-expansions": "mean",
        "Discovery Efficiency": "mean",
        "Backtrack Steps": "mean",
        "Avg Info Gain": "mean",
    })
    summary.to_csv(os.path.join(out_dir, "summary.csv"))

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(summary.to_string())
    print("=" * 60)

    agents_ordered = [a for a in ALL_AGENTS if a in df["Agent"].unique()]
    palette = [AGENT_PALETTE[a] for a in agents_ordered]

    # Suboptimality ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="Agent", y="Suboptimality Ratio", order=agents_ordered,
                palette=palette, ax=ax)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label="Optimal (1.0)")
    ax.set_title(f"Suboptimality Ratio  (Agent Cost / Optimal Cost)\n{maze_size}x{maze_size} maze, {num_runs} runs", fontsize=13)
    ax.set_ylabel("Suboptimality Ratio")
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_suboptimality_ratio.png"), dpi=150)
    plt.close()

    # Solvability rate
    fig, ax = plt.subplots(figsize=(10, 5))
    solve_rates = df.groupby("Agent")["Success"].mean().reindex(agents_ordered) * 100
    bars = ax.bar(agents_ordered, solve_rates, color=palette)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Solvability Rate (%)")
    ax.set_title(f"Solvability Rate\n{maze_size}x{maze_size} maze, {num_runs} runs")
    for bar, val in zip(bars, solve_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", fontsize=10)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_solvability_rate.png"), dpi=150)
    plt.close()

    # Decision latency
    lat_records = []
    for agent, lats in all_latencies.items():
        for l in lats:
            lat_records.append({"Agent": agent, "Latency (ms)": l})
    lat_df = pd.DataFrame(lat_records)

    if not lat_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=lat_df, x="Agent", y="Latency (ms)",
                       order=agents_ordered, palette=palette, ax=ax,
                       inner="box", cut=0)
        ax.set_title(f"Decision Latency per Step\n{maze_size}x{maze_size} maze, {num_runs} runs")
        ax.set_ylabel("Latency (ms)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "2_decision_latency.png"), dpi=150)
        plt.close()

    # Peak memory
    fig, ax = plt.subplots(figsize=(10, 5))
    mem = df.groupby("Agent")["Peak Memory (nodes)"].mean().reindex(agents_ordered)
    ax.bar(agents_ordered, mem, color=palette)
    ax.set_ylabel("Peak Memory (nodes stored)")
    ax.set_title(f"Peak Memory Footprint\n{maze_size}x{maze_size} maze, {num_runs} runs")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_peak_memory.png"), dpi=150)
    plt.close()

    # Re-expansions
    fig, ax = plt.subplots(figsize=(10, 5))
    reexp = df.groupby("Agent")["Re-expansions"].mean().reindex(agents_ordered)
    ax.bar(agents_ordered, reexp, color=palette)
    ax.set_ylabel("Average Re-expansions (revisited cells)")
    ax.set_title(f"State Re-expansions\n{maze_size}x{maze_size} maze, {num_runs} runs")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_re_expansions.png"), dpi=150)
    plt.close()

    # Discovery efficiency
    fig, ax = plt.subplots(figsize=(10, 5))
    disc = df.groupby("Agent")["Discovery Efficiency"].mean().reindex(agents_ordered) * 100
    ax.bar(agents_ordered, disc, color=palette)
    ax.set_ylabel("Map Revealed (%)")
    ax.set_title(f"Discovery Efficiency (% of map revealed)\n{maze_size}x{maze_size} maze, {num_runs} runs")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_discovery_efficiency.png"), dpi=150)
    plt.close()

    # Backtracking cost
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="Agent", y="Backtrack Steps", order=agents_ordered,
                palette=palette, ax=ax)
    ax.set_title(f"Backtracking Cost (steps spent revisiting)\n{maze_size}x{maze_size} maze, {num_runs} runs")
    ax.set_ylabel("Backtrack Steps")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_backtracking_cost.png"), dpi=150)
    plt.close()

    # Info gain
    fig, ax = plt.subplots(figsize=(10, 5))
    ig = df.groupby("Agent")["Avg Info Gain"].mean().reindex(agents_ordered)
    ax.bar(agents_ordered, ig, color=palette)
    ax.set_ylabel("Avg New Cells Revealed per Step")
    ax.set_title(f"Information Gain per Step\n{maze_size}x{maze_size} maze, {num_runs} runs")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_info_gain.png"), dpi=150)
    plt.close()

    # Heatmaps
    for agent_name in agents_ordered:
        last_run = results[agent_name][-1]
        visit_counts = last_run.get("position_visit_counts", {})
        if not visit_counts:
            continue
        heatmap = np.zeros((maze_size, maze_size))
        for (r, c), count in visit_counts.items():
            heatmap[r, c] = count

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(heatmap, cmap="YlOrRd", ax=ax, square=True,
                    cbar_kws={"label": "Visit Count"})
        ax.set_title(f"State-Space Heatmap: {agent_name}\n(Run #{num_runs}, {maze_size}x{maze_size})")
        plt.tight_layout()
        safe_name = agent_name.replace("*", "star").replace(" ", "_")
        plt.savefig(os.path.join(out_dir, f"4_heatmap_{safe_name}.png"), dpi=150)
        plt.close()

    # Summary table
    fig, ax = plt.subplots(figsize=(14, 3 + len(agents_ordered) * 0.5))
    ax.axis('off')
    table_data = []
    cols = ["Agent", "Solve %", "Avg Steps", "Avg Cost", "Subopt. Ratio",
            "Avg Latency ms", "Peak Mem", "Revisits", "Discovery %", "Backtrack"]
    for agent in agents_ordered:
        s = df[df["Agent"] == agent]
        table_data.append([
            agent,
            f"{s['Success'].mean() * 100:.0f}%",
            f"{s['Steps'].mean():.0f}",
            f"{s['Total Cost'].mean():.0f}",
            f"{s['Suboptimality Ratio'].mean():.2f}",
            f"{s['Avg Latency (ms)'].mean():.3f}",
            f"{s['Peak Memory (nodes)'].mean():.0f}",
            f"{s['Re-expansions'].mean():.0f}",
            f"{s['Discovery Efficiency'].mean() * 100:.1f}%",
            f"{s['Backtrack Steps'].mean():.0f}",
        ])
    table = ax.table(cellText=table_data, colLabels=cols, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#3a3a4a")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#1e1e28" if row % 2 == 0 else "#2a2a38")
            cell.set_text_props(color="white")
        cell.set_edgecolor("#555")
    ax.set_title(f"Comprehensive Summary — {maze_size}x{maze_size} maze, {num_runs} runs",
                 fontsize=14, fontweight="bold", color="#333", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "0_summary_table.png"), dpi=150, facecolor="#f5f5f5")
    plt.close()

    print(f"\nAll plots saved to: {out_dir}/")
    return df, summary


def run_analysis_ui():
    pygame.init()

    win_w, win_h = 640, 600
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Analysis Dashboard - Configuration")

    font_title   = pygame.font.SysFont("consolas", 26, bold=True)
    font_section = pygame.font.SysFont("consolas", 17, bold=True)
    font_btn     = pygame.font.SysFont("consolas", 14, bold=True)
    font_small   = pygame.font.SysFont("consolas", 13)
    font_tiny    = pygame.font.SysFont("consolas", 12)

    agent_buttons = []
    abtn_y = 80
    for i, name in enumerate(ALL_AGENTS):
        btn = Button(20, abtn_y, 260, 32, name, font_btn)
        btn.selected = True
        agent_buttons.append(btn)
        abtn_y += 38

    select_all_btn = Button(20, abtn_y + 5, 125, 30, "Select All", font_tiny,
                            color=(50, 80, 50), hover_color=(60, 100, 60))
    deselect_all_btn = Button(155, abtn_y + 5, 125, 30, "Deselect All", font_tiny,
                              color=(80, 50, 50), hover_color=(100, 60, 60))

    maze_sizes = [11, 15, 21, 31, 41, 51]
    maze_size_idx = 2
    maze_size_btns = []
    for i, sz in enumerate(maze_sizes):
        b = Button(320 + i * 50, 90, 44, 32, str(sz), font_btn)
        if i == maze_size_idx:
            b.selected = True
        maze_size_btns.append(b)

    run_counts = [5, 10, 25, 50, 100]
    run_count_idx = 1
    run_btns = []
    for i, n in enumerate(run_counts):
        b = Button(320 + i * 55, 170, 48, 32, str(n), font_btn)
        if i == run_count_idx:
            b.selected = True
        run_btns.append(b)

    start_btn = Button(320, win_h - 80, 280, 50, "RUN ANALYSIS", font_section,
                       color=BTN_START, hover_color=BTN_START_HVR)

    running_analysis = False
    progress_text = ""
    progress_pct = 0

    clock = pygame.time.Clock()
    running = True

    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not running_analysis:
                for btn in agent_buttons:
                    if btn.clicked(mouse_pos):
                        btn.selected = not btn.selected

                if select_all_btn.clicked(mouse_pos):
                    for b in agent_buttons:
                        b.selected = True
                if deselect_all_btn.clicked(mouse_pos):
                    for b in agent_buttons:
                        b.selected = False

                for i, btn in enumerate(maze_size_btns):
                    if btn.clicked(mouse_pos):
                        maze_size_idx = i
                        for b in maze_size_btns:
                            b.selected = False
                        btn.selected = True

                for i, btn in enumerate(run_btns):
                    if btn.clicked(mouse_pos):
                        run_count_idx = i
                        for b in run_btns:
                            b.selected = False
                        btn.selected = True

                if start_btn.clicked(mouse_pos):
                    selected = [ALL_AGENTS[i] for i, b in enumerate(agent_buttons) if b.selected]
                    if selected:
                        maze_size = maze_sizes[maze_size_idx]
                        num_runs = run_counts[run_count_idx]
                        pygame.quit()
                        return selected, maze_size, num_runs

        screen.fill(BG_COLOR)

        t = font_title.render("ANALYSIS DASHBOARD", True, ACCENT_BLUE)
        screen.blit(t, (win_w // 2 - t.get_width() // 2, 15))

        sub = font_tiny.render("Configure and run batch experiments", True, LIGHT_GRAY)
        screen.blit(sub, (win_w // 2 - sub.get_width() // 2, 48))

        sec = font_section.render("AGENTS", True, ACCENT_ORANGE)
        screen.blit(sec, (20, 60))
        for btn in agent_buttons:
            btn.draw(screen, mouse_pos)
        select_all_btn.draw(screen, mouse_pos)
        deselect_all_btn.draw(screen, mouse_pos)

        sec2 = font_section.render("MAZE SIZE", True, ACCENT_ORANGE)
        screen.blit(sec2, (320, 68))
        for btn in maze_size_btns:
            btn.draw(screen, mouse_pos)

        sec3 = font_section.render("NUMBER OF RUNS", True, ACCENT_ORANGE)
        screen.blit(sec3, (320, 148))
        for btn in run_btns:
            btn.draw(screen, mouse_pos)

        sel_count = sum(1 for b in agent_buttons if b.selected)
        msz = maze_sizes[maze_size_idx]
        nruns = run_counts[run_count_idx]
        total_sims = sel_count * nruns

        info_y = 230
        info_lines = [
            f"Selected agents: {sel_count}/{len(ALL_AGENTS)}",
            f"Grid: {msz} x {msz}",
            f"Runs per agent: {nruns}",
            f"Total simulations: {total_sims}",
            "",
            "Outputs saved to: experiments/",
        ]
        for line in info_lines:
            s = font_small.render(line, True, LIGHT_GRAY)
            screen.blit(s, (320, info_y))
            info_y += 22

        info_y += 10
        met_title = font_section.render("METRICS COLLECTED", True, ACCENT_GREEN)
        screen.blit(met_title, (320, info_y))
        info_y += 25
        metric_names = [
            "Suboptimality Ratio", "Solvability Rate",
            "Decision Latency (per-step)", "Peak Memory",
            "State Re-expansions", "Discovery Efficiency",
            "Backtracking Cost", "Info Gain / Step",
            "State-Space Heatmaps",
        ]
        for mn in metric_names:
            dot = font_tiny.render(f"  {mn}", True, (150, 150, 160))
            screen.blit(dot, (325, info_y))
            info_y += 17

        if sel_count == 0:
            start_btn.enabled = False
        else:
            start_btn.enabled = True
        start_btn.draw(screen, mouse_pos)

        pygame.display.flip()
        clock.tick(30)


def run_analysis_headless(agents, maze_size, num_runs):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("experiments", f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    def env_factory(run_idx):
        grid = imperfect_maze_grid(maze_size, seed=run_idx)
        start = (0, 0)
        goal = (maze_size - 1, maze_size - 1)
        return GridEnv(grid, start, goal, maze_size)

    agent_factories = build_agent_factories(agents, maze_size)

    def on_progress(completed, total, agent_name, run_idx):
        pct = completed / total * 100
        print(f"  [{pct:5.1f}%] {agent_name} — run {run_idx + 1}", end="\r")

    print(f"\nRunning {len(agents)} agents x {num_runs} runs on {maze_size}x{maze_size} mazes")
    print(f"Output directory: {out_dir}\n")

    batch = BatchSimulation(env_factory, agent_factories, num_runs=num_runs)
    results = batch.run(progress_callback=on_progress)

    print("\n\nGenerating plots...")
    generate_all_plots(results, maze_size, num_runs, out_dir)
    print("Done!")

    config = {
        "agents": agents,
        "maze_size": maze_size,
        "num_runs": num_runs,
        "timestamp": timestamp,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    if "--headless" in sys.argv:
        run_analysis_headless(ALL_AGENTS, maze_size=21, num_runs=10)
    else:
        selected_agents, maze_size, num_runs = run_analysis_ui()
        run_analysis_headless(selected_agents, maze_size, num_runs)
