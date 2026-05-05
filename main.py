import sys
import random
import pygame
from env.gridworld import GridEnv
from agents import Agent
from policies import BFSPolicy, AStarPolicy, DFSPolicy, RandomPolicy
from simulations import SimpleSimulation
from worlds import imperfect_maze_grid, MAZE_PRESETS

# ============================================================
# COLOR PALETTE
# ============================================================
BG_COLOR       = (25, 25, 30)
PANEL_BG       = (35, 35, 45)
WHITE          = (240, 240, 240)
LIGHT_GRAY     = (180, 180, 180)
GRAY           = (70, 70, 70)
BROWN          = (139, 69, 19)
BLACK          = (10, 10, 10)
AGENT_COLOR    = (0, 200, 100)
GOAL_COLOR     = (220, 50, 50)
START_COLOR    = (80, 180, 255)
GRID_LINE      = (40, 40, 40)
ACCENT_BLUE    = (60, 130, 200)
ACCENT_GREEN   = (60, 180, 100)
ACCENT_ORANGE  = (220, 150, 50)
BTN_NORMAL     = (55, 55, 70)
BTN_HOVER      = (75, 75, 95)
BTN_SELECTED   = (60, 130, 200)
BTN_START      = (40, 160, 80)
BTN_START_HVR  = (50, 200, 100)

# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_grid_surface(grid, goal_pos, start_pos, cell_size, revealed_cells=None, agent_pos=None):
    """Render a grid onto a new surface. If revealed_cells is None, show everything."""
    size = len(grid)
    surface = pygame.Surface((size * cell_size, size * cell_size))
    surface.fill(BLACK)

    for r in range(size):
        for c in range(size):
            rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)

            if revealed_cells is not None and (r, c) not in revealed_cells:
                pygame.draw.rect(surface, BLACK, rect)
            else:
                cell_val = grid[r][c]
                if cell_val == 1:
                    pygame.draw.rect(surface, GRAY, rect)
                elif cell_val == 2:
                    pygame.draw.rect(surface, BROWN, rect)
                else:
                    pygame.draw.rect(surface, WHITE, rect)

                if (r, c) == tuple(goal_pos):
                    pygame.draw.rect(surface, GOAL_COLOR, rect)
                elif (r, c) == tuple(start_pos):
                    pygame.draw.rect(surface, START_COLOR, rect)

            pygame.draw.rect(surface, GRID_LINE, rect, 1)

    if agent_pos is not None:
        ar, ac = agent_pos
        agent_rect = pygame.Rect(ac * cell_size + 3, ar * cell_size + 3, cell_size - 6, cell_size - 6)
        pygame.draw.rect(surface, AGENT_COLOR, agent_rect, border_radius=4)

    return surface


class Button:
    def __init__(self, x, y, w, h, text, font, color=BTN_NORMAL, hover_color=BTN_HOVER,
                 selected_color=BTN_SELECTED, text_color=WHITE):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font = font
        self.color = color
        self.hover_color = hover_color
        self.selected_color = selected_color
        self.text_color = text_color
        self.selected = False

    def draw(self, surface, mouse_pos):
        if self.selected:
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
        return self.rect.collidepoint(mouse_pos)


# ============================================================
# MENU SCREEN
# ============================================================

def run_menu():
    pygame.init()

    MAZE_SIZE = 21
    PREVIEW_CELL = 14  # small cells for preview
    preview_px = MAZE_SIZE * PREVIEW_CELL

    # Layout
    left_panel_w = 340
    right_panel_w = preview_px + 40
    win_w = left_panel_w + right_panel_w
    win_h = max(620, preview_px + 200)

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Agent Explorer - Setup")

    font_title = pygame.font.SysFont("consolas", 28, bold=True)
    font_section = pygame.font.SysFont("consolas", 18, bold=True)
    font_btn = pygame.font.SysFont("consolas", 15, bold=True)
    font_desc = pygame.font.SysFont("consolas", 13)
    font_small = pygame.font.SysFont("consolas", 12)

    # --- Maze buttons ---
    preset_names = list(MAZE_PRESETS.keys())
    maze_buttons = []
    btn_y = 100
    for name in preset_names:
        maze_buttons.append(Button(20, btn_y, 300, 36, name, font_btn))
        btn_y += 44
    # Random maze button
    random_btn = Button(20, btn_y, 300, 36, "Random Maze", font_btn,
                        color=(80, 60, 40), hover_color=(110, 80, 50))
    btn_y += 54

    # --- Agent buttons ---
    agent_options = ["Random", "DFS", "BFS", "A*"]
    agent_colors = {
        "Random": (180, 100, 50),
        "DFS":    (50, 140, 180),
        "BFS":    (50, 180, 100),
        "A*":     (200, 180, 50),
    }
    agent_buttons = []
    abtn_y = btn_y + 30
    for name in agent_options:
        agent_buttons.append(Button(20, abtn_y, 145, 36, name, font_btn))
        abtn_y += 44

    # --- Start button ---
    start_btn = Button(20, win_h - 70, 300, 50, "START SIMULATION", font_section,
                       color=BTN_START, hover_color=BTN_START_HVR)

    # State
    selected_maze = preset_names[0]
    selected_agent = "A*"
    maze_buttons[0].selected = True
    agent_buttons[3].selected = True  # A* is index 3
    random_seed = random.randint(0, 9999)

    def generate_preview():
        if selected_maze == "Random Maze":
            return imperfect_maze_grid(MAZE_SIZE, seed=random_seed)
        else:
            return MAZE_PRESETS[selected_maze]["fn"](MAZE_SIZE)

    preview_grid = generate_preview()

    clock = pygame.time.Clock()
    running = True

    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Maze presets
                for i, btn in enumerate(maze_buttons):
                    if btn.clicked(mouse_pos):
                        selected_maze = preset_names[i]
                        for b in maze_buttons:
                            b.selected = False
                        btn.selected = True
                        random_btn.selected = False
                        preview_grid = generate_preview()

                # Random maze
                if random_btn.clicked(mouse_pos):
                    selected_maze = "Random Maze"
                    random_seed = random.randint(0, 9999)
                    for b in maze_buttons:
                        b.selected = False
                    random_btn.selected = True
                    preview_grid = generate_preview()

                # Agent buttons
                for i, btn in enumerate(agent_buttons):
                    if btn.clicked(mouse_pos):
                        selected_agent = agent_options[i]
                        for b in agent_buttons:
                            b.selected = False
                        btn.selected = True

                # Start
                if start_btn.clicked(mouse_pos):
                    pygame.quit()
                    return selected_maze, selected_agent, preview_grid, MAZE_SIZE

        # --- Draw ---
        screen.fill(BG_COLOR)

        # Left panel background
        pygame.draw.rect(screen, PANEL_BG, (0, 0, left_panel_w, win_h))
        pygame.draw.line(screen, (60, 60, 70), (left_panel_w, 0), (left_panel_w, win_h), 2)

        # Title
        title_surf = font_title.render("AGENT EXPLORER", True, ACCENT_BLUE)
        screen.blit(title_surf, (left_panel_w // 2 - title_surf.get_width() // 2, 15))

        # Subtitle
        sub_surf = font_small.render("Bachelor's Thesis Simulation", True, LIGHT_GRAY)
        screen.blit(sub_surf, (left_panel_w // 2 - sub_surf.get_width() // 2, 50))

        # Section: Select Maze
        sec1 = font_section.render("SELECT MAZE", True, ACCENT_ORANGE)
        screen.blit(sec1, (20, 75))

        for btn in maze_buttons:
            btn.draw(screen, mouse_pos)
        random_btn.draw(screen, mouse_pos)

        # Section: Select Agent
        sec2 = font_section.render("SELECT AGENT", True, ACCENT_ORANGE)
        screen.blit(sec2, (20, btn_y + 6))

        for btn in agent_buttons:
            btn.draw(screen, mouse_pos)

        # Start button
        start_btn.draw(screen, mouse_pos)

        # --- Right panel: Preview ---
        preview_x = left_panel_w + 20
        preview_y = 60

        prev_title = font_section.render("MAZE PREVIEW", True, ACCENT_GREEN)
        screen.blit(prev_title, (preview_x + preview_px // 2 - prev_title.get_width() // 2, 20))

        # Maze name and description
        if selected_maze == "Random Maze":
            name_txt = f"Random Maze (seed: {random_seed})"
            desc_txt = "Randomly generated imperfect maze."
        else:
            name_txt = selected_maze
            desc_txt = MAZE_PRESETS[selected_maze]["desc"]

        name_surf = font_btn.render(name_txt, True, WHITE)
        screen.blit(name_surf, (preview_x, preview_y - 20))

        desc_surf = font_desc.render(desc_txt, True, LIGHT_GRAY)
        screen.blit(desc_surf, (preview_x, preview_y))

        # Draw preview
        start = (0, 0)
        goal = (MAZE_SIZE - 1, MAZE_SIZE - 1)
        all_cells = set((r, c) for r in range(MAZE_SIZE) for c in range(MAZE_SIZE))
        preview_surface = draw_grid_surface(preview_grid, goal, start, PREVIEW_CELL, all_cells)
        screen.blit(preview_surface, (preview_x, preview_y + 25))
        pygame.draw.rect(screen, (100, 100, 110),
                         (preview_x, preview_y + 25, preview_px, preview_px), 2)

        # Legend
        legend_y = preview_y + 25 + preview_px + 15
        items = [
            (WHITE, "Floor (cost 1)"),
            (GRAY, "Wall"),
            (BROWN, "Mud (cost 3)"),
            (START_COLOR, "Start"),
            (GOAL_COLOR, "Goal"),
            (AGENT_COLOR, "Agent"),
        ]
        leg_title = font_desc.render("LEGEND:", True, LIGHT_GRAY)
        screen.blit(leg_title, (preview_x, legend_y))
        legend_y += 18
        for color, label in items:
            pygame.draw.rect(screen, color, (preview_x, legend_y, 14, 14))
            pygame.draw.rect(screen, (100, 100, 110), (preview_x, legend_y, 14, 14), 1)
            lbl = font_small.render(label, True, LIGHT_GRAY)
            screen.blit(lbl, (preview_x + 20, legend_y))
            legend_y += 20

        # Selected agent indicator
        agent_col = agent_colors.get(selected_agent, WHITE)
        agent_info = font_btn.render(f"Agent: {selected_agent}", True, agent_col)
        screen.blit(agent_info, (preview_x, legend_y + 10))

        pygame.display.flip()
        clock.tick(30)


# ============================================================
# SIMULATION SCREEN (side-by-side)
# ============================================================

def run_simulation(grid, agent_name, maze_size):
    pygame.init()

    size = maze_size
    cell_size = 30
    width = size * cell_size
    height = size * cell_size
    padding = 40
    total_width = width * 2 + padding

    banner_height = 50
    panel_title_height = 35
    total_height = height + banner_height + panel_title_height

    screen = pygame.display.set_mode((total_width, total_height))
    pygame.display.set_caption(f"Simulation - {agent_name} Agent")

    start = (0, 0)
    goal = (size - 1, size - 1)
    env = GridEnv(grid, start, goal, size)

    # Create agent
    policy_map = {
        "Random": RandomPolicy(goal_pos=goal),
        "DFS":    DFSPolicy(goal_pos=goal),
        "BFS":    BFSPolicy(goal_pos=goal),
        "A*":     AStarPolicy(goal_pos=goal),
    }
    agent = Agent(policy_map[agent_name])
    sim = SimpleSimulation(agent, env)

    revealed_cells = set()
    all_cells = set((r, c) for r in range(size) for c in range(size))

    clock = pygame.time.Clock()
    running = True
    steps = 0
    total_cost = 0
    finished = False
    font = pygame.font.SysFont("consolas", 24, bold=True)
    small_font = pygame.font.SysFont("consolas", 18, bold=True)
    tiny_font = pygame.font.SysFont("consolas", 14)

    observation = env.observe()
    for cell in env.get_visible_cells(tuple(env.agent_pos)):
        revealed_cells.add(cell)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Agent turn
        if not env.is_terminal() and not finished:
            current_node = observation[0]
            if not sim.visited_nodes or sim.visited_nodes[-1] != current_node:
                sim.visited_nodes.append(current_node)

            action = agent.act(observation)

            if action is None:
                finished = True
            else:
                observation, step_cost = env.step(action)
                for cell in env.get_visible_cells(tuple(env.agent_pos)):
                    revealed_cells.add(cell)
                steps += 1
                total_cost += step_cost

        if env.is_terminal():
            finished = True

        # --- Render ---
        screen.fill(BG_COLOR)

        # Left panel
        left_surf = draw_grid_surface(grid, goal, start, cell_size, revealed_cells, env.agent_pos)
        screen.blit(left_surf, (0, banner_height + panel_title_height))
        pygame.draw.rect(screen, (100, 100, 100),
                         (0, banner_height + panel_title_height, width, height), 3)

        # Right panel
        right_surf = draw_grid_surface(grid, goal, start, cell_size, all_cells, env.agent_pos)
        screen.blit(right_surf, (width + padding, banner_height + panel_title_height))
        pygame.draw.rect(screen, (100, 100, 100),
                         (width + padding, banner_height + panel_title_height, width, height), 3)

        # Separator
        pygame.draw.line(screen, (80, 80, 90),
                         (width + padding // 2, banner_height),
                         (width + padding // 2, total_height), 4)

        # Top banner
        pygame.draw.rect(screen, (40, 40, 50), (0, 0, total_width, banner_height))
        pygame.draw.line(screen, (100, 100, 100), (0, banner_height), (total_width, banner_height), 2)

        info_text = f"{agent_name} Agent | Steps: {steps} | Cost: {total_cost}"
        if env.is_terminal():
            info_text += "  [GOAL REACHED!]"
        elif finished:
            info_text += "  [STUCK]"

        info_surface = font.render(info_text, True, (255, 255, 150))
        screen.blit(info_surface, (total_width // 2 - info_surface.get_width() // 2, 12))

        # Panel titles
        pygame.draw.rect(screen, (50, 70, 90), (0, banner_height, width, panel_title_height))
        tl = small_font.render("AGENT VIEW (FOG OF WAR)", True, WHITE)
        screen.blit(tl, (width // 2 - tl.get_width() // 2, banner_height + 8))

        pygame.draw.rect(screen, (50, 90, 70), (width + padding, banner_height, width, panel_title_height))
        tr = small_font.render("OMNISCIENT VIEW (FULL MAZE)", True, WHITE)
        screen.blit(tr, (width + padding + width // 2 - tr.get_width() // 2, banner_height + 8))

        # ESC hint
        esc_hint = tiny_font.render("Press ESC to exit", True, LIGHT_GRAY)
        screen.blit(esc_hint, (total_width - esc_hint.get_width() - 10, total_height - 20))

        pygame.display.flip()

        if finished:
            clock.tick(5)
        else:
            clock.tick(20)

    pygame.quit()
    print(f"\nSimulation finished. Agent: {agent_name}, Steps: {steps}, Cost: {total_cost}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    selected_maze, selected_agent, grid, maze_size = run_menu()
    run_simulation(grid, selected_agent, maze_size)