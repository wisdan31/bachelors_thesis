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
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    grid = np.ones((size, size), dtype=int)

    start_r, start_c = 0, 0
    grid[start_r, start_c] = 0

    stack = [(start_r, start_c)]

    while stack:
        r, c = stack[-1]

        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)

        moved = False
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 1:
                wall_r, wall_c = r + dr // 2, c + dc // 2
                grid[wall_r, wall_c] = 0
                grid[nr, nc] = 0

                stack.append((nr, nc))
                moved = True
                break

        if not moved:
            stack.pop()

    grid[size-1, size-1] = 0
    if size > 1:
        if grid[size-2, size-1] == 1 and grid[size-1, size-2] == 1:
             grid[size-2, size-1] = 0

    return grid

def imperfect_maze_grid(size, wall_knockdown_prob=0.08, mud_prob=0.15, seed=None):
    # 0 = floor (cost 1), 1 = wall, 2 = mud (cost 3)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    grid = maze_grid(size, seed)

    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if grid[r, c] == 1:
                if random.random() < wall_knockdown_prob:
                    grid[r, c] = 0

    for r in range(size):
        for c in range(size):
            if grid[r, c] == 0:
                if (r, c) == (0, 0) or (r, c) == (size-1, size-1):
                    continue
                if random.random() < mud_prob:
                    grid[r, c] = 2

    return grid

def omniscient_dijkstra(grid, start, goal):
    size = len(grid)
    pq = [(0, start, [start])]
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
                if cell_type != 1:
                    step_cost = 3 if cell_type == 2 else 1
                    heapq.heappush(pq, (cost + step_cost, (nr, nc), path + [(nr, nc)]))

    return float('inf'), 0


def preset_narrow_corridors(size, seed=119):
    return maze_grid(size, seed=seed)

def preset_open_arena(size, seed=12):
    random.seed(seed)
    np.random.seed(seed)
    grid = np.zeros((size, size), dtype=int)

    for _ in range(size * size // 8):
        r = random.randint(1, size - 2)
        c = random.randint(1, size - 2)
        grid[r, c] = 1
        if random.random() < 0.4:
            dr, dc = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            nr, nc = r + dr, c + dc
            if 1 <= nr < size - 1 and 1 <= nc < size - 1:
                grid[nr, nc] = 1

    grid[0, 0] = 0
    grid[size-1, size-1] = 0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            for pr, pc in [(0, 0), (size-1, size-1)]:
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < size and 0 <= nc < size:
                    grid[nr, nc] = 0
    return grid

def preset_heavy_mud(size, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    grid = maze_grid(size, seed=seed)

    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if grid[r, c] == 1 and random.random() < 0.12:
                grid[r, c] = 0

    for r in range(size):
        for c in range(size):
            if grid[r, c] == 0 and (r, c) != (0, 0) and (r, c) != (size-1, size-1):
                dist_from_diag = abs(r - c)
                if dist_from_diag < size // 3:
                    if random.random() < 0.55:
                        grid[r, c] = 2

    return grid

def preset_rooms_bottlenecks(size, seed=33):
    random.seed(seed)
    np.random.seed(seed)
    grid = np.ones((size, size), dtype=int)

    room_count = 3
    wall_positions = []
    room_size = (size - (room_count + 1)) // room_count

    rooms = []
    for ri in range(room_count):
        for ci in range(room_count):
            r_start = 1 + ri * (room_size + 1)
            c_start = 1 + ci * (room_size + 1)
            r_end = min(r_start + room_size, size - 1)
            c_end = min(c_start + room_size, size - 1)

            rooms.append((r_start, c_start, r_end, c_end))

            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    grid[r, c] = 0

    for ri in range(room_count):
        for ci in range(room_count):
            idx = ri * room_count + ci
            r_start, c_start, r_end, c_end = rooms[idx]

            if ci < room_count - 1:
                right_idx = ri * room_count + (ci + 1)
                rr_start, rc_start, rr_end, rc_end = rooms[right_idx]
                wall_col = c_end
                possible_rows = list(range(max(r_start, rr_start) + 1, min(r_end, rr_end) - 1))
                if possible_rows:
                    door_row = random.choice(possible_rows)
                    grid[door_row, wall_col] = 0
                    if len(possible_rows) > 2 and random.random() < 0.4:
                        possible_rows.remove(door_row)
                        grid[random.choice(possible_rows), wall_col] = 0

            if ri < room_count - 1:
                below_idx = (ri + 1) * room_count + ci
                br_start, bc_start, br_end, bc_end = rooms[below_idx]
                wall_row = r_end
                possible_cols = list(range(max(c_start, bc_start) + 1, min(c_end, bc_end) - 1))
                if possible_cols:
                    door_col = random.choice(possible_cols)
                    grid[wall_row, door_col] = 0
                    if len(possible_cols) > 2 and random.random() < 0.4:
                        possible_cols.remove(door_col)
                        grid[random.choice(possible_cols), wall_row] = 0

    grid[0, 0] = 0
    grid[size-1, size-1] = 0
    r_s, c_s, _, _ = rooms[0]
    grid[0, 1] = 0
    if r_s > 0:
        for r in range(0, r_s):
            grid[r, 1] = 0
    _, _, r_e, c_e = rooms[-1]
    if r_e < size - 1:
        grid[r_e, c_e - 1] = 0
        for r in range(r_e, size):
            grid[r, c_e - 1] = 0
    if c_e < size - 1:
        grid[size - 2, c_e] = 0
        for c in range(c_e, size):
            grid[size - 2, c] = 0
    grid[size-1, size-1] = 0

    for ri in range(room_count):
        for ci in range(room_count):
            idx = ri * room_count + ci
            r_start, c_start, r_end, c_end = rooms[idx]
            if random.random() < 0.4:
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        if grid[r, c] == 0 and random.random() < 0.3:
                            if (r, c) != (0, 0) and (r, c) != (size-1, size-1):
                                grid[r, c] = 2

    return grid

def preset_dense_multipath(size, seed=55):
    random.seed(seed)
    np.random.seed(seed)
    grid = maze_grid(size, seed=seed)

    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if grid[r, c] == 1 and random.random() < 0.25:
                grid[r, c] = 0

    for r in range(size):
        for c in range(size):
            if grid[r, c] == 0 and (r, c) != (0, 0) and (r, c) != (size-1, size-1):
                if random.random() < 0.08:
                    grid[r, c] = 2

    return grid


MAZE_PRESETS = {
    "Narrow Corridors": {
        "fn": preset_narrow_corridors,
        "desc": "Perfect maze, single winding path.",
    },
    "Open Arena": {
        "fn": preset_open_arena,
        "desc": "Open space with scattered pillars.",
    },
    "Heavy Mud Swamp": {
        "fn": preset_heavy_mud,
        "desc": "Diagonal mud band, clean detours available.",
    },
    "Rooms & Bottlenecks": {
        "fn": preset_rooms_bottlenecks,
        "desc": "Rooms connected by narrow doorways.",
    },
    "Dense Multi-Path": {
        "fn": preset_dense_multipath,
        "desc": "Dense corridor network, many routes.",
    },
}