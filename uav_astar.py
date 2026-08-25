"""
UAV Flight Path Planning in a Grid Environment using A* Search
================================================================

Unit-I project (Uninformed & Informed Search Techniques).

A UAV must fly from a start waypoint to a goal waypoint through a grid
airspace that contains:
    - static no-fly zones (restricted airspace / terrain)
    - a wind field, where flying against strong wind costs more energy
      than flying with it

We solve this with A* search:
    f(n) = g(n) + h(n)
    g(n) = actual cumulative cost from start to n (distance + wind penalty)
    h(n) = octile distance heuristic (admissible for 8-connected grids)

Why A* and not BFS/UCS/Greedy:
    - BFS ignores cost entirely -> wrong when wind makes some cells
      more expensive than others.
    - Uniform-Cost Search (Dijkstra) is optimal but explores blindly in
      all directions since it has no heuristic -> slower.
    - Greedy Best-First only looks at h(n) -> can be lured through a
      headwind corridor because it "looks close" to the goal.
    - A* uses both g and h, so it is optimal (given an admissible,
      consistent heuristic) AND efficient.
"""

import heapq
import numpy as np

# --------------------------------------------------------------------------
# Cell types
# --------------------------------------------------------------------------
FREE = 0
OBSTACLE = 1  # restricted airspace / no-fly zone

# 8-connected movement: (row_offset, col_offset, base_step_cost)
MOVES = [
    (-1,  0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),      # N S W E
    (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142),  # diagonals
]


class Airspace:
    """
    Grid-based airspace model.

    grid       : 2D array, 0 = free, 1 = no-fly zone
    wind_field : 2D array of the SAME shape holding a wind vector angle
                 (radians, direction the wind is BLOWING towards) per cell
    wind_speed : 2D array, wind magnitude per cell (arbitrary units)
    """

    def __init__(self, rows, cols, wind_speed=None, wind_angle=None):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)
        self.wind_speed = wind_speed if wind_speed is not None else np.zeros((rows, cols))
        self.wind_angle = wind_angle if wind_angle is not None else np.zeros((rows, cols))

    def add_obstacle_rect(self, r0, c0, r1, c1):
        """Mark a rectangular no-fly zone (inclusive bounds)."""
        self.grid[r0:r1 + 1, c0:c1 + 1] = OBSTACLE

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, r, c):
        return self.in_bounds(r, c) and self.grid[r, c] == FREE

    def move_cost(self, r0, c0, r1, c1, base_cost):
        """
        Cost of flying from (r0,c0) to (r1,c1).

        base_cost   : geometric step cost (1.0 orthogonal, sqrt2 diagonal)
        wind effect : if the UAV's heading roughly opposes the local wind,
                      it pays extra energy; a tailwind gives a small discount.
                      This is what makes the search problem more than plain
                      shortest-path -- two geometrically equal paths can have
                      very different real costs.
        """
        heading = np.arctan2(r1 - r0, c1 - c0)
        wind_dir = self.wind_angle[r1, c1]
        # Alignment: +1 = pure tailwind, -1 = pure headwind
        alignment = np.cos(heading - wind_dir)
        wind_penalty = self.wind_speed[r1, c1] * (-alignment) * 0.5
        return base_cost + max(wind_penalty, -base_cost * 0.4)  # cap the discount


def octile_heuristic(a, b):
    """Admissible heuristic for 8-connected grids (never overestimates)."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr + dc) + (1.4142 - 2) * min(dr, dc)


def a_star(airspace: Airspace, start, goal):
    """
    Returns (path, cost, nodes_expanded) where path is a list of (r, c)
    waypoints from start to goal, or (None, inf, nodes_expanded) if no
    path exists.
    """
    if not airspace.is_free(*start) or not airspace.is_free(*goal):
        raise ValueError("Start or goal lies on an obstacle / outside bounds.")

    open_heap = [(0.0, start)]          # (f_score, node) -- priority queue
    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    nodes_expanded = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            return _reconstruct_path(came_from, current), g_score[current], nodes_expanded

        for dr, dc, base_cost in MOVES:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if not airspace.is_free(nr, nc) or neighbor in closed:
                continue

            step_cost = airspace.move_cost(current[0], current[1], nr, nc, base_cost)
            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + octile_heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

    return None, float("inf"), nodes_expanded  # no path found


def _reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
