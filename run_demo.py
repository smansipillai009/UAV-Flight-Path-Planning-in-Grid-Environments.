"""
Demo / driver for the UAV A* path planner.

Builds a sample 30x30 airspace with:
    - two rectangular no-fly zones
    - a prevailing wind field (blowing roughly north-east, with a strong
      headwind corridor the "greedy" straight-line path would cross)
then runs A* and plots the result.

Run:
    python run_demo.py
Outputs:
    uav_path_result.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from uav_astar import Airspace, a_star

ROWS, COLS = 30, 30


def build_airspace():
    # Prevailing wind: blowing towards the north-east (angle ~ -45deg in
    # row/col space), moderate everywhere...
    wind_angle = np.full((ROWS, COLS), np.deg2rad(-45))
    wind_speed = np.full((ROWS, COLS), 1.0)

    # ...except a strong headwind corridor a naive straight-line path
    # would fly straight through.
    wind_speed[10:20, 12:16] = 6.0
    wind_angle[10:20, 12:16] = np.deg2rad(135)  # blowing SW -> opposes NE travel

    airspace = Airspace(ROWS, COLS, wind_speed=wind_speed, wind_angle=wind_angle)

    # Restricted airspace zones (e.g. controlled zones / terrain)
    airspace.add_obstacle_rect(5, 5, 8, 20)
    airspace.add_obstacle_rect(20, 8, 24, 22)

    return airspace


def plot_result(airspace, path, start, goal, cost, nodes_expanded):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Wind speed as background shading (darker = stronger wind)
    ax.imshow(airspace.wind_speed, cmap="Blues", origin="upper", alpha=0.5)

    # Obstacles
    for r in range(airspace.rows):
        for c in range(airspace.cols):
            if airspace.grid[r, c] == 1:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color="dimgray"))

    # Wind direction arrows (sparse, for readability)
    step = 3
    for r in range(0, airspace.rows, step):
        for c in range(0, airspace.cols, step):
            ang = airspace.wind_angle[r, c]
            ax.arrow(c, r, 0.6 * np.sin(ang), 0.6 * np.cos(ang),
                      head_width=0.25, color="steelblue", alpha=0.6)

    if path:
        rs = [p[0] for p in path]
        cs = [p[1] for p in path]
        ax.plot(cs, rs, color="crimson", linewidth=2.5, marker="o",
                 markersize=3, label=f"A* path (cost={cost:.2f})")

    ax.scatter(*start[::-1], color="green", s=120, marker="s", label="Start", zorder=5)
    ax.scatter(*goal[::-1], color="purple", s=120, marker="*", label="Goal", zorder=5)

    ax.set_title(f"UAV Flight Path Planning with A*\nNodes expanded: {nodes_expanded}")
    ax.set_xlim(-0.5, airspace.cols - 0.5)
    ax.set_ylim(airspace.rows - 0.5, -0.5)
    ax.set_xlabel("Grid column (East ->)")
    ax.set_ylabel("Grid row (South ->)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig("uav_path_result.png", dpi=150)
    print("Saved plot to uav_path_result.png")


def main():
    airspace = build_airspace()
    start = (2, 2)
    goal = (27, 27)

    path, cost, nodes_expanded = a_star(airspace, start, goal)

    if path is None:
        print("No feasible path found.")
        return

    print(f"Path found with {len(path)} waypoints.")
    print(f"Total flight cost (distance + wind penalty): {cost:.3f}")
    print(f"Nodes expanded by A*: {nodes_expanded}")

    plot_result(airspace, path, start, goal, cost, nodes_expanded)


if __name__ == "__main__":
    main()
