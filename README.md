# Unit-I Project: UAV Flight Path Planning with A* Search

## Problem
Plan a UAV's waypoint trajectory through a grid airspace containing static
no-fly zones and a spatially varying wind field, minimizing total flight
cost (distance + energy lost/gained to wind) rather than raw distance.

## Why A*
- BFS/DFS ignore cost -> can't account for wind.
- Uniform-cost search (Dijkstra) is optimal but has no sense of direction
  towards the goal, so it wastes time expanding nodes in every direction.
- Greedy best-first only chases the heuristic and can be lured into an
  expensive headwind region because it "looks" close to the goal.
- A* (f = g + h) combines both: guaranteed optimal path (since our
  heuristic is admissible) while still being efficient.

## Files
- `uav_astar.py` — the algorithm: `Airspace` grid/wind model, `a_star()`
  search, `octile_heuristic()`.
- `run_demo.py` — builds a sample 30x30 airspace, runs the planner, saves
  `uav_path_result.png`.

## Design choices
- **8-connected movement** (N/S/E/W + diagonals): a UAV isn't confined to
  4 headings, and octile distance is the correct admissible heuristic for
  this connectivity (it never overestimates true cost, which is required
  for A* to remain optimal).
- **Wind-aware cost**: `move_cost()` compares the UAV's heading to the
  local wind direction. Flying into the wind adds cost; flying with it
  gives a capped discount so wind can never make an edge cost negative
  (which would break A*'s correctness).
- **Obstacles as hard constraints**: no-fly zones are simply excluded from
  the frontier — the UAV can never legally enter them, unlike wind, which
  is a soft cost.

## How to extend for later units
- Unit II (knowledge representation): add airspace *rules* (e.g. altitude
  bands, time-windows a zone is restricted) as a small rule/CSP layer on
  top of the grid.
- Unit III/IV (ML): learn a cost or wind-prediction model from real/simulated
  flight data instead of hand-setting the wind field, then feed it into the
  same `move_cost()` function — the search logic doesn't need to change.
- Combined final project: chain these into one pipeline — ML predicts the
  wind/cost field, A* plans the path through it, exactly like the
  LunaProbe pipeline structure (perception/estimation -> search).

## Run it
```bash
pip install numpy matplotlib
python run_demo.py
```
