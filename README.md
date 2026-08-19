# Nash-DePIN — Physical Validation (TurtleBot4 / LoCoBot, no blockchain)

> **Paper:** "Fair Task Allocation in Physical Multi-Robot Systems via Nash Q-Learning: A Comparison with Value Decomposition Methods" (submitted) — physical validation experiments
> **Goal:** Show that a Nash Q-Learning policy trained in simulation (E2 Q-tables) transfers to physical robots, preserving the learned equilibrium.

## Naming: repository vs. paper

This repository was developed before the experiments were renumbered for the
manuscript. The directory names use the original numbering:

| Repository | Paper | Setup | Steps/episode | Results |
|---|---|---|---|---|
| E5 | **E3** | TurtleBot4 × 2 (homogeneous) | 200 | `e5_det_seed2/`, `e5_det_seed4/` |
| E6 | **E4** | LoCoBot + TurtleBot4 (heterogeneous) | 12 | `e6_results/seed2/`, `e6_results/seed4/` |

Likewise, the two configuration files are **not** named after their platform:

| File | Platform | Used for |
|---|---|---|
| `config/e5_config_tb4x2.yaml` | TB4 × 2 | Paper E3 |
| `config/e5_config.yaml` | LoCoBot + TB4 | Paper E4 (default of `run_e5.py`) |

Each file carries a header stating which is which.

## What each file does

| File | Purpose |
|---|---|
| `experiments/run_e5.py` | Orchestrator: joint state, Nash policy, conflicts resolved BEFORE moving (physical contact impossible), frozen/finetune phases, `--resume` for battery pauses |
| `experiments/compare_traj.py` | Logical fidelity: compares the `cells` field (occupied grid cell) step by step |
| `experiments/calibrate_action.py` | Isolated test of one motion primitive; calibration gate |
| `experiments/run_e5_preflight.py` | Pre-flight checks + auditable session record |
| `nash_depin/grid_actions.py` | Action constants (no ROS dependency) |
| `nash_depin/grid_motion.py` | TurtleBot4 closed-loop motion (rotate to heading + drive one measured cell). No Nav2 |
| `nash_depin/locobot_motion.py` | LoCoBot (Kobuki) driver, interface-compatible with the above |
| `nash_depin/nash_solver.py` | Bimatrix solver with egalitarian refinement (identical to E2) |
| `nash_depin/fairness_metrics.py` | Gini and Jain (identical to E2) |
| `scripts/sim_frozen.py` | Pure simulation of the frozen policy — generates the reference trajectory |
| `scripts/classify_equilibria.py` | Classifies each episode: monopoly / balanced / cycle |
| `scripts/analyze_e5.py` | Phase-level descriptive comparison (see the caveat below) |
| `scripts/make_traj_figure.py` | Sim-vs-physical trajectory overlay figure |
| `tests/test_e5.py` + `conftest.py` | 11 tests, no ROS required (rclpy is mocked) |

## Design decisions

1. **Closed loop, not open loop.** Fixed-time open-loop motion accumulates drift that would break the 0.15 m per-cell tolerance over tens of episodes. Motion uses `/odom` feedback — no map, no Nav2.
2. **Action translation.** E2 Q-tables use 5 absolute grid actions; the robot executes each as rotate-to-heading + advance one cell.
3. **Logical collisions.** The orchestrator detects cell conflicts (same target cell or cell swap) before moving; nobody moves and both receive −5. Semantics identical to simulation, with physical safety guaranteed by construction.
4. **Fixed task station at (0,0).** As in E2 (`num_tasks=1`); the state encoding does not include the task position, so it MUST be fixed for the Q-tables to transfer.
5. **Deterministic frozen rollouts.** In the frozen phase epsilon is forced to 0 and `act()` takes `argmax(sigma)`. This is what makes step-by-step comparison against `sim_frozen.py` an exact fidelity test.

## Reproducing the reference trajectories

The pure decision logic imports without ROS, so the reference trajectories can
be regenerated on any machine:

```bash
pip install -r requirements.txt
pytest tests/ -v                       # 11 tests
python3 scripts/sim_frozen.py --seed 2 --config config/e5_config_tb4x2.yaml
python3 experiments/compare_traj.py sim_traj_seed2.json e5_det_seed2/traj_seed2_ep1.json
```

Running the robots themselves additionally requires ROS2 Humble and `rclpy`.

## Results

**E3 (TB4 × 2, `e5_det_seed2/`, `e5_det_seed4/`)** — 40 episodes, 20 per seed.
All 40 reproduce the simulated trajectory exactly: **40/40 episodes at 200/200
identical steps (100%)**. Cell error stays within tolerance (p50 0.06–0.09 m,
max 0.18 m).

**E4 (LoCoBot + TB4, `e6_results/seed2/`, `e6_results/seed4/`)** — 40 episodes,
20 per seed. Seed 2: 20/20 episodes at 12/12 identical steps. Seed 4: **19/20**
episodes at 12/12; episode 12 diverges transiently at step 3 (8/12 = 66.7%) and
reconverges by step 8 to the same terminal equilibrium, with identical episode
aggregates (`tasks=[9,0]`, 0 collisions).

Two caveats are reported with these numbers:

- With a **12-step horizon** the probability of agreement by chance is high, so
  E4 validates correct cross-hardware *execution*, not strong policy fidelity.
- The Kobuki base **suspends `/odom` while stationary**, inflating the measured
  cell error during monopoly parking phases (max > 4 m in E4). Cross-hardware
  fidelity is therefore evaluated on the discrete cell-occupancy sequence, not
  on metric accuracy.

`e6_results/seed3/` contains a **single aborted episode** and is excluded from
all reported results; it is retained only so the run log is complete.

## Caveat on physical Gini

`scripts/analyze_e5.py` compares the physical per-episode Gini against the E2
simulation Gini. These are **not the same quantity**. The low Gini reported in
E2 is an ensemble property across seeds trained with epsilon > 0; with an
indivisible task (`num_tasks=1`), a single frozen rollout can only end in
monopoly or a cycle. The paper therefore reports logical **fidelity** as the
physical result, and uses `classify_equilibria.py` to summarize equilibrium
types. The preregistered `gini_median_max: 0.15` in the config files is kept
for transparency but is not the reported physical criterion.
