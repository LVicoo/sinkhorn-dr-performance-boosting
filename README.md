# Distributionally Robust Performance-Boosting Controller

Minimal PyTorch implementation of a Sinkhorn distributionally robust version of the performance-boosting controller on the two-agent mountain scenario from the paper: Furieri, Galimberti, Ferrari-Trecate, *Learning to Boost the Performance of Stable Nonlinear Systems*, 2024.

This repository builds on the performance-boosting controller implementation developed in the DECODE Lab:

- Original repository: [DecodEPFL/perf-boost-base](https://github.com/DecodEPFL/perf-boost-base)
- Original purpose: PyTorch implementation of performance-boosting controllers for stable nonlinear systems.
- Original reference: [Furieri, Galimberti, Ferrari-Trecate, *Learning to Boost the Performance of Stable Nonlinear Systems*, 2024](https://arxiv.org/abs/2405.00871).

The code compares two training criteria for the same stability-preserving controller architecture:

- **SAA**: sample-average training on empirical disturbance samples;
- **Sinkhorn-DRO**: worst-case expected training loss over a Sinkhorn ambiguity set centered at the empirical disturbance distribution.

The controller architecture is kept fixed. Robustness is introduced only through the training objective.

---

## 1. Repository structure

```text
.
├── README.md
├── requirements.txt
├── launch_experiments.py          # Main user-facing launcher with presets
├── run_experiment.py              # Core experiment runner
├── dr_pb/
│   ├── config.py                  # Default experiment configuration
│   ├── systems.py                 # Two-agent mountain dynamics
│   ├── controller.py              # IMC performance-boosting controller
│   ├── ren.py                     # Contractive REN module M_theta
│   ├── disturbances.py            # Latent disturbance sampling and transport cost
│   ├── losses.py                  # Tracking, control, collision, obstacle loss
│   ├── dro.py                     # Sinkhorn dual objective and lambda search
│   ├── trainers.py                # SAA and DRO training loops
│   ├── eval.py                    # Evaluation metrics and saved arrays
│   ├── plotting.py                # Diagnostic and thesis-style plots
│   └── utils.py                   # Seeding, device, JSON, output folders
└── scripts/
    ├── smoke_test.py              # Fast integration test
    ├── make_diagnostics.py        # Regenerate plots from a saved run
    └── evaluate_saved_run.py      # Re-evaluate saved controllers at another horizon
```

---

## 2. Installation

Create and activate a Python environment, then install:

```bash
pip install -r requirements.txt
```

Required packages:

```text
torch >= 2.2
numpy >= 1.24
matplotlib >= 3.8
```

Run a quick test from the repository root:

```bash
python scripts/smoke_test.py
```

The smoke test builds the system, creates small disturbance datasets, evaluates the pre-stabilized controller, trains SAA/DRO for zero epochs, and checks the golden-search lambda path.

---

## 3. Main way to run experiments

Use:

```bash
python launch_experiments.py
```

Edit these variables at the top of `launch_experiments.py`:

```python
SELECTED_PRESET = "obstacles_position_only_golden"
RUN_ALL = False
```

Set `RUN_ALL = True` to run every preset in `PRESETS` sequentially.

The launcher calls `run_experiment.py`, then regenerates diagnostics for the produced result folder.

---

## 4. Direct command-line execution

The core runner is:

```bash
python run_experiment.py --mode both
```

Available modes:

```text
saa    train only the SAA controller
dro    train only the DRO controller
both   train SAA and DRO and compare them
grid   train several fixed-lambda DRO controllers
```

Example useful DRO command:

```bash
python run_experiment.py \
  --mode both \
  --epochs 100 \
  --horizon 100 \
  --num-train 30 \
  --num-valid 40 \
  --num-test 300 \
  --batch-size 5 \
  --epsilon 0.1 \
  --rho 1.5 \
  --lambda-mode golden \
  --auto-feasible-rho \
  --std-pos-train 0.3 \
  --std-pos-ref 0.4 \
  --std-pos-test 0.4 \
  --std-force-train 0.0 \
  --std-force-ref 0.0 \
  --std-force-test 0.0 \
  --obstacle-avoidance \
  --collision-avoidance \
  --long-horizon 300
```

Use `--no-plots` to skip automatic plotting during training.

---

## 5. System implemented

The plant is a two-agent spring-damper point-mass system inspired by the mountain example in the performance-boosting paper.

For each agent,

```text
state  = [p_x, p_y, v_x, v_y]
input  = [u_x, u_y]
```

For two agents:

```text
state dimension = 8
input dimension = 4
```

The default nominal initial and target states are:

```python
x0   = [ 2, -2, 0, 0, -2, -2, 0, 0]
xbar = [-2,  2, 0, 0,  2,  2, 0, 0]
```

The implemented discrete-time noiseless dynamics have the form:

```text
x_{t+1} = f(x_t, u_t)
```

with sampling time:

```text
h = 0.05
```

and default physical parameters:

```text
mass m              = 1
spring gain k       = 1
damping b           = 1
nonlinear drag b2   = 0.1
```

If `linear_plant=False`, the system adds a speed-dependent nonlinear drag term on the velocity coordinates. This is the default.

The full disturbed rollout uses:

```text
x_{t+1} = f(x_t, u_t) + w_t
```

where `w_t` is produced from a low-dimensional latent disturbance vector.

---

## 6. Controller architecture

The trainable controller is `PerfBoostController` in `dr_pb/controller.py`.

It implements an internal-model control structure. At time `t`, it reconstructs the disturbance by comparing the measured state with the embedded noiseless plant prediction:

```text
w_hat_t = x_t - f(x_{t-1}, u_{t-1})
```

Then it applies the trainable operator:

```text
u_t = M_theta(w_hat_t)
```

In this implementation, `M_theta` is a contractive recurrent equilibrium network from `dr_pb/ren.py`.

The baseline `ZeroController` returns zero input and represents the pre-stabilized system without performance boosting.

---

## 7. Disturbance model

The low-dimensional latent disturbance is:

```text
z = [delta_p_1x, delta_p_1y, ..., delta_p_Nx, delta_p_Ny, force_x, force_y]
```

For two agents:

```text
z dimension = 6
```

The implementation uses `expand_latent_to_trajectory` to map `z` into a process-noise trajectory `w_{0:T-1}`.

### Initial position disturbance

Initial perturbations affect only position coordinates and only at `t=0`:

```text
w_0[position coordinates] = base_position_offset + sampled_position_offset
```

The default base offset is:

```python
BASE_POSITION_OFFSET = (4.0, -4.0, -4.0, -4.0)
```

This injects the nominal mountain initial condition relative to `xbar`.

### Shared force disturbance

The force disturbance is sampled once per rollout and is shared by all agents. It acts on velocity coordinates at every time step:

```text
dv = h / mass * force
```

For the final position-only experiments, the force standard deviations are set to zero.

---

## 8. Train, reference, and test distributions

The code separates three Gaussian latent distributions:

```python
std_pos_train, std_force_train   # empirical samples used by SAA and DRO outer distribution
std_pos_ref,   std_force_ref     # reference distribution nu used inside Sinkhorn-DRO
std_pos_test,  std_force_test    # final evaluation distribution
```

This separation is important.

- SAA trains only on empirical samples from the training distribution.
- DRO trains using empirical centers from the training distribution and reference samples from `nu`.
- Evaluation is performed on the test distribution.

The transport cost can be normalized by the reference standard deviations:

```python
latent_cost_on_normalized = True
```

The normalized transport cost is:

```text
c(z, z') = 0.5 || normalize(z) - normalize(z') ||^2
```

---

## 9. Loss function

The finite-horizon rollout loss is implemented in `dr_pb/losses.py`.

For a rollout with horizon `T`:

```text
ell_theta(z) = L_tracking + L_control + L_collision + L_obstacle
```

### Tracking term

```text
L_tracking = mean_t (x_t - xbar)^T Q (x_t - xbar)
```

Default:

```text
Q = I_8
```

### Control term

```text
L_control = alpha_u * mean_t ||u_t||^2
```

Default:

```text
alpha_u = 0.1 / 400 = 0.00025
```

### Collision term

For two agents, let:

```text
d_t = ||p_t^(1) - p_t^(2)||
```

The collision penalty is active when:

```text
d_t < min_dist + 0.2
```

and uses an inverse-square penalty:

```text
L_collision = alpha_col * mean_t [ 1_{d_t < min_dist + 0.2} / (d_t^2 + 1e-3) ]
```

Default:

```text
alpha_col = 100
min_dist  = 1.0
```

### Obstacle term

Obstacle avoidance is optional and controlled by `obstacle_avoidance`.

The obstacle term is a sum of Gaussian densities evaluated at each agent position:

```text
L_obstacle = alpha_obst * mean_t sum_agents sum_obstacles eta(p_t; mu_r, Sigma_r)
```

Default obstacle centers:

```text
(-2.5, 0), (2.5, 0), (-1.5, 0), (1.5, 0)
```

Default covariance for each obstacle:

```text
Sigma = diag(0.2, 0.2)
```

Default:

```text
alpha_obst = 10
```

All component plots generated by the code use the weighted components, i.e. after multiplication by the corresponding alpha coefficients.

---

## 10. SAA training

SAA training minimizes the empirical rollout loss:

```text
J_SAA(theta) = (1/N) sum_i ell_theta(z_i)
```

Implementation: `train_saa` in `dr_pb/trainers.py`.

Training loop:

```text
1. sample a mini-batch of empirical disturbances
2. expand z to a disturbance trajectory w
3. roll out the closed loop
4. compute the average finite-horizon loss
5. update theta with Adam
```

Default training values in `ExperimentConfig`:

```text
epochs       = 1000
lr           = 2e-3
batch_size   = 5
grad_clip    = 10
return_best  = True
```

The launcher overrides these with lighter defaults, usually `epochs=100`.

---

## 11. Sinkhorn-DRO training

DRO training minimizes the Monte Carlo approximation of the Sinkhorn dual objective:

```text
J_DRO(theta) = inf_{lambda > 0} [
    lambda * rho
    + lambda * epsilon * mean_i log mean_k exp(
        ell_theta(z_k^nu) / (lambda * epsilon)
        - c(z_i, z_k^nu) / epsilon
    )
]
```

where:

```text
z_i       = empirical training disturbance sample
z_k^nu    = reference-distribution sample
c         = latent transport cost
rho       = Sinkhorn ambiguity radius
epsilon   = entropic regularization
lambda    = dual variable
```

Implementation: `SinkhornDualLoss` in `dr_pb/dro.py` and `train_dro` in `dr_pb/trainers.py`.

The robust weighting favors reference samples that are both:

```text
high-loss under the current controller
close to empirical samples in transport cost
```

---

## 12. Lambda handling

The DRO dual variable can be handled in two modes.

### Fixed lambda

```python
dro_lambda_mode = "fixed"
```

If no value is supplied, the code estimates:

```text
lambda0 = median(reference rollout loss) / median(transport cost)
```

and keeps it fixed.

### Golden-search lambda

```python
dro_lambda_mode = "golden"
```

At each epoch:

```text
1. freeze controller parameters theta
2. compute reference rollout losses and transport costs once
3. minimize the scalar dual objective over lambda by golden-section search
4. update theta with Adam using the selected lambda
```

Main settings:

```python
lambda_search_low_factor  = 0.05
lambda_search_high_factor = 20.0
lambda_search_min         = 1e-4
lambda_search_max         = 1e4
lambda_search_max_iter    = 20
lambda_search_tol         = 1e-3
```

Useful diagnostics are saved in `histories.json`, including:

```text
lambda
lambda_search_lambda
lambda_search_objective
rho_effective
rho_feasibility_min_mc
```

---

## 13. Rho feasibility correction

The code can estimate a Monte Carlo feasibility floor for the Sinkhorn radius:

```text
rho_min = -epsilon * mean_i log mean_k exp(-c_ik / epsilon)
```

If:

```python
dro_auto_feasible_rho = True
```

then the code uses:

```text
rho_effective = max(rho, rho_min + margin)
```

This avoids numerical pathologies when the requested radius is below the MC feasibility floor.

Interpretation:

```text
rho_effective > rho for many epochs  -> requested rho is too small
lambda near lower bound              -> rho may be too large or objective too sharp
lambda very large                    -> robust objective may be weak or poorly conditioned
interior lambda                      -> usually healthier calibration
```

---

## 14. Output of a run

Each experiment creates a timestamped result folder:

```text
results/YYYY_MM_DD_HH_MM_SS_experiment_name_mode/
```

Typical contents:

```text
config.json                 # full run configuration
README_run.txt              # short summary of the run
metrics.json                # scalar evaluation metrics
metrics_h300.json           # optional long-horizon scalar metrics
histories.json              # training histories and lambda diagnostics
checkpoints/
  controller_saa.pt
  controller_dro.pt
arrays/
  prestable_zero_eval.npz
  random_ren_eval.npz
  saa_eval.npz
  dro_eval.npz
plots/
  *.pdf
```

Do not delete `arrays/` if you want to regenerate trajectory plots later. The diagnostic script needs the saved rollout arrays.

---

## 15. Evaluation metrics

Implemented in `dr_pb/eval.py`.

Main scalar metrics:

```text
mean_cost
median_cost
q90_cost
q95_cost
max_cost
collisions
collision_rate
final_mean_target_distance
final_q90_target_distance
min_interagent_distance
q05_min_interagent_distance
mean_control_norm
max_control_norm
mean_obstacle_penalty_raw
q95_obstacle_penalty_raw
obstacle_hits_2sigma
```

For robust-control interpretation, the most important metrics are:

```text
q90_cost
q95_cost
max_cost
collision_rate
q05_min_interagent_distance
```

Mean and median cost are still useful, but they do not capture rare unsafe rollouts.

---

## 16. Diagnostic plots

Regenerate plots for a saved run with:

```bash
python scripts/make_diagnostics.py results/<RUN_FOLDER> --device cpu
```

Useful options:

```bash
--png                  also save PNG copies
--no-titles            remove titles for thesis insertion
--background-max 60    reduce gray background trajectories
--worst-fraction 0.05  change worst-trajectory fraction
```

Plot appearance is controlled at the top of:

```text
scripts/make_diagnostics.py
```

inside:

```python
PLOT_OPTIONS = PlotConfig(...)
```

Generated plots include:

```text
loss_saa.pdf
loss_dro.pdf
training_comparison.pdf
lambda_vs_epoch.pdf
lambda_search_objective.pdf
trajectories_<controller>.pdf
trajectories_worst5pct_<controller>.pdf
trajectories_overlay.pdf
cost_boxplot.pdf
cost_histogram.pdf
cost_ecdf.pdf
metrics_bar.pdf
target_distance.pdf
final_target_distance_boxplot.pdf
interagent_distance.pdf
min_interagent_boxplot.pdf
control_norm.pdf
loss_components_bar.pdf
loss_components_boxplot.pdf
loss_components_over_time_<controller>.pdf
latent_disturbance_samples.pdf
obstacle_penalty_boxplot.pdf
```

Note: the DRO training loss curve reports the Sinkhorn dual objective, while the SAA training loss reports the empirical rollout loss. They are diagnostics, not directly identical objectives. Controller comparison should use the common evaluation metrics.

---

## 17. Long-horizon evaluation

`run_experiment.py` can evaluate trained controllers at a longer horizon by passing:

```bash
--long-horizon 300
```

For an already saved run:

```bash
python scripts/evaluate_saved_run.py results/<RUN_FOLDER> --horizon 300 --device cpu
```

This creates a separate long-horizon evaluation folder with metrics, arrays, and plots.

---

## 18. Presets and thesis experiments

The default `launch_experiments.py` contains exploratory presets.

For thesis-style comparisons, the most relevant experiment families were:

### Case A: matched full-data

```text
num_train = 30
std_pos_train = 0.4
std_pos_ref   = 0.4
std_pos_test  = 0.4
rho           = 1.5
epsilon       = 0.1
```

Purpose: check whether DRO improves tail and safety metrics without train/test distribution shift.

### Case B: reference-informed shift

```text
num_train = 30
std_pos_train = 0.3
std_pos_ref   = 0.4
std_pos_test  = 0.4
rho           = 1.5
epsilon       = 0.1
```

Purpose: main positive setting. The empirical training distribution is narrower than the reference/test distribution, so the reference law gives DRO information about plausible deployment uncertainty.

### Case C: matched scarce-data

```text
num_train = 10
std_pos_train = 0.4
std_pos_ref   = 0.4
std_pos_test  = 0.4
rho           = 1.0
epsilon       = 0.1
```

Purpose: scarce-data sensitivity study. This case was more seed-sensitive than A/B.

For repeated runs, create seeded presets in `launch_experiments.py` and run with `RUN_ALL=True`.

---

## 19. Practical interpretation of hyperparameters

### `std_pos_train`, `std_pos_ref`, `std_pos_test`

These define the geometry of the experiment.

```text
train = what the empirical dataset contains
ref   = what DRO treats as plausible deployment uncertainty
test  = what the controller is evaluated on
```

The reference distribution should cover the disturbances that are considered plausible at deployment.

### `epsilon`

Controls the entropic smoothing/locality of the Sinkhorn transport kernel.

```text
small epsilon -> sharper, more Wasserstein-like, more local and potentially noisy
large epsilon -> smoother, more reference-distribution dominated
```

### `rho`

Controls the ambiguity-set radius and hence the degree of adversarial robustness.

```text
small rho     -> close to SAA / weak robustification
moderate rho  -> useful tail-risk reduction
large rho     -> over-conservative or unstable training
```

### `lambda`

Dual variable optimized by golden search or fixed manually. It is a diagnostic of the rho/epsilon calibration.

```text
interior lambda        -> healthy regime
lambda at lower bound  -> rho may be too large
lambda very large      -> rho may be too small or objective poorly conditioned
```

---

## 20. Known limitations

- Training is nonconvex because the controller is a neural network.
- The inner Sinkhorn expectation is approximated by Monte Carlo samples.
- `rho`, `epsilon`, and the reference distribution require calibration.
- The final experiments focus on one nonlinear two-agent scenario.
- The implementation is intended as a clear research prototype, not as a fully optimized library.

---

## 21. Minimal commands for a new user

Check installation:

```bash
python scripts/smoke_test.py
```

Run one useful debug experiment:

```bash
python run_experiment.py --mode both --epochs 5 --num-train 10 --num-test 30 --horizon 30 --lambda-mode golden --auto-feasible-rho --obstacle-avoidance --rho 1.0
```

Run the launcher preset:

```bash
python launch_experiments.py
```

Regenerate plots:

```bash
python scripts/make_diagnostics.py results/<RUN_FOLDER> --device cpu
```

Evaluate saved controllers at horizon 300:

```bash
python scripts/evaluate_saved_run.py results/<RUN_FOLDER> --horizon 300 --device cpu
```
