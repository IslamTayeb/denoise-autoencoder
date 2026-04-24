# Technical Spec: Denoising Autoencoders for 1D Time-Series Signals

## 0. Project Flow

End-to-end pipeline (this is the scientific story):

1. **Ingest** Wikipedia daily pageviews for a small set of articles. Resample to weekly.
2. **Fit a sine model** (dominant sinusoid + offset + optional linear trend) to each weekly series via least squares. Treat the fitted sine as the **ground-truth clean signal** for that series.
3. **Compute residuals** `r = real - fitted` and characterize their distribution (std, kurtosis, sparsity, autocorrelation, run-length).
4. **Fit each candidate noise model** (gaussian / masking / impulse) to those residuals by choosing parameters that make synthetic residuals match the observed distribution. **Rank** the noise models by goodness-of-fit.
5. **Generate a synthetic training corpus**: sample sine-wave parameters from the fitted distribution to produce many clean windows; corrupt each window using the fitted noise models.
6. **Train denoising autoencoders** (MLP, CNN1D, RNN) on the synthetic corpus.
7. **Evaluate on synthetic test data** (MSE, SNR, SNR improvement — has true ground truth).
8. **Real-world test**: apply the trained denoiser to actual Wikipedia weekly series; evaluate the reconstruction against the fitted sine as proxy ground truth. Report explicit caveat: the fitted sine is an approximation, not true ground truth.

## 1. Stack

- Python 3.11+
- PyTorch (CPU + MPS/CUDA)
- NumPy, pandas, scipy, matplotlib
- `requests` for the Wikimedia REST API
- `scikit-learn` only for utilities (e.g. KS test alternatives are in scipy)
- Reproducibility: every entrypoint sets `torch.manual_seed`, `np.random.seed`, `random.seed` from a CLI `--seed` arg (default `0`).

## 2. Repo Layout

```
final_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── exp_1_fit.yaml
│   ├── exp_2_noise_select.yaml
│   ├── exp_3_arch.yaml
│   ├── exp_4_latent.yaml
│   ├── exp_5_noise_robustness.yaml
│   ├── exp_6_noise_levels.yaml
│   └── exp_7_real_eval.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── wiki.py             # Wikimedia Pageviews ingestion
│   │   ├── sine_fit.py         # least-squares sine-wave fitting
│   │   ├── residuals.py        # residual computation + characterization
│   │   ├── noise.py            # gaussian, masking, impulse + parameter fitting
│   │   ├── synthetic.py        # generate (clean, noisy) windows from fitted parameters
│   │   └── windowing.py        # sliding windows, per-window z-score, overlap-average
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py
│   │   ├── cnn1d.py
│   │   └── rnn.py
│   ├── train.py
│   ├── eval.py
│   ├── viz.py
│   └── pipeline.py             # shared orchestration: fit -> select noise -> build dataset
├── experiments/
│   ├── run_1_fit.py
│   ├── run_2_noise_select.py
│   ├── run_3_arch.py
│   ├── run_4_latent.py
│   ├── run_5_noise_robustness.py
│   ├── run_6_noise_levels.py
│   ├── run_7_real_eval.py
│   └── run_all.py
├── results/
│   ├── tables/
│   ├── figures/
│   ├── checkpoints/
│   └── fits/                   # serialized sine-fit + noise-fit parameters
├── data_cache/
└── tests/
```

## 3. Constants

```python
WINDOW_LEN = 128
TRAIN_N = 8000
VAL_N = 1000
TEST_N = 1000
DEFAULT_BATCH = 64
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_LATENT = 16
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

WIKI_DEFAULT_ARTICLES = ["Influenza", "Common_cold", "Fever", "Cough", "Christmas"]
WIKI_DEFAULT_START = "2018-01-01"
WIKI_DEFAULT_END = None        # None -> latest available
```

Article choice rationale: each has a strong annual cycle (flu/cold articles peak in winter; "Christmas" peaks each December and is included as a positive control with very clean periodicity).

## 4. Real Data (`src/data/wiki.py`)

```python
def load_wiki_weekly(articles: list[str] = WIKI_DEFAULT_ARTICLES,
                    start: str = WIKI_DEFAULT_START,
                    end: str | None = WIKI_DEFAULT_END,
                    cache_dir: Path = Path("data_cache")) -> pd.DataFrame:
    # Wikimedia REST API: per-article daily pageviews, all-access, user agent traffic only.
    # https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{article}/daily/{YYYYMMDD}/{YYYYMMDD}
    # Set a descriptive User-Agent header (required).
    # Cache raw JSON per (article, start, end) at data_cache/wiki_{article}_{start}_{end}.json.
    # Resample daily -> weekly (W-MON, sum). Return a DataFrame indexed by week-start with one float32 column per article.
```

Notes:
- We work in log-space: every downstream module operates on `np.log1p(pageviews)` to stabilize variance and bring the dominant cycle close to a sinusoid in the transformed domain. Helper:
```python
def to_log(df: pd.DataFrame) -> pd.DataFrame  # element-wise log1p
```

## 5. Sine-Wave Fitting (`src/data/sine_fit.py`)

For a single 1D series `y[t]` of length T (in log-pageview space), fit:

```
y_hat[t] = c + m*t + A*sin(2π*f*t + φ)
```

Procedure:
1. Detrend with a least-squares linear fit to get residuals `y_d`.
2. Estimate dominant frequency via real FFT of `y_d`; pick the bin with maximum magnitude excluding DC. Restrict search to `f ∈ [1/(2*T), 0.5]`.
3. Initialize `A0 = 2 * |Y[k_peak]| / T`, `φ0 = angle(Y[k_peak])`.
4. Refine `(c, m, A, f, φ)` jointly with `scipy.optimize.curve_fit` (or `least_squares`), bounded so `f` stays within ±20% of the FFT estimate.
5. Return the fitted parameters and the reconstructed `y_hat`.

```python
@dataclass
class SineFit:
    c: float           # offset
    m: float           # linear slope per sample
    A: float           # amplitude
    f: float           # cycles per sample (per week, since we're weekly)
    phi: float         # phase
    period_weeks: float
    r_squared: float

def fit_sine(y: np.ndarray) -> SineFit
def predict_sine(fit: SineFit, t: np.ndarray) -> np.ndarray
def fit_all(df: pd.DataFrame) -> dict[str, SineFit]   # one fit per column
```

Sanity: `period_weeks` should land near 52 for the flu/cold/Christmas articles. If it doesn't, log a warning — that article isn't a clean fit and should be flagged in the report.

## 6. Residual Characterization (`src/data/residuals.py`)

```python
@dataclass
class ResidualStats:
    mean: float
    std: float
    skew: float
    kurtosis: float          # excess kurtosis; gaussian -> 0, heavy-tailed -> >>0
    fraction_outliers: float # |r| > 3*std
    fraction_near_zero: float # |r| < 0.05*std (proxy for masking-like dropouts)
    autocorr_lag1: float
    max_run_below_thresh: int # longest run of |r| < 0.1*std (proxy for masking segments)

def compute_residuals(y: np.ndarray, fit: SineFit) -> np.ndarray
def characterize(residuals: np.ndarray) -> ResidualStats
```

These statistics are the targets for the noise-model fitting in section 7.

## 7. Noise Models + Parameter Fitting (`src/data/noise.py`)

Three noise models, each parameterized by a single intensity-like scalar plus model-specific shape parameters that get fit to data.

```python
@dataclass
class GaussianNoiseParams:
    sigma: float                 # std of additive noise

@dataclass
class MaskingNoiseParams:
    seg_rate: float              # mean number of zero-segments per window (Poisson rate)
    seg_len_mean: float          # mean length of each zero segment

@dataclass
class ImpulseNoiseParams:
    spike_rate: float            # fraction of indices that get spikes
    magnitude_scale: float       # spike magnitude in units of clean-signal std
```

Application:
```python
def apply_gaussian(clean: np.ndarray, p: GaussianNoiseParams, rng) -> np.ndarray
def apply_masking(clean: np.ndarray, p: MaskingNoiseParams, rng) -> np.ndarray  # zeros (in log-space, "zero" means subtracting the local mean)
def apply_impulse(clean: np.ndarray, p: ImpulseNoiseParams, rng) -> np.ndarray
```

Parameter fitting from observed residuals:
```python
def fit_gaussian(residuals: np.ndarray) -> GaussianNoiseParams
    # sigma = sample std

def fit_masking(residuals: np.ndarray, near_zero_thresh: float = 0.1) -> MaskingNoiseParams
    # Identify runs where |r| < near_zero_thresh * std(r). seg_rate = total_runs / n_windows_equiv,
    # seg_len_mean = mean length of those runs.

def fit_impulse(residuals: np.ndarray, outlier_thresh: float = 3.0) -> ImpulseNoiseParams
    # spike_rate = fraction of |r| > outlier_thresh * std(r)
    # magnitude_scale = mean |r| among spikes / std(clean signal estimate)
```

Goodness-of-fit scoring — given observed residuals and a candidate noise model, generate synthetic residuals (apply the model to a zero signal, or to the fitted sines) and compare distributions:
```python
def score_noise_fit(observed_residuals: np.ndarray,
                    synthetic_residuals: np.ndarray) -> dict:
    # Returns: {"wasserstein": ..., "ks_stat": ..., "ks_pvalue": ...,
    #          "kurtosis_diff": ..., "std_ratio": ..., "near_zero_diff": ...}
```

Selection — `select_best_noise(observed_residuals)` returns a ranking with parameters fitted per model and the score dict per model. **Lower Wasserstein distance is better; KS p-value is reported for transparency but not used as a hard gate.**

## 8. Synthetic Dataset Generation (`src/data/synthetic.py`)

The "clean" signal family is parameterized by the distribution of sine fits we observed on the real data (section 5). At dataset-build time, sample fit parameters and render windows.

```python
@dataclass
class SineFitDistribution:
    A_range: tuple[float, float]
    f_range: tuple[float, float]      # cycles per sample at the WINDOW_LEN scale (not weeks)
    phi_range: tuple[float, float]    # always (0, 2π)
    m_range: tuple[float, float]
    c_range: tuple[float, float]

def distribution_from_fits(fits: dict[str, SineFit], window_len: int = WINDOW_LEN) -> SineFitDistribution:
    # Take the fitted (A, m, c) ranges directly. For frequency, rescale: at the weekly cadence with annual cycle,
    # f_per_sample ≈ 1/52. Multiply that by (window_len / typical_real_T) so the rendered windows
    # show ~the same number of cycles per WINDOW_LEN as the real series shows per WINDOW_LEN weeks.
    # Allow ±50% jitter on the frequency range to give the model some variety.

def make_clean_dataset(n: int, dist: SineFitDistribution, window_len: int = WINDOW_LEN,
                      seed: int = 0) -> np.ndarray:
    # Returns float32 array of shape (n, window_len). Each row: c + m*t + A*sin(2π*f*t + φ),
    # with parameters drawn uniformly from `dist`. After sampling, z-score each row.

def make_noisy_dataset(clean: np.ndarray, noise_kind: str, params,
                      seed: int = 0) -> np.ndarray:
    # Dispatch on noise_kind ∈ {"gaussian", "masking", "impulse"} using the fitted params.
```

DataLoaders mirror what was in the previous spec:
```python
def build_loaders(dist: SineFitDistribution, noise_kind: str, params, seed: int,
                 batch: int = DEFAULT_BATCH) -> tuple[DataLoader, DataLoader, DataLoader]
    # Generates TRAIN_N / VAL_N / TEST_N independent (clean, noisy) pairs.
```

## 9. Models (`src/models/`)

All models take input shape `(B, 1, L)`, return reconstruction with the same shape. Common interface:

```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self, window_len: int, latent_dim: int): ...
    def encode(self, x: Tensor) -> Tensor
    def decode(self, z: Tensor) -> Tensor
    def forward(self, x: Tensor) -> Tensor
```

### `mlp.py` — `MLPAE`
- Encoder: `Flatten` → `Linear(L, 256)` → `ReLU` → `Linear(256, 64)` → `ReLU` → `Linear(64, latent_dim)`
- Decoder: `Linear(latent_dim, 64)` → `ReLU` → `Linear(64, 256)` → `ReLU` → `Linear(256, L)` → `Unflatten(1, (1, L))`

### `cnn1d.py` — `CNN1DAE`
- Encoder (`L=128`):
  - `Conv1d(1, 16, k=7, s=2, p=3)` → `ReLU`     # L/2
  - `Conv1d(16, 32, k=5, s=2, p=2)` → `ReLU`    # L/4
  - `Conv1d(32, 64, k=3, s=2, p=1)` → `ReLU`    # L/8
  - `Flatten` → `Linear(64 * L/8, latent_dim)`
- Decoder mirrors with `ConvTranspose1d`, final `Conv1d(16, 1, k=1)`. Adjust output_padding so final length is exactly `L`.

### `rnn.py` — `RNNAE`
- Encoder: permute to `(B, L, 1)`, `GRU(1, 64, num_layers=1, batch_first=True)`, take final hidden, `Linear(64, latent_dim)`.
- Decoder: `Linear(latent_dim, 64)`, repeat across time to `(B, L, 64)`, `GRU(64, 64, batch_first=True)`, `Linear(64, 1)`, permute to `(B, 1, L)`.

Factory:
```python
MODELS = {"mlp": MLPAE, "cnn1d": CNN1DAE, "rnn": RNNAE}
def build_model(name: str, latent_dim: int = DEFAULT_LATENT, window_len: int = WINDOW_LEN) -> nn.Module
```

## 10. Training (`src/train.py`)

```python
def train(model, train_loader, val_loader, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR,
         device=DEVICE, log_path: Path | None = None) -> dict:
    # Adam, MSE loss between model(noisy) and clean.
    # Early-stop patience=10 on val_loss. Save best checkpoint to log_path/"best.pt".
    # Returns {"history": {...}, "best_val": ..., "best_epoch": ..., "checkpoint_path": ...}.
```

## 11. Evaluation (`src/eval.py`)

```python
def mse(clean: Tensor, recon: Tensor) -> float
def snr_db(clean: Tensor, recon: Tensor) -> float
def snr_improvement_db(clean, noisy, recon) -> float

def evaluate(model, loader, device=DEVICE) -> dict:
    # {"mse": ..., "snr_input_db": ..., "snr_output_db": ..., "snr_improvement_db": ...}
```

## 12. Visualization (`src/viz.py`)

```python
def plot_triplet(clean, noisy, recon, out_path, title="") -> None
def plot_history(history, out_path) -> None
def plot_metric_vs_x(xs, ys, xlabel, ylabel, out_path, title="") -> None
def plot_sine_fit(t, y_real, y_fit, out_path, title="") -> None
def plot_residual_qq(observed, synthetic, out_path, title="") -> None
def plot_residual_hist(observed, synthetics: dict[str, np.ndarray], out_path, title="") -> None
```

Each experiment must save 4 random triplet plots per trained (model, noise) combo.

## 13. Pipeline Helper (`src/pipeline.py`)

Centralizes the "fit sines, fit noise, build distribution" sequence so multiple experiments share the same artifacts:

```python
def build_pipeline_artifacts(cfg: dict) -> dict:
    # 1. Load Wikipedia weekly data (log-space).
    # 2. Fit sine per article, save SineFit dict to results/fits/sine_fits.pkl.
    # 3. Compute residuals per article, concatenate, save to results/fits/residuals.npy.
    # 4. Fit each noise model to residuals; score; save to results/fits/noise_fits.json.
    # 5. Build SineFitDistribution.
    # 6. Return dict with: real_df, log_df, fits, residuals, residual_stats, noise_fits, distribution.
    # Cache the whole bundle to results/fits/pipeline_artifacts.pkl on first call; reload on subsequent calls.
```

## 14. Experiments

Each `experiments/run_*.py`:
- Loads its YAML config.
- Sets seeds.
- Runs experiment.
- Writes CSV(s) to `results/tables/<exp_name>.csv`.
- Writes figures to `results/figures/<exp_name>/`.
- Writes checkpoints to `results/checkpoints/<exp_name>/`.
- Prints a summary line.

### Exp 1 — `run_1_fit.py` — Sine fit + residual characterization
- Load Wikipedia weekly data.
- Fit sine per article; record `SineFit` and `ResidualStats`.
- Tables:
  - `sine_fits.csv`: `article, A, f, phi, m, c, period_weeks, r_squared`.
  - `residual_stats.csv`: `article, mean, std, skew, kurtosis, fraction_outliers, fraction_near_zero, autocorr_lag1, max_run_below_thresh`.
- Figures (per article): `<article>_sine_fit.png` (real vs fitted overlay) and `<article>_residual_hist.png`.

### Exp 2 — `run_2_noise_select.py` — Noise model selection
- Reuse Exp 1 artifacts (call `build_pipeline_artifacts` if not cached).
- For each noise model, fit parameters to the pooled residuals.
- Generate synthetic residuals at matched length; score against observed via `score_noise_fit`.
- Table `noise_selection.csv`: `noise_kind, params_json, wasserstein, ks_stat, ks_pvalue, kurtosis_diff, std_ratio, near_zero_diff, rank`.
- Figures: `residual_hist_overlay.png` (observed + each candidate distribution overlaid); QQ plot per candidate.
- Print: `BEST_NOISE=<kind>` so downstream experiments can read it.

### Exp 3 — `run_3_arch.py` — Architecture comparison (synthetic)
- Use the **best** noise model from Exp 2 (config field `noise_kind: auto` reads from Exp 2 output; otherwise honor the explicit value).
- Build synthetic loaders from `SineFitDistribution`.
- Train `mlp`, `cnn1d`, `rnn` with the same data and budget.
- Table `arch_comparison.csv`: `model, n_params, train_loss_final, val_loss_best, test_mse, snr_input_db, snr_output_db, snr_improvement_db, train_seconds`.
- Figures: training-curve overlay; one triplet plot per model on the same test windows.

### Exp 4 — `run_4_latent.py` — Latent dimension sweep
- Best architecture from Exp 3 (config field `model: auto` or explicit), best noise from Exp 2.
- Latents: `[4, 8, 16, 32, 64]`.
- Table `latent_sweep.csv`: `latent_dim, n_params, test_mse, snr_improvement_db`.
- Figures: MSE vs latent_dim, SNR-improvement vs latent_dim.

### Exp 5 — `run_5_noise_robustness.py` — Cross-noise robustness
- Best architecture, latent=16.
- Train one model per noise model in `{gaussian, masking, impulse}` using each model's fitted parameters from Exp 2.
- Cross-evaluate every trained model on every test set (3×3).
- Table `noise_robustness.csv`: `train_noise, eval_noise, test_mse, snr_improvement_db`.
- Figure: 3×3 heatmap of `snr_improvement_db`.

### Exp 6 — `run_6_noise_levels.py` — Generalization across noise levels
- Best architecture + best noise model, trained at the fitted (real-data) intensity.
- Define an `intensity_multiplier` sweep: `[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]`. Apply to the fitted noise parameters (e.g., `sigma_eval = multiplier * sigma_fit`; `spike_rate_eval = multiplier * spike_rate_fit`, clamp ≤ 1.0; `seg_rate_eval = multiplier * seg_rate_fit`).
- Train at multiplier=1.0; evaluate at all multipliers.
- Table `noise_levels.csv`: `intensity_multiplier, snr_input_db, snr_output_db, snr_improvement_db, test_mse`.
- Figure: SNR-improvement vs multiplier with vertical line at multiplier=1.0.

### Exp 7 — `run_7_real_eval.py` — Real-world test on Wikipedia data
- Reuse Exp 1 artifacts (sine fits per article, full real series).
- Reload the best architecture + best noise model checkpoint (from Exp 3 / Exp 5).
- For each article:
  - Slide windows over the real log-pageview series with `stride=1`.
  - Per-window z-score; run through the autoencoder; un-z-score.
  - Reconstitute the full series via overlap-average of window reconstructions.
  - Compare denoised series against the **fitted sine** as proxy ground truth, and against the **raw real series** as the noisy input.
- Table `real_eval.csv`: per-article rows with `mse_recon_vs_sine, snr_db_recon_vs_sine, snr_improvement_db, std_residual_after_denoise`.
- Figures (per article): single plot with three lines — raw real series, fitted sine, autoencoder reconstruction.
- Print explicit caveat in summary: the fitted sine is an approximation to the true clean signal. SNR improvement here measures "did the autoencoder pull the real series toward the fitted sine," not "did it recover ground truth."

## 15. CLI

Each runner takes `--config <path>` and `--seed <int>`.

```bash
python experiments/run_1_fit.py --config configs/exp_1_fit.yaml --seed 0
# ... etc.
```

`experiments/run_all.py` invokes Exp 1 → 2 → 3 → 4 → 5 → 6 → 7 sequentially; exits non-zero on any failure. Each later experiment can read upstream artifacts/results, so order matters.

## 16. Config Schema (YAML)

Common keys per experiment:
```yaml
seed: 0
window_len: 128
output_dir: results/exp_<name>
data:
  articles: ["Influenza", "Common_cold", "Fever", "Cough", "Christmas"]
  start: "2018-01-01"
  end: null
training:
  epochs: 50
  batch: 64
  lr: 0.001
  patience: 10
```

Per-experiment additions:
```yaml
# exp_3_arch
models: [mlp, cnn1d, rnn]
noise_kind: auto                # "auto" reads from results/tables/noise_selection.csv
model:
  latent_dim: 16

# exp_4_latent
model:
  name: auto                    # reads from arch_comparison.csv (lowest test_mse)
  latents: [4, 8, 16, 32, 64]
noise_kind: auto

# exp_5_noise_robustness
model:
  name: auto
  latent_dim: 16
noise_kinds: [gaussian, masking, impulse]

# exp_6_noise_levels
model:
  name: auto
  latent_dim: 16
noise_kind: auto
intensity_multipliers: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# exp_7_real_eval
model:
  name: auto
  latent_dim: auto              # reads from latent_sweep.csv (best snr_improvement_db)
noise_kind: auto
checkpoint: results/checkpoints/exp_5_noise_robustness/<noise_kind>/best.pt
```

## 17. Tests (`tests/`, pytest, smoke-level)

- `test_wiki.py`: fake the HTTP layer with a fixture; assert weekly resampling shape.
- `test_sine_fit.py`: fit synthetic sine + noise, assert recovered `(A, f, phi)` within tolerance.
- `test_residuals.py`: known synthetic noise produces expected `ResidualStats` ranges.
- `test_noise.py`: shapes preserved; gaussian std grows with sigma; masking creates zero-runs of expected length; impulse adds outliers above threshold.
- `test_models.py`: each model accepts `(B, 1, L)`, returns same shape, gradients flow.
- `test_eval.py`: `snr_db(clean, clean)` is large/finite; `snr_improvement` positive when recon is closer than noisy.

## 18. README

Single page covering:
- Setup (`pip install -r requirements.txt`).
- The seven-step pipeline (mirror section 0).
- How to run each experiment.
- Where outputs land.
- Caveat for Exp 7: fitted sine is proxy ground truth, not real ground truth.
- Citations: Wikimedia Pageviews API; CDC FluView for context on why these articles are seasonally relevant.

## 19. Acceptance Criteria

A clean run of `python experiments/run_all.py` (with internet access for Exp 1 / 7) produces:

- 7 CSV tables in `results/tables/`.
- All figures listed under each experiment in `results/figures/<exp_name>/`.
- Trained checkpoints in `results/checkpoints/<exp_name>/`.
- Cached Wikipedia JSON in `data_cache/`.
- `results/fits/` populated with `sine_fits.pkl`, `residuals.npy`, `noise_fits.json`, `pipeline_artifacts.pkl`.
- Pytest passes: `pytest tests/ -q`.
- No uncaught exceptions in any runner.
