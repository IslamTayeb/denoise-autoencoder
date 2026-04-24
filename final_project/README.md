# Denoising Autoencoders for 1D Time-Series Signals

Course project (CS 675). Authors: Nikhil Pesaladinne, Islam Tayeb.

## Pipeline

The scientific story is one straight line:

1. Pull weekly Wikipedia pageviews (`Influenza`, `Common_cold`, `Fever`, `Cough`, `Christmas`).
2. Fit a dominant sinusoid (with offset and linear trend) to each series in log-space — treat that fit as **proxy clean signal**.
3. Compute residuals `r = real - fitted` and characterize the distribution.
4. Fit each candidate noise model (gaussian, masking, impulse) to the residuals; pick the best by Wasserstein distance.
5. Synthesize a large training corpus by sampling sine parameters from the fitted distribution and corrupting with the fitted noise.
6. Train denoising autoencoders (MLP, 1D CNN, GRU) on the synthetic data.
7. Evaluate on synthetic test data (true ground truth) and on the real Wikipedia series (proxy ground truth = fitted sine).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Sequentially
python experiments/run_all.py

# Or one at a time
python experiments/run_1_fit.py             --config configs/exp_1_fit.yaml
python experiments/run_2_noise_select.py    --config configs/exp_2_noise_select.yaml
python experiments/run_3_arch.py            --config configs/exp_3_arch.yaml
python experiments/run_4_latent.py          --config configs/exp_4_latent.yaml
python experiments/run_5_noise_robustness.py --config configs/exp_5_noise_robustness.yaml
python experiments/run_6_noise_levels.py    --config configs/exp_6_noise_levels.yaml
python experiments/run_7_real_eval.py       --config configs/exp_7_real_eval.yaml
```

## Outputs

- `results/tables/` — CSV summaries per experiment.
- `results/figures/<exp_name>/` — PNG figures per experiment.
- `results/checkpoints/<exp_name>/` — best PyTorch checkpoints.
- `results/fits/` — sine-fit and noise-fit artifacts.
- `data_cache/` — raw Wikipedia JSON.

## Tests

```bash
pytest tests/ -q
```

## Caveat

Experiment 7 (real-world evaluation) compares the autoencoder's reconstruction to the **fitted sine wave**, which is itself an approximation of the true clean signal. Reported "SNR improvement" on real data measures whether denoising pulled the real series toward the fitted sine, not whether it recovered ground truth.

## Data sources

- Wikimedia Pageviews API: <https://wikimedia.org/api/rest_v1/>
- Topic motivation (seasonality of disease-related Wikipedia traffic): Generous et al. 2014.
