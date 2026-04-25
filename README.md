# Denoising Autoencoders for 1D Time-Series Signals

CS 675 course project by Nikhil Pesaladinne and Islam Tayeb.

## Summary

This project studies denoising autoencoders for 1D time-series signals.
We use weekly Wikipedia pageviews for five seasonal articles.
We fit a sine curve to each series and use it as a proxy clean signal.
We fit Gaussian, masking, and impulse noise models to the residuals.
We then build synthetic training pairs and train autoencoders.
The final report compares an MLP and a 1D CNN.
The repo also includes an RNN autoencoder implementation.

## Pipeline

1. Pull weekly Wikipedia pageviews for `Influenza`, `Common_cold`, `Fever`, `Cough`, and `Christmas`.
2. Fit a dominant sine curve with an offset and a linear trend.
3. Compute residuals `r = real - fitted`.
4. Fit Gaussian, masking, and impulse noise models to the residuals.
5. Build a synthetic training set from the fitted sine distribution and the fitted noise.
6. Train denoising autoencoders on the synthetic data.
7. Evaluate on the raw Wikipedia series with the fitted sine as a proxy target.
8. Evaluate on per-article sine recovery with known fitted sines plus sampled noise.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python experiments/run_all.py

python experiments/run_1_fit.py --config configs/exp_1_fit.yaml
python experiments/run_2_noise_select.py --config configs/exp_2_noise_select.yaml
python experiments/run_3_arch.py --config configs/exp_3_arch.yaml
python experiments/run_4_latent.py --config configs/exp_4_latent.yaml
python experiments/run_5_noise_robustness.py --config configs/exp_5_noise_robustness.yaml
python experiments/run_6_noise_levels.py --config configs/exp_6_noise_levels.yaml
python experiments/run_7_real_eval.py --config configs/exp_7_real_eval.yaml
python experiments/run_8_sine_recovery.py --config configs/exp_8_sine_recovery.yaml
```

## Outputs

- `results/tables/` contains CSV summaries.
- `results/figures/<exp_name>/` contains figures.
- `results/checkpoints/<exp_name>/` contains saved checkpoints.
- `results/fits/` contains sine-fit and noise-fit artifacts.
- `data_cache/` contains cached Wikipedia JSON.

## Tests

```bash
pytest tests/ -q
```

## Caveat

Experiment 7 works on the raw Wikipedia series.
It measures error against the fitted sine.
That fitted sine is only a proxy clean target.
It is not ground truth.

## Links

- Report source: `report/report.tex`
- Report PDF: `report/report.pdf`
- GitHub: <https://github.com/IslamTayeb/denoise-autoencoder>
