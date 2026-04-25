from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("experiments/run_1_fit.py", "configs/exp_1_fit.yaml"),
    ("experiments/run_2_noise_select.py", "configs/exp_2_noise_select.yaml"),
    ("experiments/run_3_arch.py", "configs/exp_3_arch.yaml"),
    ("experiments/run_4_latent.py", "configs/exp_4_latent.yaml"),
    ("experiments/run_5_noise_robustness.py", "configs/exp_5_noise_robustness.yaml"),
    ("experiments/run_6_noise_levels.py", "configs/exp_6_noise_levels.yaml"),
    ("experiments/run_7_real_eval.py", "configs/exp_7_real_eval.yaml"),
    ("experiments/run_8_sine_recovery.py", "configs/exp_8_sine_recovery.yaml"),
]


def main() -> int:
    for script, cfg in EXPERIMENTS:
        cmd = [sys.executable, str(REPO_ROOT / script), "--config", str(REPO_ROOT / cfg)]
        print(f"\n=== running {script} ===")
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        if proc.returncode != 0:
            print(f"!! {script} exited with code {proc.returncode}")
            return proc.returncode
    print("\n=== all experiments completed successfully ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
