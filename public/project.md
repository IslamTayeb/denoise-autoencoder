# Project 2: Denoising Autoencoders for 1D Time-Series Signals

**Required: 2 people**

In this project, students will instead work with 1D time-series signals, such as audio, sensor measurements, or biomedical signals. The goal is to design neural networks that can remove noise from corrupted signals. The project centers on the concept of a **denoising autoencoder**. Given a clean signal $x$, we generate a corrupted observation $x_{\text{noisy}}$ by injecting noise. A neural network $f_\theta$ is then trained to reconstruct the original signal:

$$\hat{x} = f_\theta(x_{\text{noisy}}).$$

The model parameters $\theta$ are learned by minimizing a reconstruction loss such as

$$\mathcal{L}(\theta) = \mathbb{E}\left[\|x - f_\theta(x_{\text{noisy}})\|_2^2\right].$$

Beyond basic denoising, this project also explores how different neural architectures model temporal structure.

## Datasets

Teams may choose one of the following types of data:

- Synthetic signals generated from combinations of sine waves, chirps, or periodic patterns
- Speech signals (e.g., Google Speech Commands dataset)
- Sensor signals (e.g., accelerometer or wearable device data)
- Biomedical signals (e.g., ECG time-series data)

In real datasets, signals should be segmented into fixed-length windows suitable for neural network input.

## Noise Models

To create input-target training pairs, students will corrupt clean signals using artificial noise. Each team should implement at least two of the following noise models:

- **Gaussian noise**

$$x_{\text{noisy}} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2).$$

- **Random masking (dropout noise):** randomly remove short segments of the signal.
- **Impulse noise:** randomly insert spikes or outliers into the signal.
- **Sinusoidal interference:** add periodic noise to simulate background interference.

These noise processes simulate common distortions encountered in real-world signals.

## Model Architectures

Students will implement and compare at least two different denoising autoencoder architectures. Possible choices include:

- Fully connected autoencoder
- Recurrent autoencoder using LSTM or GRU layers to capture temporal dependencies
- 1D convolutional autoencoder using temporal convolutions to extract local patterns
- (Optional) Transformer-based encoder-decoder

Each architecture follows an encoder-decoder structure:

$$z = \text{Encoder}(x_{\text{noisy}}), \quad \hat{x} = \text{Decoder}(z),$$

where $z$ is a latent representation with a lower dimension than the input signal.

## Experiments and Analysis

Students should conduct systematic experiments to understand how architectural choices affect performance. Suggested experiments include:

- **Comparing different model architectures:** Implement at least two architectures (e.g., MLP, LSTM, or 1D CNN autoencoders) and compare their reconstruction performance and training behavior.
- **Studying the effect of the latent dimension:** Evaluate how different bottleneck sizes influence reconstruction quality and compression ability.
- **Evaluating robustness to different noise types:** Train or test the models under multiple noise models (e.g., Gaussian noise, masking noise, or impulse noise) and compare performance.
- **Testing generalization across noise levels:** Train the model with a fixed noise level and evaluate its denoising performance under different noise intensities.

## Evaluation Metrics

Models should be evaluated using quantitative metrics such as:

- **Mean Squared Error (MSE)** between clean and reconstructed signals
- **Signal-to-Noise Ratio (SNR)** improvement

The signal-to-noise ratio can be computed as

$$\text{SNR}(x, \hat{x}) = 10 \log_{10}\left(\frac{\|x\|_2^2}{\|x - \hat{x}\|_2^2}\right).$$

Students should also provide visualizations of reconstructed signals, showing examples of (i) the original clean signal, (ii) the corrupted noisy signal, and (iii) the reconstructed signal produced by the model.

## Deliverables

Each team should submit:

- A complete implementation of the models and training pipeline.
- A report describing the dataset, model architectures, and experimental results.
- Quantitative comparisons between the different approaches.
- Visualizations demonstrating denoising performance.
