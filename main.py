import argparse
import csv
import sys

import numpy as np
import soundfile as sf


def analytic_signal(x):
    N = len(x)
    X = np.fft.fft(x)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2
    return np.fft.ifft(X * h)


def esprit(x, n_components, n_snapshots=None):
    N = len(x)
    if n_snapshots is None:
        n_snapshots = min(max(8 * n_components + 2, 64), N // 4)
    M = N - n_snapshots + 1
    Xmat = np.array([x[i : i + n_snapshots] for i in range(M)])
    R = Xmat.T.conj() @ Xmat / M
    _, eigenvectors = np.linalg.eigh(R)
    Es = eigenvectors[:, -n_components:]
    Phi = np.linalg.lstsq(Es[:-1], Es[1:], rcond=None)[0]
    return np.linalg.eigvals(Phi)


def estimate_amplitudes(x, poles):
    n = np.arange(len(x))
    V = poles[np.newaxis, :] ** n[:, np.newaxis]
    a, _, _, _ = np.linalg.lstsq(V, x, rcond=None)
    return a


def analyze_file(path, n_components=100, threshold=-40.0, max_pole=0.9999,
                 max_samples=22050, n_snapshots=None):
    """Load audio and run ESPRIT. Returns a results dict."""
    data, sample_rate = sf.read(path)
    n_channels = data.shape[1] if data.ndim > 1 else 1
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float64)

    first_nonzero = int(np.argmax(data != 0))
    data = data[first_nonzero:]

    was_truncated = len(data) > max_samples
    n_used = min(len(data), max_samples)
    data = data[:n_used]

    z = analytic_signal(data)
    poles = esprit(z, n_components, n_snapshots)

    nyquist = sample_rate / 2
    freqs_hz = np.abs(np.angle(poles)) / (2 * np.pi) * sample_rate
    damping = np.abs(poles)

    valid = (freqs_hz > 0) & (freqs_hz < nyquist)
    poles, freqs_hz, damping = poles[valid], freqs_hz[valid], damping[valid]

    amplitudes = estimate_amplitudes(z, poles)
    amp_mag = np.abs(amplitudes)
    amp_db = 20 * np.log10(amp_mag / amp_mag.max() + 1e-12)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_d = np.log10(np.maximum(damping, 1e-30))
        t60_ms = np.where(damping < 1.0, -3000.0 / (sample_rate * log_d), np.nan)

    order = np.argsort(freqs_hz)
    freqs_hz, amp_db, damping, t60_ms = (
        freqs_hz[order], amp_db[order], damping[order], t60_ms[order]
    )

    keep = damping < max_pole
    n_dropped_pole = int(np.sum(~keep))
    freqs_hz, amp_db, damping, t60_ms = freqs_hz[keep], amp_db[keep], damping[keep], t60_ms[keep]

    n_dropped_threshold = 0
    if threshold is not None:
        keep = amp_db >= threshold
        n_dropped_threshold = int(np.sum(~keep))
        freqs_hz, amp_db, damping, t60_ms = freqs_hz[keep], amp_db[keep], damping[keep], t60_ms[keep]

    return {
        'file': str(path),
        'freqs_hz': freqs_hz,
        'amp_db': amp_db,
        'damping': damping,
        't60_ms': t60_ms,
        'sample_rate': sample_rate,
        'nyquist': nyquist,
        'n_samples': n_used,
        'n_channels': n_channels,
        'n_leading_silence': first_nonzero,
        'was_truncated': was_truncated,
        'n_dropped_pole': n_dropped_pole,
        'n_dropped_threshold': n_dropped_threshold,
    }


def make_figure(results):
    """Create and return a matplotlib Figure from an analyze_file results dict."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    freqs_hz = results['freqs_hz']
    amp_db = results['amp_db']
    damping = results['damping']
    t60_ms = results['t60_ms']
    nyquist = results['nyquist']

    stable = damping < 1.0
    colors = np.where(stable, "steelblue", "tomato")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax = axes[0]
    for f, a, c in zip(freqs_hz, amp_db, colors):
        ax.plot([f, f], [amp_db.min() - 3, a], color=c, linewidth=0.9, alpha=0.8)
        ax.plot(f, a, "o", color=c, markersize=4)
    ax.set_ylabel("Amplitude (dBr)")
    ax.set_title(f"ESPRIT modal estimates — {results['file']}")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xscale("log")

    ax = axes[1]
    stable_mask = np.isfinite(t60_ms)
    for f, t, c in zip(freqs_hz[stable_mask], t60_ms[stable_mask], colors[stable_mask]):
        ax.plot([f, f], [0, t], color=c, linewidth=0.9, alpha=0.8)
        ax.plot(f, t, "o", color=c, markersize=4)
    ax.set_ylabel("T60 (ms)")
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xscale("log")

    axes[0].legend(handles=[
        Line2D([0], [0], color="steelblue", marker="o", label="stable (|pole| < 1)"),
        Line2D([0], [0], color="tomato",    marker="o", label="unstable (|pole| ≥ 1)"),
    ], fontsize=8)

    if len(freqs_hz) > 0:
        xlim = (max(20, freqs_hz.min() * 0.8), nyquist)
        for ax in axes:
            ax.set_xlim(xlim)

    fig.tight_layout()
    return fig


def save_csv(results, path):
    """Write analysis results to a CSV file."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'frequency_hz', 'amplitude_dbr', 't60_ms', 'pole_magnitude'])
        for i, (freq, amp, t60, damp) in enumerate(
            zip(results['freqs_hz'], results['amp_db'], results['t60_ms'], results['damping'])
        ):
            writer.writerow([
                i + 1,
                f'{freq:.4f}',
                f'{amp:.4f}',
                f'{t60:.4f}' if np.isfinite(t60) else '',
                f'{damp:.6f}',
            ])


def main():
    parser = argparse.ArgumentParser(description="ESPRIT frequency estimator for audio files")
    parser.add_argument("file", help="Path to the audio file")
    parser.add_argument(
        "-n", "--components", type=int, default=4,
        help="Number of sinusoidal components to estimate (default: 4)",
    )
    parser.add_argument(
        "--snapshots", type=int, default=None,
        help="Snapshot (window) size for covariance estimation",
    )
    parser.add_argument(
        "--max-samples", type=int, default=22050,
        help="Maximum number of samples to use (default: 22050)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Discard modes below this amplitude in dBr (e.g. --threshold -40)",
    )
    parser.add_argument(
        "--max-pole", type=float, default=0.9999,
        help="Discard modes with |pole| >= this value (default: 0.9999)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Suppress the plot")
    parser.add_argument(
        "--save-plot", type=str, default=None, metavar="PATH",
        help="Save the plot to this path instead of displaying it",
    )
    args = parser.parse_args()

    if args.threshold is not None and args.threshold > 0:
        print(f"Warning: --threshold {args.threshold} is positive; dBr values are always ≤ 0. "
              f"Did you mean --threshold -{args.threshold}?")

    try:
        results = analyze_file(
            args.file,
            n_components=args.components,
            threshold=args.threshold,
            max_pole=args.max_pole,
            max_samples=args.max_samples,
            n_snapshots=args.snapshots,
        )
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        sys.exit(1)

    sr = results['sample_rate']
    if results['n_channels'] > 1:
        print("Stereo/multi-channel file detected — using channel 0 only.")
    if results['n_leading_silence'] > 0:
        print(f"Skipping {results['n_leading_silence']} silent samples "
              f"({results['n_leading_silence'] / sr:.3f}s of leading silence).")
    if results['was_truncated']:
        print(f"Using first {results['n_samples']} samples after silence "
              f"(use --max-samples to change).")
    print(f"Loaded {results['n_samples']} samples at {sr:.0f} Hz\n")

    if results['n_dropped_pole'] > 0:
        print(f"Dropped {results['n_dropped_pole']} near-undamped poles with |pole| >= {args.max_pole}.")
    if results['n_dropped_threshold'] > 0:
        print(f"Dropped {results['n_dropped_threshold']} modes below {args.threshold} dBr threshold.")
    print()

    freqs_hz = results['freqs_hz']
    if len(freqs_hz) == 0:
        print("No modes remain after filtering.")
        return

    amp_db, damping, t60_ms = results['amp_db'], results['damping'], results['t60_ms']
    print(f"{'#':>4}  {'Freq (Hz)':>10}  {'Amp (dBr)':>9}  {'T60 (ms)':>9}  {'|pole|':>7}")
    print("─" * 50)
    for i, (f, a, t, d) in enumerate(zip(freqs_hz, amp_db, t60_ms, damping)):
        t_str = f"{t:.1f}" if np.isfinite(t) else "  —"
        print(f"  {i+1:>3}  {f:>10.2f}  {a:>9.1f}  {t_str:>9}  {d:>7.4f}")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig = make_figure(results)
        if args.save_plot:
            fig.savefig(args.save_plot, dpi=150)
            print(f"\nPlot saved to {args.save_plot}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
