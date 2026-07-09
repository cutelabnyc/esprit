# esprit

An ESPRIT-based modal analyser for audio files. It estimates the frequency, amplitude,
and decay time (T60) of the dominant sinusoidal/damped-exponential components in a
recording — useful for analysing resonances, impacts, and other decaying sounds.

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

### CLI

```sh
uv run cute path/to/audio.wav
```

Options:

- `-n, --components` — number of sinusoidal components to estimate (default: 4)
- `--snapshots` — snapshot (window) size for covariance estimation
- `--max-samples` — maximum number of samples to use (default: 22050)
- `--threshold` — discard modes below this amplitude in dBr (e.g. `--threshold -40`)
- `--max-pole` — discard modes with `|pole| >= this value` (default: 0.9999)
- `--no-plot` — suppress the plot
- `--save-plot PATH` — save the plot to a file instead of displaying it

### GUI

```sh
uv run cute-gui
```

Drag and drop an audio file onto the window to analyse it. Results are plotted and
saved alongside the source file as `<name>-esprit.png` and `<name>-esprit.csv`.
