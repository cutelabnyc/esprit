import pathlib
import threading
import sys

import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
except ImportError:
    print("tkinterdnd2 not installed. Run: uv add tkinterdnd2", file=sys.stderr)
    sys.exit(1)

from main import analyze_file, make_figure, save_csv


class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("cute — ESPRIT modal analyser")
        self.geometry("720x520")
        self.minsize(500, 400)
        self._build_ui()

    def _build_ui(self):
        # ── Settings bar ──────────────────────────────────────────────
        bar = ttk.Frame(self, padding=(10, 8))
        bar.pack(fill="x")

        ttk.Label(bar, text="Modes:").pack(side="left")
        self.n_modes = tk.IntVar(value=100)
        ttk.Spinbox(bar, from_=1, to=500, textvariable=self.n_modes,
                    width=6).pack(side="left", padx=(3, 16))

        ttk.Label(bar, text="Threshold (dBr):").pack(side="left")
        self.threshold = tk.DoubleVar(value=-40.0)
        ttk.Spinbox(bar, from_=-100, to=0, increment=5,
                    textvariable=self.threshold, width=7).pack(side="left", padx=(3, 0))

        # ── Drop zone ─────────────────────────────────────────────────
        self.drop_zone = tk.Frame(self, bg="#f2f2f2", relief="solid",
                                  borderwidth=1, cursor="arrow")
        self.drop_zone.pack(fill="both", expand=True, padx=12, pady=(2, 6))

        self.drop_label = tk.Label(
            self.drop_zone, text="Drop an audio file here",
            font=("Helvetica", 20), fg="#aaaaaa", bg="#f2f2f2",
        )
        self.drop_label.place(relx=0.5, rely=0.42, anchor="center")

        self.sub_label = tk.Label(
            self.drop_zone, text="",
            font=("Helvetica", 11), fg="#888888", bg="#f2f2f2",
        )
        self.sub_label.place(relx=0.5, rely=0.58, anchor="center")

        for widget in (self.drop_zone, self.drop_label, self.sub_label):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>",      self._on_drop)
            widget.dnd_bind("<<DragEnter>>", self._on_drag_enter)
            widget.dnd_bind("<<DragLeave>>", self._on_drag_leave)

        # ── Log area ──────────────────────────────────────────────────
        self.log = scrolledtext.ScrolledText(
            self, height=8, font=("Menlo", 10),
            state="disabled", wrap="none", bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white",
        )
        self.log.pack(fill="x", padx=12, pady=(0, 10))

    # ── Drag-and-drop events ──────────────────────────────────────────

    def _on_drag_enter(self, event):
        self._set_zone_color("#ddeeff", "#3399cc")

    def _on_drag_leave(self, event):
        self._set_zone_color("#f2f2f2", "#aaaaaa")

    def _set_zone_color(self, bg, fg):
        self.drop_zone.config(bg=bg)
        self.drop_label.config(bg=bg, fg=fg)
        self.sub_label.config(bg=bg)

    def _on_drop(self, event):
        self._on_drag_leave(event)
        paths = self.tk.splitlist(event.data)
        if not paths:
            return
        file_path = pathlib.Path(paths[0])
        if not file_path.exists():
            self._log(f"File not found: {file_path}")
            return
        self._start_analysis(file_path)

    # ── Analysis ──────────────────────────────────────────────────────

    def _start_analysis(self, file_path):
        self.drop_label.config(text="Analysing…", fg="#555555")
        self.sub_label.config(text=file_path.name)
        self._log(f"\n{'─' * 58}")
        self._log(f"▶  {file_path}")

        n_modes   = self.n_modes.get()
        threshold = self.threshold.get()

        def worker():
            try:
                results = analyze_file(str(file_path),
                                       n_components=n_modes,
                                       threshold=threshold)
                self.after(0, lambda: self._on_done(file_path, results))
            except Exception as e:
                self.after(0, lambda: self._on_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, file_path, results):
        sr = results['sample_rate']
        if results['n_channels'] > 1:
            self._log("   stereo — using channel 0")
        if results['n_leading_silence'] > 0:
            self._log(f"   skipped {results['n_leading_silence']} silent samples "
                      f"({results['n_leading_silence'] / sr:.3f}s)")
        if results['was_truncated']:
            self._log(f"   truncated to {results['n_samples']} samples")
        if results['n_dropped_pole'] > 0:
            self._log(f"   dropped {results['n_dropped_pole']} near-undamped poles")
        if results['n_dropped_threshold'] > 0:
            self._log(f"   dropped {results['n_dropped_threshold']} below threshold")

        n = len(results['freqs_hz'])
        self._log(f"   {n} modes estimated")

        self.drop_label.config(text="Drop an audio file here", fg="#aaaaaa")
        self.sub_label.config(text="")

        if n == 0:
            self._log("   nothing to plot — try lowering the threshold")
            return

        # Save outputs alongside the audio file
        stem    = file_path.stem
        out_dir = file_path.parent
        png_path = out_dir / f"{stem}-esprit.png"
        csv_path = out_dir / f"{stem}-esprit.csv"

        fig = make_figure(results)
        fig.savefig(png_path, dpi=150)
        save_csv(results, csv_path)
        self._log(f"   saved → {png_path.name}")
        self._log(f"   saved → {csv_path.name}")

        self._show_figure(fig, title=f"ESPRIT — {file_path.name}")

    def _on_error(self, msg):
        self._log(f"   error: {msg}")
        self.drop_label.config(text="Drop an audio file here", fg="#aaaaaa")
        self.sub_label.config(text="")

    # ── Figure window ─────────────────────────────────────────────────

    def _show_figure(self, fig, title=""):
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("1100x600")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, win, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)

        win.protocol("WM_DELETE_WINDOW", lambda: (win.destroy(),
                                                   __import__('matplotlib').pyplot.close(fig)))

    # ── Log ───────────────────────────────────────────────────────────

    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
