from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import tkinter as tk

from prepare_krea2_style_dataset import KREA2_BUCKETS


PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)]")
BUCKET_LABELS = tuple(f"{width}x{height}" for width, height in KREA2_BUCKETS)


@dataclass(frozen=True)
class RunConfig:
    input_dir: Path
    output_dir: Path
    models_dir: Path
    quality: int
    buckets: tuple[str, ...]
    smart_crop: bool
    caption_prompt: str | None = None
    caption_max_tokens: int = 160


def validate_config(config: RunConfig) -> str | None:
    if not config.input_dir.is_dir():
        return f"Input folder does not exist:\n{config.input_dir}"
    if config.input_dir.resolve() == config.output_dir.resolve():
        return "Input and output folders must be different."
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        return f"Output folder must be empty:\n{config.output_dir}"
    if not 1 <= config.quality <= 100:
        return "JPEG quality must be between 1 and 100."
    if not config.buckets:
        return "Select at least one output bucket."
    if config.caption_prompt is not None and not config.caption_prompt.strip():
        return "Caption prompt cannot be empty."
    if config.caption_prompt is not None and not 1 <= config.caption_max_tokens <= 2048:
        return "Caption max tokens must be between 1 and 2048."
    return None


def build_cli_args(config: RunConfig) -> list[str]:
    script_path = Path(__file__).with_name("prepare_krea2_style_dataset.py")
    args = [
        sys.executable,
        "-u",
        str(script_path),
        str(config.input_dir),
        str(config.output_dir),
        "--quality",
        str(config.quality),
        "--models-dir",
        str(config.models_dir),
    ]
    for bucket in config.buckets:
        args.extend(("--bucket", bucket))
    if not config.smart_crop:
        args.append("--no-smart-crop")
    if config.caption_prompt is not None:
        args.extend(("--caption-prompt", config.caption_prompt))
        args.extend(("--caption-max-tokens", str(config.caption_max_tokens)))
    return args


class KreaDatasetGUI:
    BG = "#f3f5f7"
    SURFACE = "#ffffff"
    TEXT = "#172033"
    MUTED = "#687386"
    BORDER = "#d8dee8"
    BRAND = "#1769e0"
    BRAND_ACTIVE = "#0f56bc"
    SUCCESS = "#16845b"
    ERROR = "#c43d4b"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Krea 2 · Style Dataset Prep")
        self.root.geometry("860x840")
        self.root.minsize(720, 700)
        self.root.configure(background=self.BG)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(4, weight=1)

        cwd = Path.cwd()
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(cwd / "krea2_ready"))
        self.models_var = tk.StringVar(value=str(cwd / "models"))
        self.quality_var = tk.StringVar(value="95")
        self.smart_crop_var = tk.BooleanVar(value=True)
        self.caption_var = tk.BooleanVar(value=False)
        self.caption_max_tokens_var = tk.StringVar(value="160")
        self.bucket_vars = {label: tk.BooleanVar(value=True) for label in BUCKET_LABELS}
        self.status_var = tk.StringVar(value="Ready · choose source and output folders")
        self.count_var = tk.StringVar(value="0 / 0")

        self._events: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._process: subprocess.Popen[str] | None = None
        self._running = False
        self._cancel_requested = False

        self._configure_styles()
        self._build_header()
        self._build_settings()
        self._build_captioning()
        self._build_actions()
        self._build_log()
        self._sync_caption_controls()

        self.root.after(80, self._drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=self.BG, foreground=self.TEXT)
        style.configure("Surface.TFrame", background=self.SURFACE)
        style.configure("Surface.TLabel", background=self.SURFACE, foreground=self.TEXT)
        style.configure("Muted.TLabel", background=self.SURFACE, foreground=self.MUTED)
        style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI Semibold", 19))
        style.configure("Eyebrow.TLabel", background=self.BG, foreground=self.BRAND, font=("Segoe UI Semibold", 9))
        style.configure(
            "Primary.TButton",
            background=self.BRAND,
            foreground="#ffffff",
            bordercolor=self.BRAND,
            focusthickness=2,
            focuscolor=self.BRAND,
            font=("Segoe UI Semibold", 10),
            padding=(18, 7),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.BRAND_ACTIVE), ("disabled", "#9eb9df")],
            bordercolor=[("active", self.BRAND_ACTIVE), ("disabled", "#9eb9df")],
            foreground=[("disabled", "#eef4fc")],
        )
        style.configure("Secondary.TButton", background=self.SURFACE, padding=(12, 6), bordercolor=self.BORDER)
        style.map("Secondary.TButton", background=[("active", "#edf1f6")])
        style.configure(
            "TEntry",
            fieldbackground="#fbfcfd",
            bordercolor=self.BORDER,
            lightcolor=self.BORDER,
            darkcolor=self.BORDER,
            padding=(8, 5),
        )
        style.map("TEntry", bordercolor=[("focus", self.BRAND)], lightcolor=[("focus", self.BRAND)])
        style.configure("TCheckbutton", background=self.SURFACE, foreground=self.TEXT)
        style.map("TCheckbutton", background=[("active", self.SURFACE)])
        style.configure(
            "Horizontal.TProgressbar",
            background=self.BRAND,
            troughcolor="#e4e9f0",
            bordercolor="#e4e9f0",
            lightcolor=self.BRAND,
            darkcolor=self.BRAND,
        )

    def _build_header(self) -> None:
        header = ttk.Frame(self.root, padding=(24, 12, 24, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="LOCAL DATASET UTILITY", style="Eyebrow.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Krea 2 Style Dataset Prep", style="Title.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(
            header,
            text="Crop-minimized JPEG conversion · optional local AI captions · exact Krea buckets",
            foreground=self.MUTED,
        ).grid(row=2, column=0, sticky="w", pady=(5, 0))

    def _card(self, row: int, padding: tuple[int, int, int, int] = (20, 12, 20, 12)) -> ttk.Frame:
        outer = tk.Frame(self.root, background=self.BORDER, padx=1, pady=1)
        outer.grid(row=row, column=0, sticky="nsew", padx=24, pady=(0, 12))
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        frame = ttk.Frame(outer, style="Surface.TFrame", padding=padding)
        frame.grid(row=0, column=0, sticky="nsew")
        return frame

    def _build_settings(self) -> None:
        card = self._card(1)
        card.columnconfigure(1, weight=1)
        card.columnconfigure(3, weight=1)

        ttk.Label(card, text="Folders", style="Surface.TLabel", font=("Segoe UI Semibold", 12)).grid(
            row=0, column=0, columnspan=4, sticky="w"
        )
        self._path_field(card, 1, "Source images", self.input_var, self._pick_input)
        self._path_field(card, 2, "Output folder", self.output_var, self._pick_output)
        self._path_field(card, 3, "Models folder", self.models_var, self._pick_models)

        separator = ttk.Separator(card)
        separator.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 10))

        ttk.Label(card, text="Output buckets", style="Surface.TLabel", font=("Segoe UI Semibold", 12)).grid(
            row=5, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(card, text="JPEG quality", style="Muted.TLabel").grid(row=5, column=2, sticky="w", padx=(22, 8))
        quality = ttk.Spinbox(card, from_=1, to=100, textvariable=self.quality_var, width=7)
        quality.grid(row=5, column=3, sticky="w")

        bucket_grid = ttk.Frame(card, style="Surface.TFrame")
        bucket_grid.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        for index, label in enumerate(BUCKET_LABELS):
            ttk.Checkbutton(bucket_grid, text=label, variable=self.bucket_vars[label]).grid(
                row=index // 3, column=index % 3, sticky="w", padx=(0, 16), pady=1
            )

        smart = ttk.Frame(card, style="Surface.TFrame")
        smart.grid(row=6, column=2, columnspan=2, sticky="nw", padx=(22, 0), pady=(8, 0))
        ttk.Checkbutton(smart, text="Smart crop (person + pose)", variable=self.smart_crop_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            smart,
            text="Falls back to center crop when models are unavailable.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

    def _path_field(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: Callable[[], None],
    ) -> None:
        ttk.Label(parent, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w", pady=(9, 0))
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=(14, 8), pady=(6, 0)
        )
        ttk.Button(parent, text="Browse", command=command, style="Secondary.TButton").grid(
            row=row, column=3, sticky="e", pady=(6, 0)
        )

    def _build_actions(self) -> None:
        card = self._card(3, (20, 10, 20, 10))
        card.columnconfigure(0, weight=1)

        status = ttk.Frame(card, style="Surface.TFrame")
        status.grid(row=0, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status_var, style="Surface.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(status, textvariable=self.count_var, style="Muted.TLabel", font=("Consolas", 10)).grid(
            row=0, column=1, sticky="e"
        )
        self.progress = ttk.Progressbar(card, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(6, 9))

        buttons = ttk.Frame(card, style="Surface.TFrame")
        buttons.grid(row=2, column=0, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        self.open_button = ttk.Button(
            buttons, text="Open output", command=self._open_output, style="Secondary.TButton"
        )
        self.open_button.grid(row=0, column=1, padx=(8, 0))
        self.cancel_button = ttk.Button(
            buttons, text="Cancel", command=self._cancel, style="Secondary.TButton", state="disabled"
        )
        self.cancel_button.grid(row=0, column=2, padx=(8, 0))
        self.run_button = ttk.Button(
            buttons, text="Prepare dataset", command=self._start, style="Primary.TButton"
        )
        self.run_button.grid(row=0, column=3, padx=(8, 0))

    def _build_log(self) -> None:
        card = self._card(4, (16, 12, 8, 8))
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="Activity", style="Surface.TLabel", font=("Segoe UI Semibold", 11)).grid(
            row=0, column=0, sticky="w", padx=4, pady=(0, 8)
        )
        self.log = tk.Text(
            card,
            height=10,
            state="disabled",
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
            background="#f8fafc",
            foreground="#344054",
            insertbackground=self.TEXT,
            font=("Cascadia Mono", 9),
            padx=12,
            pady=10,
        )
        self.log.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(card, command=self.log.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

    def _build_captioning(self) -> None:
        card = self._card(2, (20, 10, 20, 10))
        card.columnconfigure(1, weight=1)
        self.caption_check = ttk.Checkbutton(
            card,
            text="Generate AI captions",
            variable=self.caption_var,
            command=self._sync_caption_controls,
        )
        self.caption_check.grid(row=0, column=0, sticky="w")
        ttk.Label(
            card,
            text="Local Qwen3.5 2B · CUDA · UTF-8 .txt sidecars",
            style="Muted.TLabel",
        ).grid(row=0, column=1, sticky="w", padx=(12, 16))
        ttk.Label(card, text="Max tokens", style="Muted.TLabel").grid(row=0, column=2, sticky="e", padx=(0, 8))
        self.caption_tokens = ttk.Spinbox(
            card,
            from_=1,
            to=2048,
            textvariable=self.caption_max_tokens_var,
            width=7,
        )
        self.caption_tokens.grid(row=0, column=3, sticky="e")

        self.caption_prompt = tk.Text(
            card,
            height=3,
            wrap="word",
            borderwidth=1,
            relief="solid",
            highlightthickness=1,
            highlightbackground=self.BORDER,
            highlightcolor=self.BRAND,
            background="#fbfcfd",
            foreground=self.TEXT,
            font=("Segoe UI", 10),
            padx=9,
            pady=7,
        )
        self.caption_prompt.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.caption_prompt.insert(
            "1.0",
            "Write a concise training caption for this image. Describe only visible subjects, pose, "
            "clothing, setting, composition, lighting, colors, camera angle, and visual style. "
            "Do not identify people. Return caption only.",
        )
        self.caption_prompt.bind("<Button-3>", self._show_caption_menu)
        self.caption_prompt.bind("<Control-a>", self._select_all_caption_text)
        self.caption_menu = tk.Menu(self.root, tearoff=False)
        self.caption_menu.add_command(label="Cut", command=lambda: self.caption_prompt.event_generate("<<Cut>>"))
        self.caption_menu.add_command(label="Copy", command=lambda: self.caption_prompt.event_generate("<<Copy>>"))
        self.caption_menu.add_command(label="Paste", command=lambda: self.caption_prompt.event_generate("<<Paste>>"))
        self.caption_menu.add_separator()
        self.caption_menu.add_command(label="Select all", command=self._select_all_caption_text)

    def _sync_caption_controls(self) -> None:
        state = "disabled" if self._running else "normal"
        if state == "normal":
            self.caption_prompt.configure(state=state, background="#fbfcfd", foreground=self.TEXT)
        else:
            self.caption_prompt.configure(state=state, background="#f0f3f7", foreground=self.MUTED)
        self.caption_tokens.configure(state=state)

    def _show_caption_menu(self, event: tk.Event) -> str:
        self.caption_prompt.focus_set()
        edit_state = "disabled" if self._running else "normal"
        self.caption_menu.entryconfigure("Cut", state=edit_state)
        self.caption_menu.entryconfigure("Paste", state=edit_state)
        try:
            self.caption_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.caption_menu.grab_release()
        return "break"

    def _select_all_caption_text(self, _event: tk.Event | None = None) -> str:
        self.caption_prompt.tag_add("sel", "1.0", "end-1c")
        self.caption_prompt.mark_set("insert", "1.0")
        self.caption_prompt.see("insert")
        return "break"

    def _pick_input(self) -> None:
        selected = filedialog.askdirectory(title="Choose source image folder", initialdir=self.input_var.get() or None)
        if selected:
            self.input_var.set(selected)
            if self.output_var.get() == str(Path.cwd() / "krea2_ready"):
                self.output_var.set(str(Path(selected).with_name(f"{Path(selected).name}_krea2")))

    def _pick_output(self) -> None:
        selected = filedialog.askdirectory(title="Choose empty output folder", initialdir=self.output_var.get() or None)
        if selected:
            self.output_var.set(selected)

    def _pick_models(self) -> None:
        selected = filedialog.askdirectory(title="Choose models folder", initialdir=self.models_var.get() or None)
        if selected:
            self.models_var.set(selected)

    def _read_config(self) -> RunConfig | None:
        try:
            quality = int(self.quality_var.get())
        except ValueError:
            messagebox.showerror("Invalid settings", "JPEG quality must be a whole number from 1 to 100.")
            return None
        caption_prompt = None
        caption_max_tokens = 160
        if self.caption_var.get():
            caption_prompt = self.caption_prompt.get("1.0", "end").strip()
            try:
                caption_max_tokens = int(self.caption_max_tokens_var.get())
            except ValueError:
                messagebox.showerror("Invalid settings", "Caption max tokens must be a whole number.")
                return None
        return RunConfig(
            input_dir=Path(self.input_var.get().strip()),
            output_dir=Path(self.output_var.get().strip()),
            models_dir=Path(self.models_var.get().strip()),
            quality=quality,
            buckets=tuple(label for label in BUCKET_LABELS if self.bucket_vars[label].get()),
            smart_crop=self.smart_crop_var.get(),
            caption_prompt=caption_prompt,
            caption_max_tokens=caption_max_tokens,
        )

    def _start(self) -> None:
        if self._running:
            return
        config = self._read_config()
        if config is None:
            return
        error = validate_config(config)
        if error:
            messagebox.showerror("Cannot start", error)
            return

        self._set_running(True)
        self._cancel_requested = False
        self.progress["value"] = 0
        self.count_var.set("0 / 0")
        self.status_var.set("Starting…")
        self._clear_log()
        self._append_log(f"Source: {config.input_dir}\nOutput: {config.output_dir}\n\n")
        threading.Thread(target=self._worker, args=(build_cli_args(config),), daemon=True).start()

    def _worker(self, args: list[str]) -> None:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self._process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            if self._cancel_requested:
                self._process.terminate()
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self._events.put(("line", line))
            code = self._process.wait()
            self._events.put(("done", code))
        except OSError as exc:
            self._events.put(("error", str(exc)))
        finally:
            self._process = None

    def _drain_events(self) -> None:
        try:
            while True:
                event, payload = self._events.get_nowait()
                if event == "line":
                    line = str(payload)
                    self._append_log(line)
                    if line.startswith("Caption model: loading"):
                        self.status_var.set("Loading local Qwen caption model…")
                    match = PROGRESS_RE.match(line)
                    if match:
                        current, total = (int(value) for value in match.groups())
                        self.progress["value"] = current / total * 100
                        self.count_var.set(f"{current} / {total}")
                        self.status_var.set("Generating captions…" if "CAPTION" in line else "Preparing images…")
                elif event == "done":
                    self._finish(int(payload))
                elif event == "error":
                    self._append_log(f"ERROR: {payload}\n")
                    self._finish(2)
        except queue.Empty:
            pass
        self.root.after(80, self._drain_events)

    def _finish(self, code: int) -> None:
        self._set_running(False)
        if self._cancel_requested:
            self.status_var.set("Cancelled")
        elif code == 0:
            self.progress["value"] = 100
            self.status_var.set("Complete · dataset and manifest are ready")
            messagebox.showinfo("Dataset ready", f"Finished successfully.\n\n{self.output_var.get()}")
        elif code == 1:
            self.progress["value"] = 100
            self.status_var.set("Complete with skipped images · review Activity log")
            messagebox.showwarning("Dataset prepared with skips", "Some images were skipped. Check the Activity log.")
        else:
            self.status_var.set(f"Finished with errors · exit code {code}")
            messagebox.showerror("Preparation failed", "Check the Activity log for details.")

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.run_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")
        self.caption_check.configure(state="disabled" if running else "normal")
        self._sync_caption_controls()

    def _cancel(self) -> None:
        self._cancel_requested = True
        process = self._process
        if process is not None and process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                process.terminate()
            self.status_var.set("Cancelling…")

    def _open_output(self) -> None:
        output = Path(self.output_var.get().strip())
        if not output.is_dir():
            messagebox.showwarning("Output unavailable", "Output folder does not exist yet.")
            return
        if os.name == "nt":
            os.startfile(output)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(output)])

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self._running and not messagebox.askyesno("Task is running", "Cancel preparation and close the window?"):
            return
        self._cancel()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    KreaDatasetGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
