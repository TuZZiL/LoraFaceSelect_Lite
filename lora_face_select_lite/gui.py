from __future__ import annotations

import io
import queue
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import tkinter as tk

from .cli import CropShortfall, main as cli_main
from .profiles import MODEL_PROFILES


class _Redirect:
    """Capture writes to a queue so a background thread can feed the log box."""

    def __init__(self, target: "queue.Queue[str]") -> None:
        self._target = target
        self._buffer = ""

    def write(self, text: str) -> int:
        if "\r" in text:
            self.flush()
            for progress in text.split("\r"):
                if progress:
                    self._target.put(progress + "\n")
            return len(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._target.put(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._target.put(self._buffer)
            self._buffer = ""


class LoraFaceSelectGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LoRA Face Select Lite")
        self.root.minsize(680, 700)
        self.root.columnconfigure(0, weight=1)

        self.dataset_var = tk.StringVar()
        self.references_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path.cwd() / "result"))
        self.models_dir_var = tk.StringVar(value="models")
        self.model_profile_var = tk.StringVar(value="stable")
        self.count_var = tk.StringVar(value="20")
        self.min_similarity_var = tk.StringVar(value="0.50")
        self.min_quality_var = tk.StringVar(value="0.15")
        self.max_abs_yaw_var = tk.StringVar(value="25.0")
        self.min_face_width_var = tk.StringVar(value="48")
        self.prepare_crops_var = tk.BooleanVar(value=True)
        self.appearance_rerank_var = tk.BooleanVar(value=True)
        self.body_attributes_var = tk.BooleanVar(value=True)
        self.parsing_previews_var = tk.BooleanVar(value=True)
        self.analyze_videos_var = tk.BooleanVar(value=True)
        self.video_sample_fps_var = tk.StringVar(value="0.5")
        self.video_max_samples_var = tk.StringVar(value="120")
        self.video_max_candidates_var = tk.StringVar(value="3")
        self.overwrite_var = tk.BooleanVar(value=False)

        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._running = False

        self._build_paths()
        self._build_params()
        self._build_video()
        self._build_actions()
        self._build_log()
        self._sync_video_controls()

        self.root.after(100, self._drain_log)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- layout helpers -------------------------------------------------

    def _section(self, parent: tk.Widget, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.columnconfigure(1, weight=1)
        return frame

    def _dir_field(self, parent: tk.Widget, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew")
        ttk.Button(parent, text="…", width=3, command=lambda: self._pick_dir(variable)).grid(row=row, column=2)

    def _file_field(self, parent: tk.Widget, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew")
        ttk.Button(parent, text="…", width=3, command=lambda: self._pick_files(variable)).grid(row=row, column=2)

    def _int_field(self, parent: tk.Widget, row: int, label: str, variable: tk.StringVar) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
        entry = ttk.Entry(parent, textvariable=variable, width=10)
        entry.grid(row=row, column=1, sticky="w")
        return entry

    def _float_field(self, parent: tk.Widget, row: int, label: str, variable: tk.StringVar) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
        entry = ttk.Entry(parent, textvariable=variable, width=10)
        entry.grid(row=row, column=1, sticky="w")
        return entry

    def _check_field(self, parent: tk.Widget, row: int, label: str, variable: tk.BooleanVar) -> None:
        ttk.Checkbutton(parent, text=label, variable=variable).grid(row=row, column=0, columnspan=2, sticky="w")

    def _combo_field(self, parent: tk.Widget, row: int, label: str, variable: tk.StringVar, values: list[str]) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=16)
        combo.grid(row=row, column=1, sticky="w")
        return combo

    # --- widgets --------------------------------------------------------

    def _build_paths(self) -> None:
        frame = self._section(self.root, "Paths")
        frame.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self._dir_field(frame, 0, "Dataset folder", self.dataset_var)
        self._file_field(frame, 1, "Reference images", self.references_var)
        self._dir_field(frame, 2, "Output folder", self.output_var)
        self._dir_field(frame, 3, "Models folder", self.models_dir_var)
        ttk.Label(frame, text="(references: select 1–5 images)").grid(row=4, column=1, sticky="w")

    def _build_params(self) -> None:
        frame = self._section(self.root, "Parameters")
        frame.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        profile_combo = self._combo_field(frame, 0, "Model profile", self.model_profile_var, list(MODEL_PROFILES))
        profile_combo.bind("<<ComboboxSelected>>", self._on_profile_change)
        self._int_field(frame, 1, "Count", self.count_var)
        self._float_field(frame, 2, "Min similarity", self.min_similarity_var)
        self._float_field(frame, 3, "Min quality", self.min_quality_var)
        self._float_field(frame, 4, "Max abs yaw", self.max_abs_yaw_var)
        self._int_field(frame, 5, "Min face width", self.min_face_width_var)
        self._check_field(frame, 6, "Prepare crops", self.prepare_crops_var)
        self._check_field(frame, 7, "Appearance rerank", self.appearance_rerank_var)
        self._check_field(frame, 8, "Body attributes", self.body_attributes_var)
        self._check_field(frame, 9, "Parsing previews", self.parsing_previews_var)

    def _build_actions(self) -> None:
        frame = ttk.Frame(self.root, padding=(8, 6))
        frame.grid(row=3, column=0, sticky="ew")
        style = ttk.Style(self.root)
        style.configure("RunSelect.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 6))
        self.action_buttons: list[ttk.Button] = []
        for col, (text, command) in enumerate((
            ("Download models", self._run_download),
            ("Doctor", self._run_doctor),
            ("Run select", self._run_select),
            ("Open review", self._open_review),
        )):
            frame.columnconfigure(col, weight=1)
            button = ttk.Button(frame, text=text, command=command, style="RunSelect.TButton" if text == "Run select" else "TButton")
            button.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 4, 0))
            self.action_buttons.append(button)
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 0))

    def _build_log(self) -> None:
        frame = self._section(self.root, "Log")
        frame.grid(row=4, column=0, sticky="nsew", padx=8, pady=6)
        self.root.rowconfigure(4, weight=1)
        self.log = tk.Text(frame, height=12, state="disabled", wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(frame, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log["yscrollcommand"] = scrollbar.set
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    def _build_video(self) -> None:
        frame = self._section(self.root, "Local videos")
        frame.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        ttk.Checkbutton(
            frame, text="Analyze videos in dataset", variable=self.analyze_videos_var,
            command=self._sync_video_controls,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self.video_controls = [
            self._float_field(frame, 1, "Sample frames / sec", self.video_sample_fps_var),
            self._int_field(frame, 2, "Max samples / video", self.video_max_samples_var),
            self._int_field(frame, 3, "Keep frames / video", self.video_max_candidates_var),
        ]
        self._check_field(frame, 4, "Reuse checkpoint / replace prior results", self.overwrite_var)
        ttk.Label(
            frame,
            text="Sequential CPU scan · bounded memory · checkpoint after each video",
            foreground="#666666",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _on_profile_change(self, _event: object = None) -> None:
        """Populate min_similarity with the selected profile's recommended threshold.

        The field stays a normal Entry, so the value remains editable afterwards.
        """
        profile = MODEL_PROFILES.get(self.model_profile_var.get())
        if profile is not None:
            self.min_similarity_var.set(f"{profile.recommended_min_similarity:.2f}")

    def _sync_video_controls(self) -> None:
        state = "normal" if self.analyze_videos_var.get() else "disabled"
        for control in self.video_controls:
            control.configure(state=state)

    # --- pickers --------------------------------------------------------

    def _pick_dir(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    def _pick_files(self, variable: tk.StringVar) -> None:
        paths = filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All", "*.*")]
        )
        if paths:
            variable.set(" ".join(str(Path(p)) for p in paths))

    # --- logging --------------------------------------------------------

    def _drain_log(self) -> None:
        try:
            while True:
                line = self._log_queue.get_nowait()
                self.log["state"] = "normal"
                self.log.insert("end", line)
                self.log.see("end")
                self.log["state"] = "disabled"
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log)

    def _log_line(self, text: str) -> None:
        self._log_queue.put(text + "\n")

    # --- command execution ---------------------------------------------

    def _run_command(
        self, argv: list[str], *, capture_stdin: bool = False, resolve_crop_shortfall: bool = False,
    ) -> None:
        if self._running:
            messagebox.showwarning("Busy", "A task is already running. Wait for it to finish.")
            return
        self._running = True
        for button in self.action_buttons:
            button.configure(state="disabled")
        self.progress.start()

        def worker() -> None:
            saved_stdout = sys.stdout
            saved_stderr = sys.stderr
            saved_stdin = sys.stdin
            sys.stdout = _Redirect(self._log_queue)
            sys.stderr = sys.stdout
            if capture_stdin:
                sys.stdin = io.StringIO("n\n")  # non-interactive shortfall reply
            try:
                decision = self._request_crop_decision if resolve_crop_shortfall else None
                code = cli_main(argv, crop_decision=decision)
            except SystemExit as exc:
                code = int(exc.code or 0)
            except Exception as exc:  # noqa: BLE001
                self._log_line(f"ERROR: {exc}")
                code = 2
            finally:
                sys.stdout = saved_stdout
                sys.stderr = saved_stderr
                sys.stdin = saved_stdin
            self._log_line(f"--- exited with code {code} ---")
            self.root.after(0, self._finish_command)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_command(self) -> None:
        self._running = False
        self.progress.stop()
        for button in self.action_buttons:
            button.configure(state="normal")

    def _request_crop_decision(self, context: CropShortfall) -> str:
        if not (context.output / "crop_skipped").is_dir():
            return "finish"
        result = {"action": "finish"}
        ready = threading.Event()
        self.root.after(0, lambda: self._show_crop_shortfall_dialog(context, result, ready))
        ready.wait()
        return result["action"]

    def _show_crop_shortfall_dialog(
        self, context: CropShortfall, result: dict[str, str], ready: threading.Event,
    ) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Потрібне рішення щодо crop")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.columnconfigure(0, weight=1)

        content = ttk.Frame(dialog, padding=18)
        content.grid(sticky="nsew")
        content.columnconfigure(0, weight=1)
        style = ttk.Style(dialog)
        style.configure("CropPrimary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.configure("CropSecondary.TButton", padding=(12, 8))
        ttk.Label(
            content,
            text=f"Підготовлено {context.prepared} із {context.requested}",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            content,
            text=f"{context.skipped} фото не мають безпечного стандартного Krea crop.\nОригінали збережено у crop_skipped/.",
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 16))

        def choose(action: str) -> None:
            result["action"] = action
            dialog.grab_release()
            dialog.destroy()
            ready.set()

        actions = (
            ("Добрати інші фото", "Рекомендовано · взяти наступні strong candidates до потрібної кількості", "backfill"),
            ("Спробувати tighter crop", "Залишити identity/head; тілом і контекстом можна пожертвувати", "tight"),
            (f"Завершити з {context.prepared} фото", "Не добирати заміни; shortfall буде записаний у summary", "finish"),
        )
        primary_button: ttk.Button | None = None
        for row, (title, description, action) in enumerate(actions, 2):
            block = ttk.Frame(content)
            block.grid(row=row, column=0, sticky="ew", pady=(0, 10))
            block.columnconfigure(0, weight=1)
            button = ttk.Button(
                block,
                text=title,
                style="CropPrimary.TButton" if action == "backfill" else "CropSecondary.TButton",
                command=lambda value=action: choose(value),
            )
            button.grid(row=0, column=0, sticky="ew")
            if action == "backfill":
                primary_button = button
            ttk.Label(block, text=description, foreground="#666666", wraplength=470).grid(row=1, column=0, sticky="w", pady=(3, 0))

        dialog.protocol("WM_DELETE_WINDOW", lambda: choose("finish"))
        dialog.bind("<Escape>", lambda _event: choose("finish"))
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - dialog.winfo_width()) // 2)
        y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - dialog.winfo_height()) // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.grab_set()
        if primary_button is not None:
            primary_button.bind("<Return>", lambda _event: choose("backfill"))
            primary_button.focus_set()

    def _select_argv(self) -> list[str]:
        argv = [
            "select",
            self.dataset_var.get(),
            "--references", *self.references_var.get().split(),
            "--output", self.output_var.get(),
            "--models-dir", self.models_dir_var.get(),
            "--model-profile", self.model_profile_var.get(),
            "--count", self.count_var.get(),
            "--min-similarity", self.min_similarity_var.get(),
            "--min-quality", self.min_quality_var.get(),
            "--max-abs-yaw", self.max_abs_yaw_var.get(),
            "--min-face-width", self.min_face_width_var.get(),
            "--video-sample-fps", self.video_sample_fps_var.get(),
            "--video-max-samples", self.video_max_samples_var.get(),
            "--video-max-candidates", self.video_max_candidates_var.get(),
        ]
        if not self.prepare_crops_var.get():
            argv.append("--no-prepare-crops")
        if not self.appearance_rerank_var.get():
            argv.append("--no-appearance-rerank")
        if not self.body_attributes_var.get():
            argv.append("--no-body-attributes")
        if not self.parsing_previews_var.get():
            argv.append("--no-parsing-previews")
        if not self.analyze_videos_var.get():
            argv.append("--no-analyze-videos")
        if self.overwrite_var.get():
            argv.append("--overwrite")
        return argv

    def _require(self, *vars: tk.StringVar) -> bool:
        missing = [v for v in vars if not v.get().strip()]
        if missing:
            messagebox.showerror("Missing input", "Fill in dataset, references and output first.")
            return False
        return True

    def _run_download(self) -> None:
        self._run_command(["download-models", "--models-dir", self.models_dir_var.get(), "--profile", self.model_profile_var.get()])

    def _run_doctor(self) -> None:
        self._run_command(["doctor", "--models-dir", self.models_dir_var.get(), "--model-profile", self.model_profile_var.get()])

    def _run_select(self) -> None:
        if not self._require(self.dataset_var, self.references_var, self.output_var):
            return
        self._run_command(self._select_argv(), capture_stdin=True, resolve_crop_shortfall=True)

    def _open_review(self) -> None:
        output = Path(self.output_var.get())
        target = output / "review.html"
        if not target.is_file():
            messagebox.showinfo("Review", "Run select first — review.html not found yet.")
            return
        import webbrowser

        webbrowser.open(target.as_uri())

    def _on_close(self) -> None:
        if self._running:
            messagebox.showwarning("Busy", "A task is still running. Close after it finishes.")
            return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    LoraFaceSelectGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
