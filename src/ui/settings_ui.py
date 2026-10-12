"""
Tkinter-based settings UI for the Audio Priority Control System.

Lets the user:
  - Drag category rows to reorder priorities
  - Adjust volume multipliers via sliders
  - Set base volume and response/recovery timings
  - View real-time classification output and current volume
  - Start/stop the control engine
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

CATEGORY_COLORS = {
    "alarm":       "#FF4444",
    "speech":      "#4CAF50",
    "music":       "#2196F3",
    "keyboard":    "#FF9800",
    "environment": "#9E9E9E",
}


class PriorityRow(tk.Frame):
    """Single draggable row representing one sound category."""

    def __init__(self, parent, category: str, display_name: str,
                 priority: int, multiplier: float,
                 on_multiplier_change: Callable):
        super().__init__(parent, bd=1, relief=tk.GROOVE, padx=6, pady=4)
        self.category = category
        self._on_change = on_multiplier_change

        color = CATEGORY_COLORS.get(category, "#888888")

        # Priority badge
        tk.Label(self, text=f"#{priority}", width=3,
                 font=("Helvetica", 11, "bold"),
                 fg=color).pack(side=tk.LEFT, padx=(0, 8))

        # Category name
        tk.Label(self, text=display_name, width=20,
                 anchor="w", font=("Helvetica", 11)).pack(side=tk.LEFT)

        # Volume multiplier slider
        tk.Label(self, text="볼륨 ×", font=("Helvetica", 10)).pack(side=tk.LEFT)
        self._var = tk.DoubleVar(value=multiplier)
        slider = ttk.Scale(self, from_=0.0, to=1.5, variable=self._var,
                           orient=tk.HORIZONTAL, length=160,
                           command=lambda _: self._on_change(category, self._var.get()))
        slider.pack(side=tk.LEFT, padx=4)
        self._label = tk.Label(self, text=f"{multiplier:.2f}", width=5,
                               font=("Helvetica", 10))
        self._label.pack(side=tk.LEFT)

    def update_priority_badge(self, priority: int):
        for w in self.winfo_children():
            if isinstance(w, tk.Label) and w.cget("text").startswith("#"):
                w.config(text=f"#{priority}")
                break

    def update_value_label(self, value: float):
        self._label.config(text=f"{value:.2f}")


class SettingsWindow:
    """Main settings window."""

    SAVE_DELAY_MS = 500   # slider drags are written to disk once they settle

    def __init__(self, priority_manager, engine=None):
        self._pm = priority_manager
        self._engine = engine

        self._root = tk.Tk()
        self._root.title("오디오 우선순위 제어 시스템")
        self._root.geometry("700x600")
        self._root.resizable(False, False)

        self._rows: dict[str, PriorityRow] = {}
        self._status_var = tk.StringVar(value="대기 중")
        self._volume_var = tk.StringVar(value="볼륨: --")
        self._detection_var = tk.StringVar(value="감지: --")
        self._running = False
        self._save_job: Optional[str] = None

        self._build_ui()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = self._root

        # ── Header ──────────────────────────────────────────────────
        header = tk.Frame(root, bg="#1a1a2e", pady=12)
        header.pack(fill=tk.X)
        tk.Label(header, text="🎵 오디오 우선순위 제어 시스템",
                 font=("Helvetica", 16, "bold"),
                 fg="white", bg="#1a1a2e").pack()
        tk.Label(header, text="소리 유형을 실시간으로 분류하여 볼륨을 자동 조절합니다",
                 font=("Helvetica", 10),
                 fg="#aaaaaa", bg="#1a1a2e").pack()

        # ── Priority List ────────────────────────────────────────────
        frame_label = tk.Frame(root, pady=4)
        frame_label.pack(fill=tk.X, padx=20)
        tk.Label(frame_label, text="소리 우선순위 설정 (높을수록 먼저 처리)",
                 font=("Helvetica", 12, "bold")).pack(anchor="w")
        tk.Label(frame_label,
                 text="↑ ↓ 버튼으로 우선순위 변경 | 슬라이더로 감지 시 볼륨 배수 조절",
                 font=("Helvetica", 9), fg="#555555").pack(anchor="w")

        self._list_frame = tk.Frame(root, padx=20)
        self._list_frame.pack(fill=tk.X)

        self._rebuild_priority_list()

        # ── General Settings ─────────────────────────────────────────
        settings_frame = ttk.LabelFrame(root, text="일반 설정", padding=10)
        settings_frame.pack(fill=tk.X, padx=20, pady=(10, 4))

        settings = self._pm.get_settings()
        self._base_vol_var = tk.DoubleVar(value=settings.get("base_volume", 1.0))
        self._resp_var = tk.IntVar(value=settings.get("response_time_ms", 200))
        self._recv_var = tk.IntVar(value=settings.get("recovery_time_ms", 1500))
        self._thresh_var = tk.DoubleVar(value=settings.get("detection_threshold", 0.3))

        self._add_slider_row(settings_frame, "기준 볼륨 (시작 시 대비)", self._base_vol_var,
                             0.0, 1.5,
                             lambda v: self._on_setting_change("base_volume", round(float(v), 2)),
                             row=0)
        self._add_slider_row(settings_frame, "응답 속도 (ms)", self._resp_var,
                             50, 1000, lambda v: self._on_setting_change("response_time_ms", int(v)),
                             row=1)
        self._add_slider_row(settings_frame, "복구 시간 (ms)", self._recv_var,
                             500, 5000, lambda v: self._on_setting_change("recovery_time_ms", int(v)),
                             row=2)
        self._add_slider_row(settings_frame, "감지 임계값", self._thresh_var,
                             0.1, 0.9,
                             lambda v: self._on_setting_change("detection_threshold", round(float(v), 2)),
                             row=3)

        # ── Status Bar ───────────────────────────────────────────────
        status_frame = tk.Frame(root, bg="#f0f0f0", pady=6)
        status_frame.pack(fill=tk.X, padx=20, pady=(4, 0))

        tk.Label(status_frame, textvariable=self._detection_var,
                 font=("Helvetica", 11), bg="#f0f0f0").pack(side=tk.LEFT, padx=8)
        tk.Label(status_frame, textvariable=self._volume_var,
                 font=("Helvetica", 11), bg="#f0f0f0").pack(side=tk.LEFT, padx=8)

        # ── Controls ─────────────────────────────────────────────────
        btn_frame = tk.Frame(root, pady=10)
        btn_frame.pack()

        self._start_btn = tk.Button(
            btn_frame, text="▶  시작", width=14, height=2,
            font=("Helvetica", 11, "bold"),
            bg="#4CAF50", fg="white",
            command=self._toggle_engine,
        )
        self._start_btn.pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_frame, text="기본값 복원", width=12, height=2,
            font=("Helvetica", 10),
            command=self._reset_defaults,
        ).pack(side=tk.LEFT, padx=6)

        tk.Label(root, textvariable=self._status_var,
                 font=("Helvetica", 9), fg="#888888").pack(pady=(0, 2))
        tk.Label(root,
                 text="※ 스피커 사용 시 마이크가 재생 중인 소리를 다시 감지할 수 있어 헤드폰을 권장합니다",
                 font=("Helvetica", 9), fg="#888888").pack(pady=(0, 6))

    def _add_slider_row(self, parent, label, var, from_, to, command, row):
        tk.Label(parent, text=label, width=18, anchor="w").grid(row=row, column=0, sticky="w")
        ttk.Scale(parent, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL,
                  length=200, command=lambda _: command(var.get())).grid(
            row=row, column=1, padx=6)
        tk.Label(parent, textvariable=var, width=8).grid(row=row, column=2)

    def _rebuild_priority_list(self):
        for w in self._list_frame.winfo_children():
            w.destroy()
        self._rows.clear()

        for cat, info in self._pm.sorted_categories():
            row_frame = tk.Frame(self._list_frame)
            row_frame.pack(fill=tk.X, pady=2)

            # Up/Down arrows
            btn_up = tk.Button(row_frame, text="▲", width=2,
                               command=lambda c=cat: self._move_up(c))
            btn_up.pack(side=tk.LEFT, padx=(0, 2))
            btn_dn = tk.Button(row_frame, text="▼", width=2,
                               command=lambda c=cat: self._move_down(c))
            btn_dn.pack(side=tk.LEFT, padx=(0, 4))

            row = PriorityRow(
                row_frame, cat, info["display_name"],
                info["priority"], info["volume_multiplier"],
                on_multiplier_change=self._on_multiplier_change,
            )
            row.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._rows[cat] = row

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_multiplier_change(self, category: str, value: float):
        # The engine reads multipliers from the manager on every chunk
        self._pm.set_volume_multiplier(category, round(value, 2), save=False)
        self._schedule_save()
        if category in self._rows:
            self._rows[category].update_value_label(value)

    def _on_setting_change(self, key: str, value):
        self._pm.update_setting(key, value, save=False)
        self._schedule_save()
        if self._engine:
            self._engine.update_settings(self._pm.get_settings())

    def _schedule_save(self):
        if self._save_job is not None:
            self._root.after_cancel(self._save_job)
        self._save_job = self._root.after(self.SAVE_DELAY_MS, self._flush_save)

    def _flush_save(self):
        if self._save_job is not None:
            self._root.after_cancel(self._save_job)
            self._save_job = None
        self._pm.save()

    def _move_up(self, category: str):
        cats = self._pm.sorted_categories()
        idx = next(i for i, (c, _) in enumerate(cats) if c == category)
        if idx > 0:
            self._pm.swap_priorities(category, cats[idx - 1][0])
            self._rebuild_priority_list()

    def _move_down(self, category: str):
        cats = self._pm.sorted_categories()
        idx = next(i for i, (c, _) in enumerate(cats) if c == category)
        if idx < len(cats) - 1:
            self._pm.swap_priorities(category, cats[idx + 1][0])
            self._rebuild_priority_list()

    def _toggle_engine(self):
        if not self._engine:
            messagebox.showinfo("안내", "엔진이 연결되지 않았습니다.\nmain.py를 통해 실행하세요.")
            return
        if self._running:
            self._engine.stop()
            self._running = False
            self._start_btn.config(text="▶  시작", bg="#4CAF50")
            self._status_var.set("중지됨")
        else:
            self._engine.start()
            self._running = True
            self._start_btn.config(text="⏹  중지", bg="#F44336")
            self._status_var.set("실행 중 ...")
            self._poll_status()

    def _reset_defaults(self):
        if messagebox.askyesno("확인", "우선순위를 기본값으로 되돌릴까요?"):
            if self._save_job is not None:
                self._root.after_cancel(self._save_job)
                self._save_job = None
            self._pm.reset_to_defaults()
            self._rebuild_priority_list()

            settings = self._pm.get_settings()
            self._base_vol_var.set(settings.get("base_volume", 1.0))
            self._resp_var.set(settings.get("response_time_ms", 200))
            self._recv_var.set(settings.get("recovery_time_ms", 1500))
            self._thresh_var.set(settings.get("detection_threshold", 0.3))
            if self._engine:
                self._engine.update_settings(settings)

    def _on_close(self):
        self._flush_save()
        if self._engine and self._running:
            self._engine.stop()
            self._running = False
        self._root.destroy()

    def _poll_status(self):
        if not self._running or not self._engine:
            return
        status = self._engine.get_status()
        if status:
            cat = status.get("top_category", "--")
            conf = status.get("confidence", 0.0)
            vol = status.get("volume", 0.0)
            self._detection_var.set(f"감지: {cat} ({conf:.0%})")
            self._volume_var.set(f"볼륨: {vol:.0%}")
        self._root.after(300, self._poll_status)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_status(self, category: str, confidence: float, volume: float):
        """Thread-safe status update (call from engine thread)."""
        self._root.after(0, lambda: (
            self._detection_var.set(f"감지: {category} ({confidence:.0%})"),
            self._volume_var.set(f"볼륨: {volume:.0%}"),
        ))

    def run(self):
        self._root.mainloop()
