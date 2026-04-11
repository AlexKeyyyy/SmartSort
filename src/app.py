from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from copy import deepcopy
from pathlib import Path
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
except Exception as exc:  # pragma: no cover - depends on environment
    ctk = None
    CTK_IMPORT_ERROR = exc
else:
    CTK_IMPORT_ERROR = None

from src.mover import FileMover
from src.model_profiles import (
    build_model_profiles,
    first_available_model_key,
    is_profile_available,
    resolve_profile_model_path,
)
from src.predictor import SmartSortPredictor
from src.storage import Storage
from src.watcher import FolderWatcher


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = {
    "watch_dir": "",
    "output_dir": "",
    "confidence_threshold": 0.55,
    "auto_move": False,
    "selected_model_key": "transformer_distilbert",
    "categories": {
        "Code": "💻",
        "Docs": "📄",
        "Finance": "💰",
        "Study": "📚",
        "Media": "🎬",
        "Archives": "📦",
        "Logs": "📋",
        "Other": "📂",
    },
}


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        return deepcopy(DEFAULT_CONFIG)

    try:
        with path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except Exception:
        return deepcopy(DEFAULT_CONFIG)

    merged = deepcopy(DEFAULT_CONFIG)
    merged.update({key: value for key, value in loaded.items() if key != "categories"})
    merged["categories"].update(loaded.get("categories", {}))
    return merged


def save_config(config_path: str | Path, config: dict) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


if ctk is not None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")


if ctk is None:  # pragma: no cover - depends on environment
    class SmartSortApp:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"CustomTkinter is not available: {CTK_IMPORT_ERROR}")


else:
    class SmartSortApp(ctk.CTk):
        def __init__(self, config_path: str | Path, config: dict | None = None):
            super().__init__()

            self.base_dir = BASE_DIR
            self.config_path = Path(config_path)
            self.config_data = deepcopy(DEFAULT_CONFIG)
            if config:
                self.config_data.update({key: value for key, value in config.items() if key != "categories"})
                self.config_data["categories"].update(config.get("categories", {}))

            self.model_profiles = build_model_profiles(self.base_dir)
            self.model_display_to_key: dict[str, str] = {}
            self.model_key_to_display: dict[str, str] = {}
            self.active_model_key: str = ""
            self.active_model_path: Path | None = None
            self.active_model_display_name: str = ""

            startup_key = first_available_model_key(
                self.model_profiles,
                str(self.config_data.get("selected_model_key", "")),
            )
            if startup_key is None:
                raise FileNotFoundError(
                    "No available model found. Train at least one model first "
                    "(legacy: `python -m src.train_model`, transformer: `python -m src.train_transformer_model`)."
                )

            self.active_model_key = startup_key
            self.config_data["selected_model_key"] = startup_key
            self.predictor = self._load_predictor_for_key(startup_key)
            self.storage = Storage(self.base_dir / "smartsort.db")
            self.mover = FileMover(self._get_output_dir(), self.storage)
            self.watcher: FolderWatcher | None = None
            self.event_queue: queue.Queue = queue.Queue()
            self.watch_results: list[dict] = []
            self.pages: dict[str, ctk.CTkFrame] = {}
            self.nav_buttons: dict[str, ctk.CTkButton] = {}
            self.card_labels: dict[str, ctk.CTkLabel] = {}
            self.result_row_frames: list[ctk.CTkFrame] = []
            self.log_row_frames: list[ctk.CTkFrame] = []
            self.result_counter = 0
            self.queue_after_id: str | None = None
            self.is_closing = False

            self.title("SmartSort")
            self.geometry("1000x700")
            self.minsize(980, 680)

            self.watch_dir_var = ctk.StringVar(value=self.config_data.get("watch_dir", ""))
            self.output_dir_var = ctk.StringVar(value=self.config_data.get("output_dir", ""))
            self.auto_move_var = ctk.BooleanVar(value=bool(self.config_data.get("auto_move", False)))
            self.threshold_var = ctk.DoubleVar(value=float(self.config_data.get("confidence_threshold", 0.55)))
            self.status_var = ctk.StringVar(value="Stopped")
            self.status_color = "#d9534f"
            self.threshold_label_var = ctk.StringVar()
            self.model_var = ctk.StringVar(value="")
            self.model_info_var = ctk.StringVar(value="")

            self._build_layout()
            self._update_threshold_label()
            self._refresh_model_selector_options()
            self._sync_model_selector_with_active()
            self._update_model_info_label()
            self.refresh_dashboard()
            self.refresh_log()
            self.refresh_watcher_table()
            self._update_status_indicator()
            self.show_page("Dashboard")

            self.queue_after_id = self.after(250, self._process_event_queue)
            self.protocol("WM_DELETE_WINDOW", self.on_close)

        def _profile_is_available(self, model_key: str) -> bool:
            profile = self.model_profiles.get(model_key)
            if not profile:
                return False
            return is_profile_available(profile)

        def _resolve_model_path(self, model_key: str) -> Path | None:
            profile = self.model_profiles.get(model_key)
            if not profile:
                return None
            return resolve_profile_model_path(profile)

        def _model_display_name(self, model_key: str) -> str:
            profile = self.model_profiles[model_key]
            status = "ready" if self._profile_is_available(model_key) else "not trained"
            return f"{profile['title']} [{status}]"

        def _refresh_model_selector_options(self) -> None:
            self.model_display_to_key.clear()
            self.model_key_to_display.clear()

            values: list[str] = []
            for key in self.model_profiles:
                display = self._model_display_name(key)
                values.append(display)
                self.model_display_to_key[display] = key
                self.model_key_to_display[key] = display

            if hasattr(self, "model_selector"):
                self.model_selector.configure(values=values)

        def _sync_model_selector_with_active(self) -> None:
            display = self.model_key_to_display.get(self.active_model_key, "")
            if display:
                self.model_var.set(display)

        def _load_predictor_for_key(self, model_key: str) -> SmartSortPredictor:
            path = self._resolve_model_path(model_key)
            if path is None:
                profile = self.model_profiles[model_key]
                raise FileNotFoundError(
                    f"Selected model is not available: {profile['title']}. "
                    f"Expected path: {profile['model_path']}"
                )

            predictor = SmartSortPredictor(path)
            current_threshold = float(self.config_data.get("confidence_threshold", 0.55))
            if hasattr(self, "threshold_var"):
                try:
                    current_threshold = float(self.threshold_var.get())
                except Exception:
                    pass
            predictor.threshold = current_threshold
            self.active_model_path = path
            self.active_model_display_name = self.model_profiles[model_key]["title"]
            return predictor

        def _switch_model(self, model_key: str, notify: bool = False, force_reload: bool = False) -> bool:
            if model_key == self.active_model_key and not force_reload:
                return True

            if not self._profile_is_available(model_key):
                missing_profile = self.model_profiles[model_key]
                messagebox.showwarning(
                    "SmartSort",
                    f"Модель пока не готова: {missing_profile['title']}.\n"
                    f"Ожидается: {missing_profile['model_path']}",
                )
                self._sync_model_selector_with_active()
                return False

            try:
                self.predictor = self._load_predictor_for_key(model_key)
            except Exception as exc:
                messagebox.showerror("SmartSort", f"Не удалось переключить модель: {exc}")
                self._sync_model_selector_with_active()
                return False

            self.active_model_key = model_key
            self.config_data["selected_model_key"] = model_key
            if self.watcher is not None:
                self.watcher.predictor = self.predictor
                self.watcher.event_handler.predictor = self.predictor

            self._sync_model_selector_with_active()
            self._update_model_info_label()
            if notify:
                messagebox.showinfo("SmartSort", f"Активна модель: {self.active_model_display_name}")
            return True

        def _on_model_selected(self, selected_display: str) -> None:
            model_key = self.model_display_to_key.get(selected_display)
            if not model_key:
                return
            self._switch_model(model_key, notify=False)

        def _update_model_info_label(self) -> None:
            profile = self.model_profiles.get(self.active_model_key, {})
            backend = str(profile.get("backend", "unknown"))
            model_path = str(self.active_model_path) if self.active_model_path else "n/a"
            self.model_info_var.set(
                f"Active: {self.active_model_display_name} | backend: {backend} | path: {model_path}"
            )

        def _get_output_dir(self) -> Path:
            raw_output = self.config_data.get("output_dir", "")
            if raw_output:
                return Path(raw_output)
            return self.base_dir / "sorted"

        def _build_layout(self) -> None:
            self.grid_columnconfigure(1, weight=1)
            self.grid_rowconfigure(0, weight=1)

            sidebar = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color="#111827")
            sidebar.grid(row=0, column=0, sticky="nsew")
            sidebar.grid_propagate(False)

            logo = ctk.CTkLabel(
                sidebar,
                text="SmartSort",
                font=ctk.CTkFont(size=28, weight="bold"),
            )
            logo.pack(padx=20, pady=(28, 24), anchor="w")

            for name in ["Dashboard", "Watcher", "Log", "Settings"]:
                button = ctk.CTkButton(
                    sidebar,
                    text=name,
                    height=42,
                    anchor="w",
                    command=lambda value=name: self.show_page(value),
                )
                button.pack(fill="x", padx=16, pady=6)
                self.nav_buttons[name] = button

            status_frame = ctk.CTkFrame(sidebar, fg_color="#1f2937")
            status_frame.pack(fill="x", padx=16, pady=(24, 12), side="bottom")
            self.status_dot = ctk.CTkLabel(status_frame, text="●", font=ctk.CTkFont(size=20))
            self.status_dot.pack(side="left", padx=(14, 8), pady=12)
            self.status_label = ctk.CTkLabel(status_frame, textvariable=self.status_var)
            self.status_label.pack(side="left", pady=12)

            content = ctk.CTkFrame(self, fg_color="transparent")
            content.grid(row=0, column=1, sticky="nsew")
            content.grid_rowconfigure(0, weight=1)
            content.grid_columnconfigure(0, weight=1)
            self.content_frame = content

            self.pages["Dashboard"] = self._build_dashboard_page(content)
            self.pages["Watcher"] = self._build_watcher_page(content)
            self.pages["Log"] = self._build_log_page(content)
            self.pages["Settings"] = self._build_settings_page(content)

            for frame in self.pages.values():
                frame.grid(row=0, column=0, sticky="nsew")

        def _build_dashboard_page(self, parent):
            page = ctk.CTkFrame(parent, fg_color="transparent")
            for column in range(4):
                page.grid_columnconfigure(column, weight=1)

            title = ctk.CTkLabel(page, text="Dashboard", font=ctk.CTkFont(size=26, weight="bold"))
            title.grid(row=0, column=0, columnspan=4, sticky="w", padx=20, pady=(20, 18))

            categories = list(self.config_data["categories"].items())
            for index, (category, icon) in enumerate(categories):
                row = index // 4 + 1
                column = index % 4
                card = ctk.CTkFrame(page, fg_color="#162033")
                card.grid(row=row, column=column, padx=12, pady=12, sticky="nsew")
                card.grid_columnconfigure(0, weight=1)

                ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=28)).grid(
                    row=0, column=0, sticky="w", padx=16, pady=(16, 8)
                )
                ctk.CTkLabel(card, text=category, font=ctk.CTkFont(size=18, weight="bold")).grid(
                    row=1, column=0, sticky="w", padx=16
                )
                count_label = ctk.CTkLabel(card, text="0", font=ctk.CTkFont(size=30, weight="bold"))
                count_label.grid(row=2, column=0, sticky="w", padx=16, pady=(10, 18))
                self.card_labels[category] = count_label

            self.dashboard_undo_button = ctk.CTkButton(
                page,
                text="Undo последнее действие",
                command=self.undo_last_action,
                height=42,
            )
            self.dashboard_undo_button.grid(row=3, column=0, columnspan=4, sticky="ew", padx=20, pady=(18, 20))

            return page

        def _build_watcher_page(self, parent):
            page = ctk.CTkFrame(parent, fg_color="transparent")
            page.grid_columnconfigure(0, weight=1)
            page.grid_rowconfigure(5, weight=1)

            ctk.CTkLabel(page, text="Watcher", font=ctk.CTkFont(size=26, weight="bold")).grid(
                row=0, column=0, sticky="w", padx=20, pady=(20, 18)
            )

            watch_row = ctk.CTkFrame(page, fg_color="transparent")
            watch_row.grid(row=1, column=0, sticky="ew", padx=20, pady=6)
            watch_row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(watch_row, text="Папка для слежения").grid(row=0, column=0, sticky="w", pady=(0, 6))
            self.watch_dir_entry = ctk.CTkEntry(watch_row, textvariable=self.watch_dir_var, height=38)
            self.watch_dir_entry.grid(row=1, column=0, sticky="ew", padx=(0, 10))
            ctk.CTkButton(watch_row, text="Browse", width=110, command=self.browse_watch_dir).grid(
                row=1, column=1, sticky="e"
            )

            out_row = ctk.CTkFrame(page, fg_color="transparent")
            out_row.grid(row=2, column=0, sticky="ew", padx=20, pady=6)
            out_row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(out_row, text="Папка назначения").grid(row=0, column=0, sticky="w", pady=(0, 6))
            self.output_dir_entry = ctk.CTkEntry(out_row, textvariable=self.output_dir_var, height=38)
            self.output_dir_entry.grid(row=1, column=0, sticky="ew", padx=(0, 10))
            ctk.CTkButton(out_row, text="Browse", width=110, command=self.browse_output_dir).grid(
                row=1, column=1, sticky="e"
            )

            controls = ctk.CTkFrame(page, fg_color="transparent")
            controls.grid(row=3, column=0, sticky="ew", padx=20, pady=10)
            controls.grid_columnconfigure(4, weight=1)
            self.auto_move_checkbox = ctk.CTkCheckBox(
                controls,
                text="Авто-перемещение",
                variable=self.auto_move_var,
            )
            self.auto_move_checkbox.grid(row=0, column=0, sticky="w", padx=(0, 16))
            self.start_stop_button = ctk.CTkButton(controls, text="Start", width=120, command=self.toggle_watcher)
            self.start_stop_button.grid(row=0, column=1, padx=6)
            self.apply_all_button = ctk.CTkButton(
                controls,
                text="Применить всё",
                width=140,
                command=self.apply_all_pending,
            )
            self.apply_all_button.grid(row=0, column=2, padx=6)

            table_frame = ctk.CTkFrame(page, fg_color="#0f172a")
            table_frame.grid(row=5, column=0, sticky="nsew", padx=20, pady=(8, 20))
            table_frame.grid_rowconfigure(1, weight=1)
            table_frame.grid_columnconfigure(0, weight=1)

            header = ctk.CTkFrame(table_frame, fg_color="#1e293b")
            header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
            for idx, text in enumerate(["Файл", "Категория", "Confidence", "Статус"]):
                header.grid_columnconfigure(idx, weight=1)
                ctk.CTkLabel(header, text=text, anchor="w").grid(row=0, column=idx, sticky="ew", padx=10, pady=10)

            self.results_scroll = ctk.CTkScrollableFrame(table_frame, fg_color="transparent")
            self.results_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
            self.results_scroll.grid_columnconfigure(0, weight=1)

            return page

        def _build_log_page(self, parent):
            page = ctk.CTkFrame(parent, fg_color="transparent")
            page.grid_rowconfigure(1, weight=1)
            page.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(page, text="Log", font=ctk.CTkFont(size=26, weight="bold")).grid(
                row=0, column=0, sticky="w", padx=20, pady=(20, 18)
            )

            table_frame = ctk.CTkFrame(page, fg_color="#0f172a")
            table_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
            table_frame.grid_rowconfigure(1, weight=1)
            table_frame.grid_columnconfigure(0, weight=1)

            header = ctk.CTkFrame(table_frame, fg_color="#1e293b")
            header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
            for idx, text in enumerate(["Время", "Файл", "Из", "В", "Confidence", "Undo"]):
                header.grid_columnconfigure(idx, weight=1)
                ctk.CTkLabel(header, text=text, anchor="w").grid(row=0, column=idx, sticky="ew", padx=8, pady=10)

            self.log_scroll = ctk.CTkScrollableFrame(table_frame, fg_color="transparent")
            self.log_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
            self.log_scroll.grid_columnconfigure(0, weight=1)

            return page

        def _build_settings_page(self, parent):
            page = ctk.CTkFrame(parent, fg_color="transparent")
            page.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(page, text="Settings", font=ctk.CTkFont(size=26, weight="bold")).grid(
                row=0, column=0, sticky="w", padx=20, pady=(20, 18)
            )

            model_frame = ctk.CTkFrame(page, fg_color="#162033")
            model_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
            model_frame.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(model_frame, text="Model profile").grid(
                row=0, column=0, sticky="w", padx=16, pady=(14, 8)
            )
            self.model_selector = ctk.CTkOptionMenu(
                model_frame,
                values=["Loading..."],
                variable=self.model_var,
                command=self._on_model_selected,
            )
            self.model_selector.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
            ctk.CTkLabel(
                model_frame,
                text="Switch between legacy baseline and neural profiles.",
                text_color="#93a4c3",
                justify="left",
            ).grid(row=2, column=0, sticky="w", padx=16, pady=(0, 2))
            ctk.CTkLabel(
                model_frame,
                textvariable=self.model_info_var,
                justify="left",
                wraplength=760,
            ).grid(row=3, column=0, sticky="w", padx=16, pady=(0, 14))

            threshold_frame = ctk.CTkFrame(page, fg_color="#162033")
            threshold_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
            threshold_frame.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(threshold_frame, text="Порог confidence").grid(
                row=0, column=0, sticky="w", padx=16, pady=(14, 8)
            )
            self.threshold_slider = ctk.CTkSlider(
                threshold_frame,
                from_=0.0,
                to=1.0,
                variable=self.threshold_var,
                command=lambda _value: self._update_threshold_label(),
            )
            self.threshold_slider.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
            self.threshold_value_label = ctk.CTkLabel(threshold_frame, textvariable=self.threshold_label_var)
            self.threshold_value_label.grid(row=2, column=0, sticky="w", padx=16, pady=(0, 14))

            buttons_frame = ctk.CTkFrame(page, fg_color="transparent")
            buttons_frame.grid(row=3, column=0, sticky="w", padx=20, pady=10)
            ctk.CTkButton(buttons_frame, text="Refresh Model List", command=self.refresh_model_profiles).pack(
                side="left", padx=(0, 12)
            )
            ctk.CTkButton(buttons_frame, text="Retrain Active Model", command=self.retrain_model).pack(
                side="left", padx=(0, 12)
            )
            ctk.CTkButton(buttons_frame, text="Сохранить настройки", command=self.save_settings).pack(side="left")

            return page

        def show_page(self, name: str) -> None:
            for page_name, frame in self.pages.items():
                if page_name == name:
                    frame.tkraise()

            active_color = "#2563eb"
            default_color = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
            for page_name, button in self.nav_buttons.items():
                button.configure(fg_color=active_color if page_name == name else default_color)

        def _update_threshold_label(self) -> None:
            self.threshold_label_var.set(f"Текущее значение: {self.threshold_var.get():.2f}")

        def refresh_model_profiles(self, notify: bool = True) -> None:
            self.model_profiles = build_model_profiles(self.base_dir)
            self._refresh_model_selector_options()

            preferred_key = self.active_model_key or str(self.config_data.get("selected_model_key", ""))
            next_key = first_available_model_key(self.model_profiles, preferred_key)
            if next_key is None:
                messagebox.showerror(
                    "SmartSort",
                    "No trained models are available. Train a model and try again.",
                )
                return

            switched = self._switch_model(next_key, notify=False)
            if switched:
                self._update_model_info_label()
                if notify:
                    messagebox.showinfo("SmartSort", "Model list refreshed.")

        def _update_status_indicator(self) -> None:
            running = self.watcher is not None and self.watcher.is_running()
            self.status_var.set("Watching" if running else "Stopped")
            self.status_dot.configure(text_color="#22c55e" if running else "#ef4444")
            self.start_stop_button.configure(text="Stop" if running else "Start")

        def _process_event_queue(self) -> None:
            if self.is_closing:
                return

            try:
                while True:
                    event_type, payload = self.event_queue.get_nowait()
                    if event_type == "prediction":
                        self._handle_prediction(payload)
                    elif event_type == "move_complete":
                        self._handle_move_complete(payload)
                    elif event_type == "undo_complete":
                        self._handle_undo_complete(payload)
                    elif event_type == "retrain_complete":
                        self._handle_retrain_complete(payload)
            except queue.Empty:
                pass
            finally:
                if not self.is_closing:
                    self.queue_after_id = self.after(250, self._process_event_queue)

        def _handle_prediction(self, result: dict) -> None:
            self.result_counter += 1
            result["id"] = self.result_counter
            result["status"] = "Pending"
            self.watch_results.insert(0, result)
            self.refresh_watcher_table()

            if self.auto_move_var.get():
                self._queue_move(result["id"])

        def _handle_move_complete(self, payload: dict) -> None:
            for item in self.watch_results:
                if item["id"] == payload["id"]:
                    item["status"] = payload["status"]
                    item["moved_to"] = payload.get("moved_to", "")
                    item["error"] = payload.get("error", "")
                    break
            self.refresh_watcher_table()
            self.refresh_dashboard()
            self.refresh_log()

        def _handle_undo_complete(self, payload: dict) -> None:
            if payload["success"]:
                self.refresh_dashboard()
                self.refresh_log()
                messagebox.showinfo("SmartSort", "Последнее действие успешно отменено.")
            else:
                messagebox.showwarning("SmartSort", payload.get("message", "Не удалось выполнить откат."))

        def _handle_retrain_complete(self, payload: dict) -> None:
            if payload["success"]:
                self.refresh_model_profiles(notify=False)
                self._switch_model(self.active_model_key, notify=False, force_reload=True)
                messagebox.showinfo("SmartSort", "Model retrained and reloaded.")
            else:
                messagebox.showerror("SmartSort", payload.get("message", "Переобучение завершилось ошибкой."))

        def refresh_dashboard(self) -> None:
            stats = self.storage.get_stats()
            for category in self.config_data["categories"]:
                self.card_labels[category].configure(text=str(stats.get(category, 0)))

        def refresh_watcher_table(self) -> None:
            for frame in self.result_row_frames:
                frame.destroy()
            self.result_row_frames.clear()

            for row_index, item in enumerate(self.watch_results[:100]):
                frame = ctk.CTkFrame(self.results_scroll, fg_color=self._confidence_color(item["confidence"]))
                frame.grid(row=row_index, column=0, sticky="ew", pady=4)
                for idx, text in enumerate(
                    [
                        item["filename"],
                        f"{item['icon']} {item['category']}",
                        f"{item['confidence']:.0%}",
                        item["status"],
                    ]
                ):
                    frame.grid_columnconfigure(idx, weight=1)
                    ctk.CTkLabel(frame, text=text, anchor="w").grid(
                        row=0, column=idx, sticky="ew", padx=10, pady=10
                    )
                self.result_row_frames.append(frame)

        def refresh_log(self) -> None:
            for frame in self.log_row_frames:
                frame.destroy()
            self.log_row_frames.clear()

            rows = self.storage.get_recent(limit=50)
            latest_active_id = next((row["id"] for row in rows if row["undone"] == 0), None)

            for row_index, item in enumerate(rows):
                frame = ctk.CTkFrame(self.log_scroll, fg_color="#111827" if item["undone"] == 0 else "#1f2937")
                frame.grid(row=row_index, column=0, sticky="ew", pady=4)
                for column in range(5):
                    frame.grid_columnconfigure(column, weight=1)

                values = [
                    item["ts"],
                    Path(item["src"]).name,
                    item["src"],
                    item["dst"],
                    f"{item['confidence']:.0%}",
                ]
                for column, text in enumerate(values):
                    ctk.CTkLabel(frame, text=text, anchor="w").grid(
                        row=0, column=column, sticky="ew", padx=8, pady=10
                    )

                button = ctk.CTkButton(
                    frame,
                    text="Undo",
                    width=80,
                    command=self.undo_last_action,
                    state="normal" if item["undone"] == 0 and item["id"] == latest_active_id else "disabled",
                )
                button.grid(row=0, column=5, padx=8, pady=8)
                self.log_row_frames.append(frame)

        @staticmethod
        def _confidence_color(confidence: float) -> str:
            if confidence >= 0.8:
                return "#133b2b"
            if confidence >= 0.55:
                return "#4a3d18"
            return "#374151"

        def browse_watch_dir(self) -> None:
            selected = filedialog.askdirectory(title="Выберите папку для слежения")
            if selected:
                self.watch_dir_var.set(selected)

        def browse_output_dir(self) -> None:
            selected = filedialog.askdirectory(title="Выберите папку назначения")
            if selected:
                self.output_dir_var.set(selected)
                self.mover = FileMover(Path(selected), self.storage)

        def toggle_watcher(self) -> None:
            if self.watcher is not None and self.watcher.is_running():
                self.stop_watcher()
            else:
                self.start_watcher()

        def start_watcher(self) -> None:
            watch_dir = self.watch_dir_var.get().strip()
            output_dir = self.output_dir_var.get().strip()
            if not watch_dir:
                messagebox.showwarning("SmartSort", "Укажите папку для слежения.")
                return
            if not output_dir:
                messagebox.showwarning("SmartSort", "Укажите папку назначения.")
                return

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.mover = FileMover(Path(output_dir), self.storage)

            try:
                self.watcher = FolderWatcher(watch_dir, self.enqueue_prediction, self.predictor)
                self.watcher.start()
            except Exception as exc:
                self.watcher = None
                messagebox.showerror("SmartSort", f"Не удалось запустить watcher: {exc}")
                return

            self._update_status_indicator()

        def stop_watcher(self) -> None:
            if self.watcher is not None:
                self.watcher.stop()
                self.watcher = None
            self._update_status_indicator()

        def enqueue_prediction(self, result: dict) -> None:
            self.event_queue.put(("prediction", result))

        def _queue_move(self, record_id: int) -> None:
            item = next((entry for entry in self.watch_results if entry["id"] == record_id), None)
            if item is None or item["status"] not in {"Pending", "Skipped"}:
                return

            item["status"] = "Moving"
            self.refresh_watcher_table()
            threshold = float(self.threshold_var.get())
            thread = threading.Thread(target=self._move_result_worker, args=(record_id, threshold), daemon=True)
            thread.start()

        def _move_result_worker(self, record_id: int, threshold: float) -> None:
            item = next((entry for entry in self.watch_results if entry["id"] == record_id), None)
            if item is None:
                return

            if item["confidence"] < threshold:
                self.event_queue.put(
                    ("move_complete", {"id": record_id, "status": f"Skipped (< {threshold:.2f})"})
                )
                return

            try:
                destination = self.mover.move_file(
                    item["filepath"],
                    item["category"],
                    item["confidence"],
                )
                self.event_queue.put(
                    ("move_complete", {"id": record_id, "status": "Moved", "moved_to": destination})
                )
            except Exception as exc:
                self.event_queue.put(
                    ("move_complete", {"id": record_id, "status": "Error", "error": str(exc)})
                )

        def apply_all_pending(self) -> None:
            if not self.output_dir_var.get().strip():
                messagebox.showwarning("SmartSort", "Сначала укажите папку назначения.")
                return

            pending_ids = [
                item["id"]
                for item in self.watch_results
                if item["status"] == "Pending"
            ]
            for record_id in pending_ids:
                self._queue_move(record_id)

        def undo_last_action(self) -> None:
            thread = threading.Thread(target=self._undo_worker, daemon=True)
            thread.start()

        def _undo_worker(self) -> None:
            try:
                success = self.mover.undo_last()
                self.event_queue.put(
                    (
                        "undo_complete",
                        {
                            "success": success,
                            "message": "Нет действий для отката." if not success else "",
                        },
                    )
                )
            except Exception as exc:
                self.event_queue.put(("undo_complete", {"success": False, "message": str(exc)}))

        def retrain_model(self) -> None:
            thread = threading.Thread(target=self._retrain_worker, daemon=True)
            thread.start()

        def _build_retrain_command(self) -> list[str]:
            profile = self.model_profiles[self.active_model_key]
            backend = profile["backend"]
            if backend == "legacy":
                return [sys.executable, "-m", "src.train_model"]

            model_name = str(profile.get("model_name", "")).strip()
            output_dir = Path(profile["model_path"])
            report_dir = Path(profile["report_dir"])
            if not model_name:
                raise ValueError(f"Missing model_name in profile: {self.active_model_key}")

            return [
                sys.executable,
                "-m",
                "src.train_transformer_model",
                "--model-name",
                model_name,
                "--output-dir",
                str(output_dir),
                "--report-dir",
                str(report_dir),
                "--no-auto-quick",
            ]

        def _retrain_worker(self) -> None:
            try:
                train_command = self._build_retrain_command()
                subprocess.run(train_command, cwd=self.base_dir, check=True)
                self.event_queue.put(("retrain_complete", {"success": True}))
            except Exception as exc:
                self.event_queue.put(("retrain_complete", {"success": False, "message": str(exc)}))

        def save_settings(self) -> None:
            self.config_data["watch_dir"] = self.watch_dir_var.get().strip()
            self.config_data["output_dir"] = self.output_dir_var.get().strip()
            self.config_data["auto_move"] = bool(self.auto_move_var.get())
            self.config_data["confidence_threshold"] = round(float(self.threshold_var.get()), 2)
            self.config_data["selected_model_key"] = self.active_model_key
            self.predictor.threshold = float(self.config_data["confidence_threshold"])

            try:
                save_config(self.config_path, self.config_data)
                messagebox.showinfo("SmartSort", "Настройки сохранены.")
            except Exception as exc:
                messagebox.showerror("SmartSort", f"Не удалось сохранить настройки: {exc}")

        def on_close(self) -> None:
            self.is_closing = True
            if self.queue_after_id is not None:
                try:
                    self.after_cancel(self.queue_after_id)
                except Exception:
                    pass
                self.queue_after_id = None
            self.stop_watcher()
            self.destroy()


def run_app(config_path: str | Path = BASE_DIR / "config.json", config: dict | None = None) -> None:
    if ctk is None:  # pragma: no cover - depends on environment
        raise RuntimeError(f"CustomTkinter is not available: {CTK_IMPORT_ERROR}")

    app = SmartSortApp(config_path=config_path, config=config or load_config(config_path))
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    run_app()
