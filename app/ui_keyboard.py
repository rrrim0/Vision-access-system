from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class OnScreenKeyboard(ttk.Frame):
    def __init__(self, master: tk.Misc, target_entry_getter):
        super().__init__(master)
        self.target_entry_getter = target_entry_getter
        self._build_ui()

    def _build_ui(self) -> None:
        rows = [
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
            ["Z", "X", "C", "V", "B", "N", "M"],
        ]

        for row_keys in rows:
            row = ttk.Frame(self)
            row.pack(fill="x", pady=2)
            for key in row_keys:
                ttk.Button(
                    row,
                    text=key,
                    width=3,
                    command=lambda k=key: self.insert_text(k),
                ).pack(side="left", padx=2, pady=1)

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", pady=2)

        ttk.Button(bottom, text="Пробел", command=lambda: self.insert_text(" ")).pack(side="left", padx=2)
        ttk.Button(bottom, text="-", width=3, command=lambda: self.insert_text("-")).pack(side="left", padx=2)
        ttk.Button(bottom, text="_", width=3, command=lambda: self.insert_text("_")).pack(side="left", padx=2)
        ttk.Button(bottom, text="⌫", width=4, command=self.backspace).pack(side="left", padx=2)
        ttk.Button(bottom, text="Очистить", command=self.clear).pack(side="left", padx=2)

    def current_entry(self):
        return self.target_entry_getter()

    def insert_text(self, text: str) -> None:
        entry = self.current_entry()
        if entry is None:
            return
        entry.insert("end", text)

    def backspace(self) -> None:
        entry = self.current_entry()
        if entry is None:
            return
        value = entry.get()
        if value:
            entry.delete(len(value) - 1, "end")

    def clear(self) -> None:
        entry = self.current_entry()
        if entry is None:
            return
        entry.delete(0, "end")