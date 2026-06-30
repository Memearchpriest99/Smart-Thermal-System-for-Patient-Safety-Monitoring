import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy, QShortcut
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence


STATES = {
    "idle":  {
        "bg":      "lightgray",
        "text":    "",
        "color":   "#222",
    },
    "touch": {
        "bg":      "#9b59b6",
        "text":    "Touch Detected",
        "color":   "white",
    },
    "fire":  {
        "bg":      "#e74c3c",
        "text":    "Fire Detected",
        "color":   "white",
    },
    "human": {
        "bg":      "#27ae60",
        "text":    "Human Detected",
        "color":   "white",
    },
}


class WardWatcher(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        title = QLabel("Ward Watcher")
        title.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        title.setStyleSheet("font-size: 20px; color: blue;")
        layout.addWidget(title, alignment=Qt.AlignHCenter)

        # Room display frame
        self.room_display_rect = QFrame()
        self.room_display_rect.setObjectName("roomDisplayRect")
        self.room_display_rect.setMinimumSize(800, 500)
        self.room_display_rect.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        frame_layout = QVBoxLayout(self.room_display_rect)
        frame_layout.setContentsMargins(16, 14, 16, 14)
        frame_layout.setSpacing(8)

        self.header_label = QLabel("Room 1 Display", self.room_display_rect)
        self.header_label.setWordWrap(True)
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("font-size: 32px; font-weight: bold; color: black;")
        frame_layout.addWidget(self.header_label, alignment=Qt.AlignHCenter | Qt.AlignTop)

        self.status_label = QLabel("", self.room_display_rect)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        frame_layout.addWidget(self.status_label)

        layout.addWidget(self.room_display_rect, 1)

        # Footer with state buttons + exit
        footer = QHBoxLayout()
        footer.setSpacing(10)

        btn_specs = [
            ("Idle",           "idle",  "#888888"),
            ("Touch Detected", "touch", "#9b59b6"),
            ("Fire Detected",  "fire",  "#e74c3c"),
            ("Human Detected", "human", "#27ae60"),
        ]

        for label, state_key, color in btn_specs:
            btn = QPushButton(label)
            btn.setFixedHeight(44)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border-radius: 6px;
                    font-size: 14px;
                    padding: 0 16px;
                }}
                QPushButton:hover {{ opacity: 0.85; }}
            """)
            btn.clicked.connect(lambda _, s=state_key: self._set_state(s))
            footer.addWidget(btn)

        footer.addStretch(1)

        exit_button = QPushButton("Exit")
        exit_button.setFixedHeight(44)
        exit_button.setStyleSheet("color: blue; font-size: 14px; padding: 0 16px;")
        exit_button.clicked.connect(self.close)
        footer.addWidget(exit_button, 0, Qt.AlignRight)

        layout.addLayout(footer)

        QShortcut(QKeySequence(Qt.Key_Escape), self, self.close)

        self._set_state("idle")

    def _set_state(self, state_key: str):
        s = STATES[state_key]
        bg = s["bg"]

        self.room_display_rect.setStyleSheet(f"""
        #roomDisplayRect {{
            background-color: {bg};
            border: 1px solid #b0b0b0;
            border-radius: 24px;
        }}
        #roomDisplayRect QLabel {{
            background: transparent;
            border: none;
            padding: 0;
        }}
        """)

        if s["text"]:
            self.status_label.setText(s["text"])
            self.status_label.setStyleSheet(
                f"font-size: 64px; font-weight: bold; color: {s['color']};"
            )
            self.status_label.show()
            self.header_label.setStyleSheet(
                "font-size: 32px; font-weight: bold; color: white;"
            )
        else:
            self.status_label.clear()
            self.status_label.hide()
            self.header_label.setStyleSheet(
                "font-size: 32px; font-weight: bold; color: black;"
            )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WardWatcher()
    window.setWindowTitle('Ward Watcher')
    window.showFullScreen()
    sys.exit(app.exec_())
