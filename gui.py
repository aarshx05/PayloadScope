import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QPushButton, QScrollArea, QTabWidget, QHBoxLayout,
    QProgressBar, QSizePolicy
)
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import Qt
import joblib
import base64
import binascii
import urllib.parse
import re

# Matplotlib for charts
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ------------------------------------------------------------
# MODEL PATH (same as before)
# ------------------------------------------------------------
MODEL_PATH = r"./model_output/payload_classifier_tfidf_lr.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
classes = model.named_steps["clf"].classes_

THRESHOLD = 0.80


# ------------------------------------------------------------
# DECODING FUNCTIONS
# ------------------------------------------------------------
def try_base64_decode(s):
    try:
        s_bytes = s.encode()
        missing_padding = len(s_bytes) % 4
        if missing_padding:
            s_bytes += b"=" * (4 - missing_padding)
        decoded = base64.b64decode(s_bytes, validate=True)
        return decoded.decode("utf-8", errors="ignore")
    except Exception:
        return s


def try_url_decode(s):
    try:
        return urllib.parse.unquote(s)
    except Exception:
        return s


def try_hex_decode(s):
    try:
        s_clean = re.sub(r"(\\x|0x)", "", s)
        if len(s_clean) % 2 != 0:
            return s
        decoded = bytes.fromhex(s_clean)
        return decoded.decode("utf-8", errors="ignore")
    except Exception:
        return s


def recursive_decode(payload, max_depth=2):
    current = payload
    for _ in range(max_depth):
        prev = current
        current = try_base64_decode(current)
        current = try_url_decode(current)
        current = try_hex_decode(current)
        if current == prev:
            break
    return current


# ------------------------------------------------------------
# Matplotlib Canvas for Graphs
# ------------------------------------------------------------
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(4, 3), facecolor="#1b1f23")
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.ax.set_facecolor("#25292e")
        self.fig.tight_layout()


# ------------------------------------------------------------
# Main GUI
# ------------------------------------------------------------
class PayloadGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Payload Classifier Dashboard")
        self.setMinimumSize(1150, 720)
        self.apply_theme()
        self.build_ui()

    # --------------------------------------------------------
    # Theme B – Professional Defender-style
    # --------------------------------------------------------
    def apply_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#1b1f23"))        # background
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#f5f5f5"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#25292e"))          # text areas
        palette.setColor(QPalette.ColorRole.Text, QColor("#f5f5f5"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#2d3439"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
        self.setPalette(palette)

        self.setStyleSheet("""
            QWidget {
                font-family: "Segoe UI";
                color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 0px;
            }
            QTabBar::tab {
                background: #2d3439;
                color: #d7d7d7;
                padding: 8px 16px;
                border-radius: 10px;
                margin: 0 4px;
            }
            QTabBar::tab:selected {
                background: #0078d4;
                color: #ffffff;
            }
            QTextEdit {
                border-radius: 10px;
                border: 1px solid #3a4147;
                padding: 10px;
                background-color: #25292e;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                padding: 12px;
                border-radius: 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1084e3;
            }
        """)

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------
    def build_ui(self):
        main_layout = QHBoxLayout()

        # LEFT PANEL
        left_panel = QVBoxLayout()

        header = QLabel("Payload Analyzer")
        header.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        sub_header = QLabel("Machine-learning powered payload classification")
        sub_header.setFont(QFont("Segoe UI", 10))
        sub_header.setStyleSheet("color: #9ba3ab;")

        left_panel.addWidget(header)
        left_panel.addWidget(sub_header)

        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("Paste or type a payload here...")
        self.input_box.setFont(QFont("Consolas", 11))
        self.input_box.setMinimumHeight(260)
        left_panel.addWidget(self.input_box)

        analyze_btn = QPushButton("Analyze Payload")
        analyze_btn.setFont(QFont("Segoe UI Semibold", 13))
        analyze_btn.clicked.connect(self.analyze_payload)
        left_panel.addWidget(analyze_btn)

        left_panel.addStretch()
        main_layout.addLayout(left_panel, 38)

        # RIGHT PANEL – TABS
        self.tabs = QTabWidget()

        # Tab 1: Summary
        self.summary_tab = QTextEdit()
        self.summary_tab.setReadOnly(True)
        self.summary_tab.setFont(QFont("Consolas", 11))
        self.tabs.addTab(self.summary_tab, "Results Summary")

        # Tab 2: Probability Graph
        self.graph_canvas = MplCanvas()
        self.tabs.addTab(self.graph_canvas, "Probability Graph")

        # Tab 3: Threat Overview (progress bar style)
        threat_tab_layout = QVBoxLayout()
        threat_tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        threat_title = QLabel("Overall Threat Assessment")
        threat_title.setFont(QFont("Segoe UI Semibold", 16))
        threat_tab_layout.addWidget(threat_title)

        self.threat_desc = QLabel("Run an analysis to view threat level.")
        self.threat_desc.setFont(QFont("Segoe UI", 10))
        self.threat_desc.setStyleSheet("color: #9ba3ab;")
        threat_tab_layout.addWidget(self.threat_desc)

        self.threat_bar = QProgressBar()
        self.threat_bar.setRange(0, 100)
        self.threat_bar.setValue(0)
        self.threat_bar.setTextVisible(True)
        self.threat_bar.setFormat("Threat level: %p%")
        self.threat_bar.setMinimumHeight(40)
        self.threat_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.set_threat_bar_style("low")
        threat_tab_layout.addWidget(self.threat_bar)

        threat_info = QLabel(
            "0–39%  ·  Low risk\n"
            "40–69% ·  Elevated risk\n"
            "70–100% ·  High risk"
        )
        threat_info.setFont(QFont("Segoe UI", 9))
        threat_info.setStyleSheet("color: #9ba3ab;")
        threat_tab_layout.addWidget(threat_info)

        threat_container = QWidget()
        threat_container.setLayout(threat_tab_layout)
        self.tabs.addTab(threat_container, "Threat Overview")

        main_layout.addWidget(self.tabs, 62)
        self.setLayout(main_layout)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def set_threat_bar_style(self, level: str):
        """Adjusts bar colour based on risk level; avoids harsh red."""
        if level == "low":
            chunk_color = "#107C10"   # Defender green
        elif level == "medium":
            chunk_color = "#FFB900"   # amber
        else:  # high
            chunk_color = "#F7630C"   # warm orange, not bright red

        self.threat_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #3a4147;
                border-radius: 8px;
                background-color: #25292e;
                text-align: center;
                color: #f5f5f5;
            }}
            QProgressBar::chunk {{
                border-radius: 8px;
                background-color: {chunk_color};
            }}
        """)

    # --------------------------------------------------------
    # Core: Run model & update views
    # --------------------------------------------------------
    def analyze_payload(self):
        payload = self.input_box.toPlainText().strip()
        if not payload:
            self.summary_tab.setText("⚠ No payload provided.")
            self.threat_desc.setText("No payload analyzed yet.")
            self.threat_bar.setValue(0)
            self.set_threat_bar_style("low")
            return

        decoded_payload = recursive_decode(payload)

        prob = model.predict_proba([decoded_payload])[0]
        pred_index = prob.argmax()
        pred_label = classes[pred_index]
        pred_conf = float(prob[pred_index])

        if pred_conf < THRESHOLD:
            pred_label_display = "uncertain / likely benign"
        else:
            pred_label_display = pred_label

        prob_lines = [
            f"{cls}: {round(float(p), 4)}"
            for cls, p in zip(classes, prob)
        ]
        prob_text = "\n".join(prob_lines)

        summary = f"""
======================
     ANALYSIS SUMMARY
======================

Payload:
{payload}

Decoded Payload:
{decoded_payload}

Prediction:  {pred_label_display}
Confidence:  {round(pred_conf, 4)}

----------------------
Class Probabilities:
{prob_text}
"""
        self.summary_tab.setText(summary)

        # Update probability graph
        self.update_graph(prob)

        # Update threat UI
        self.update_threat_view(pred_conf)

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------
    def update_graph(self, prob):
        ax = self.graph_canvas.ax
        ax.clear()

        x = list(range(len(classes)))
        ax.bar(x, prob, width=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=15, ha="right")

        ax.set_ylim(0, 1.05)
        ax.set_title("Prediction Probabilities", color="#f5f5f5", pad=10)
        ax.tick_params(colors="#d0d0d0")
        for spine in ax.spines.values():
            spine.set_color("#4b5258")

        ax.yaxis.label.set_color("#d0d0d0")
        ax.xaxis.label.set_color("#d0d0d0")
        ax.set_ylabel("Probability")

        self.graph_canvas.draw()

    # --------------------------------------------------------
    # Threat bar logic
    # --------------------------------------------------------
    def update_threat_view(self, confidence: float):
        score = round(confidence * 100)

        if score < 40:
            level = "low"
            desc = "Low risk detected based on current model confidence."
        elif score < 70:
            level = "medium"
            desc = "Elevated risk. Review payload and context carefully."
        else:
            level = "high"
            desc = "High risk. Treat this payload as potentially dangerous."

        self.set_threat_bar_style(level)
        self.threat_bar.setValue(score)
        self.threat_desc.setText(desc)


# ------------------------------------------------------------
# RUN APP
# ------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PayloadGUI()
    gui.show()
    sys.exit(app.exec())
