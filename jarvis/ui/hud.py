import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPointF, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QRadialGradient, QPen, QBrush

class ArcReactor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(600, 700)

        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 20, 40, 0.4);
                border-radius: 30px;
                border: 2px solid rgba(0, 242, 255, 0.2);
            }
        """)

        layout = QVBoxLayout(self)
        layout.addSpacing(450)

        self.label = QLabel("INITIALISATION...", self)
        font = QFont("OCR A Extended", 12)
        if font.family() != "OCR A Extended": font = QFont("Consolas", 12)

        self.label.setFont(font)
        self.label.setStyleSheet("color: #00f2ff; background: transparent; border: none; padding: 20px;")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.label)

        self.angle_outer = 0
        self.pulse_inner = 0
        self.is_thinking = False
        self.target_text = ""
        self.current_text = ""

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(20)

    def animate(self):
        try:
            self.angle_outer = (self.angle_outer + (4 if self.is_thinking else 1)) % 360
            self.pulse_inner += 0.15 if self.is_thinking else 0.05
            if self.target_text:
                if len(self.current_text) < len(self.target_text):
                    self.current_text += self.target_text[len(self.current_text)]
                    self.label.setText(self.current_text)
            self.update()
        except: pass

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            center = QPointF(self.width() / 2, 250)

            # Glow central
            glow = QRadialGradient(center, 300)
            glow.setColorAt(0, QColor(0, 242, 255, 30))
            glow.setColorAt(1, Qt.GlobalColor.transparent)
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center, 300, 300)

            painter.save()
            painter.translate(center)
            painter.rotate(self.angle_outer)
            pen_outer = QPen(QColor(0, 242, 255, 100), 2)
            pen_outer.setDashPattern([10, 10])
            painter.setPen(pen_outer)
            painter.drawEllipse(QRectF(-120, -120, 240, 240))
            painter.restore()

            inner_scale = 1.0 + 0.1 * np.sin(self.pulse_inner)
            inner_radius = 60 * inner_scale
            color_inner = QColor(0, 242, 255, 220) if not self.is_thinking else QColor(255, 50, 50, 220)
            painter.setPen(QPen(color_inner, 3))
            painter.drawEllipse(center, inner_radius, inner_radius)
        except: pass

    def set_text(self, text):
        if text != self.target_text:
            self.target_text = text
            self.current_text = ""
            self.label.setText("")

    def set_thinking(self, thinking: bool):
        self.is_thinking = thinking

class JarvisSignals(QObject):
    transcription_received = pyqtSignal(str)
    thinking_state_changed = pyqtSignal(bool)
    state_changed = pyqtSignal(object)
