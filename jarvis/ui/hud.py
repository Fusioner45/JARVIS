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

        layout = QVBoxLayout(self)
        layout.addSpacing(450)
        self.label = QLabel("INITIALISATION...", self)
        self.label.setFont(QFont("Consolas", 12))
        self.label.setStyleSheet("color: #00f2ff; background: transparent;")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        # UI State
        self.angle = 0
        self.pulse = 0
        self.is_thinking = False
        self.target_text = ""
        self.current_text = ""

        # Pre-calculated points for performance
        self._center = QPointF(300, 250)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(20) # 50 FPS

    def animate(self):
        """Update animation states without logic overhead."""
        try:
            self.angle = (self.angle + (4 if self.is_thinking else 1)) % 360
            self.pulse += 0.1

            # Typewriter effect logic
            if self.target_text and len(self.current_text) < len(self.target_text):
                self.current_text += self.target_text[len(self.current_text)]
                if self.label:
                    self.label.setText(self.current_text)

            self.update()
        except Exception:
            pass

    def paintEvent(self, event):
        """Safe and efficient drawing code."""
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Use pre-calculated center
            center = self._center

            # 1. Glow Effect (Manual instead of shadow to prevent parameter errors)
            glow = QRadialGradient(center, 120)
            glow.setColorAt(0, QColor(0, 242, 255, 40))
            glow.setColorAt(1, Qt.GlobalColor.transparent)
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center, 120, 120)

            # 2. Main Arc Rings
            painter.setPen(QPen(QColor(0, 242, 255, 180), 2))
            painter.drawEllipse(center, 80, 80)

            painter.save()
            painter.translate(center)
            painter.rotate(self.angle)

            # Decorative rotating blocks
            block_color = QColor(0, 242, 255, 200) if not self.is_thinking else QColor(255, 50, 50, 200)
            painter.setPen(QPen(block_color, 2))
            for i in range(0, 360, 45):
                painter.drawRect(QRectF(85, -5, 15, 10))
                painter.rotate(45)
            painter.restore()

            # 3. Inner pulsing core
            pulse_val = abs(np.sin(self.pulse))
            core_radius = 40 + (10 * pulse_val)
            core_glow = QRadialGradient(center, core_radius)
            core_glow.setColorAt(0, QColor(0, 242, 255, int(150 * pulse_val)))
            core_glow.setColorAt(1, Qt.GlobalColor.transparent)
            painter.setBrush(QBrush(core_glow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(center, core_radius, core_radius)
        except Exception:
            pass

    def set_text(self, text):
        """Thread-safe text update."""
        if text != self.target_text:
            self.target_text = text
            self.current_text = ""
            if self.label:
                self.label.setText("")

    def set_thinking(self, thinking: bool):
        """Switch visual mode."""
        self.is_thinking = thinking

class JarvisSignals(QObject):
    transcription_received = pyqtSignal(str)
    thinking_state_changed = pyqtSignal(bool)
