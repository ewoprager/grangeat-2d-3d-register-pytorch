from qtpy.QtWidgets import QWidget, QVBoxLayout
from qtpy.QtCore import Qt
import napari


class FloatingWidget(QWidget):
    def __init__(self, title: str, widget: QWidget):
        super().__init__(napari.Viewer().window._qt_window)

        self.setWindowTitle(title)

        layout = QVBoxLayout()
        layout.addWidget(widget)
        self.setLayout(layout)

        # Optional: behave like a tool panel
        self.setWindowFlags(Qt.Window |  # make it a real window
                            Qt.Tool  # stays on top of parent, no taskbar entry
                            )
