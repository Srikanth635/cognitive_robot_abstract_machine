from typing import Optional, List, Union
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QComboBox,
    QListWidget,
    QAbstractItemView,
    QVBoxLayout,
    QLabel,
    QPushButton,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QIcon
from superqt import QRangeSlider, QDoubleRangeSlider

from random_events.variable import Variable, Continuous, Symbolic, Integer
from random_events.product_algebra import SimpleEvent, Event, VariableMap
from random_events.interval import closed, Interval, SimpleInterval, Bound
from random_events.set import Set, SetElement


class VariableConstraintWidget(QWidget):
    """
    A widget that allows selecting a variable and defining its constraints.
    """

    changed = Signal()

    def __init__(
        self,
        variables: List[Variable],
        priors: Optional[VariableMap] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.variables = sorted(variables, key=lambda v: v.name)
        self.priors = priors
        self.init_ui()

    def init_ui(self):
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Variable selector
        self.variable_combo = QComboBox()
        self.variable_combo.addItem("Select Variable...", None)
        for var in self.variables:
            self.variable_combo.addItem(var.name, var)

        self.variable_combo.currentIndexChanged.connect(self.on_variable_changed)
        self.layout.addWidget(self.variable_combo, 1)

        # Container for constraint input
        self.constraint_container = QWidget()
        self.constraint_layout = QVBoxLayout(self.constraint_container)
        self.constraint_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.constraint_container, 2)

        self.constraint_widget = None

    def on_variable_changed(self):
        # Clear previous constraint widgets
        while self.constraint_layout.count():
            item = self.constraint_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                # Could be a layout
                sub_layout = item.layout()
                if sub_layout:
                    # Clear sub_layout
                    while sub_layout.count():
                        sub_item = sub_layout.takeAt(0)
                        sub_widget = sub_item.widget()
                        if sub_widget:
                            sub_widget.deleteLater()
        self.constraint_widget = None

        variable = self.variable_combo.currentData()
        if variable:
            self.create_constraint_widget(variable)

        self.changed.emit()

    def create_constraint_widget(self, variable: Variable):
        if variable.is_numeric:
            # Range Slider for Continuous or Integer
            # Default to domain
            try:
                mini = variable.domain.simple_sets[0].lower
                maxi = variable.domain.simple_sets[-1].upper
            except (AttributeError, IndexError):
                # Fallback for composite sets
                try:
                    mini = variable.domain.simple_sets[0].simple_sets[0].lower
                    maxi = variable.domain.simple_sets[-1].simple_sets[-1].upper
                except (AttributeError, IndexError):
                    mini, maxi = 0, 1

            # Try to get from priors (support)
            if self.priors and variable in self.priors:
                try:
                    support = self.priors[variable].support
                    if support.simple_sets:
                        mini = support.simple_sets[0][variable].simple_sets[0].lower
                        maxi = support.simple_sets[-1][variable].simple_sets[-1].upper
                except Exception:
                    pass

            # Handle infinity
            if mini == float("-inf"):
                mini = -100.0
            if maxi == float("inf"):
                maxi = 100.0

            # Handle equality
            if mini == maxi:
                mini -= 1.0
                maxi += 1.0

            self.slider_min = mini
            self.slider_max = maxi

            # Current value label
            self.value_label = QLabel()
            self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.constraint_layout.addWidget(self.value_label)

            # Container for multiple sliders
            self.sliders_container = QWidget()
            self.sliders_layout = QVBoxLayout(self.sliders_container)
            self.sliders_layout.setContentsMargins(0, 0, 0, 0)
            self.constraint_layout.addWidget(self.sliders_container)
            self.constraint_widget = self.sliders_container
            self.sliders: List[QRangeSlider] = []

            self.add_slider(variable, (mini, maxi))

            # Marks
            self.marks_layout = QHBoxLayout()
            size = maxi - mini
            if size > 0:
                import numpy as np

                steps = [x for x in np.arange(mini, maxi, size / 5)] + [maxi]
                for step in steps:
                    text = (
                        f"{int(round(step))}"
                        if isinstance(variable, Integer)
                        else f"{step:.2f}"
                    )
                    mark_label = QLabel(text)
                    mark_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    mark_label.setStyleSheet("font-size: 8pt; color: gray;")
                    self.marks_layout.addWidget(mark_label)
            self.constraint_layout.addLayout(self.marks_layout)

            # Add/Remove buttons
            self.buttons_layout = QHBoxLayout()
            self.add_button = QPushButton()
            self.add_button.setIcon(QIcon("icon:/primary/checklist.svg"))
            self.add_button.setFixedWidth(30)
            self.add_button.clicked.connect(lambda: self.on_add_range(variable))
            self.buttons_layout.addWidget(self.add_button)

            self.remove_button = QPushButton()
            self.remove_button.setIcon(QIcon("icon:/primary/close.svg"))
            self.remove_button.setFixedWidth(30)
            self.remove_button.clicked.connect(lambda: self.on_remove_range(variable))
            self.buttons_layout.addWidget(self.remove_button)
            self.buttons_layout.addStretch()
            self.constraint_layout.addLayout(self.buttons_layout)
        else:
            # List Widget for Symbolic (multi-selection)
            list_widget = QListWidget()
            list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

            # variable.domain is a Set for Symbolic variables
            for element in variable.domain.all_elements:
                list_widget.addItem(str(element))
                item = list_widget.item(list_widget.count() - 1)
                item.setSelected(True)  # Default select all

            list_widget.itemSelectionChanged.connect(self.changed.emit)
            # Set a reasonable height for the list
            list_widget.setMaximumHeight(100)
            self.constraint_widget = list_widget

        self.constraint_layout.addWidget(self.constraint_widget)

    def add_slider(self, variable: Variable, values: tuple[float, float]):
        if isinstance(variable, Continuous):
            slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(self.slider_min)
            slider.setMaximum(self.slider_max)
            slider.setValue(values)
        else:
            slider = QRangeSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(self.slider_min))
            slider.setMaximum(int(self.slider_max))
            slider.setValue((int(values[0]), int(values[1])))

        slider.setStyleSheet(
            "QRangeSlider, QDoubleRangeSlider { qproperty-barColor: #1de9b6; min-height: 25px; }"
            "QRangeSlider::handle, QDoubleRangeSlider::handle { background-color: #1de9b6; border: 2px solid white; width: 16px; height: 16px; border-radius: 0px; image: none; }"
            "QRangeSlider::groove, QDoubleRangeSlider::groove { background-color: #31363b; height: 8px; border-radius: 4px; }"
        )

        slider.valueChanged.connect(lambda _: self.update_ranges_label(variable))
        slider.valueChanged.connect(lambda _: self.changed.emit())

        self.sliders.append(slider)
        self.sliders_layout.addWidget(slider)
        self.update_ranges_label(variable)

    def update_ranges_label(self, variable: Variable):
        ranges = []
        for slider in self.sliders:
            val = slider.value()
            if isinstance(variable, Integer):
                ranges.append(f"[{int(val[0])}, {int(val[1])}]")
            else:
                ranges.append(f"[{val[0]:.2f}, {val[1]:.2f}]")
        self.value_label.setText("Range: " + ", ".join(ranges))

    def on_add_range(self, variable: Variable):
        if not self.sliders:
            self.add_slider(variable, (self.slider_min, self.slider_max))
            return

        last_slider = self.sliders[-1]
        _, last_high = last_slider.value()

        remaining = self.slider_max - last_high
        if remaining > (self.slider_max - self.slider_min) * 0.1:
            new_min = last_high + remaining * 0.05
            new_max = last_high + remaining * 0.15
        else:
            # Add at the end if not enough space
            new_max = self.slider_max
            new_min = self.slider_max - (self.slider_max - self.slider_min) * 0.05

        self.add_slider(variable, (new_min, new_max))
        self.changed.emit()

    def on_remove_range(self, variable: Variable):
        if len(self.sliders) > 1:
            slider = self.sliders.pop()
            slider.deleteLater()
            self.update_ranges_label(variable)
            self.changed.emit()

    def get_constraint(self) -> Optional[tuple[Variable, Union[Interval, Set]]]:
        variable = self.variable_combo.currentData()
        if not variable:
            return None

        if isinstance(variable, (Continuous, Integer)):
            intervals = []
            for slider in self.sliders:
                vals = list(slider.value())

                intervals.append(
                    SimpleInterval(vals[0], vals[1], Bound.CLOSED, Bound.CLOSED)
                )

            if not intervals:
                return variable, variable.domain

            return variable, Interval(*intervals)
        elif isinstance(variable, Symbolic):
            list_widget: QListWidget = self.constraint_widget
            selected_items = list_widget.selectedItems()
            selected_values = [item.text() for item in selected_items]
            # Create a Set from selected values
            # Need to be careful with types here, selected_values are strings
            # and the original domain might have different types.
            # For now, assuming strings or using match.
            all_elements = variable.domain.all_elements
            matched_elements = [e for e in all_elements if str(e) in selected_values]

            if not matched_elements:
                return variable, Set()  # Empty set

            return variable, Set(
                *[SetElement(e, all_elements) for e in matched_elements]
            )

        return None

    def set_constraint(self, variable: Variable, constraint: Union[Interval, Set]):
        """
        Sets the variable and its constraint programmatically.
        """
        # Select the variable in the combo box
        for i in range(self.variable_combo.count()):
            data = self.variable_combo.itemData(i)
            if data is not None and data == variable:
                self.variable_combo.setCurrentIndex(i)
                break

        # Now the constraint widget should be created via on_variable_changed
        if isinstance(variable, (Continuous, Integer)):
            # Clear existing sliders
            for slider in self.sliders:
                slider.deleteLater()
            self.sliders.clear()

            # constraint is an Interval (composite set)
            # Sort simple sets by lower bound
            sorted_simple_sets = sorted(constraint.simple_sets, key=lambda s: s.lower)
            for simple_set in sorted_simple_sets:
                self.add_slider(variable, (simple_set.lower, simple_set.upper))

        elif isinstance(variable, Symbolic):
            if not self.constraint_widget:
                return
            list_widget: QListWidget = self.constraint_widget
            # constraint is a Set
            selected_str_values = [str(e.element) for e in constraint.simple_sets]
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                item.setSelected(item.text() in selected_str_values)
