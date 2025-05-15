import sys
import json
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                             QLabel, QLineEdit, QPushButton, QGridLayout, QListWidget,
                             QComboBox, QTextBrowser, QMessageBox, QStatusBar, QHBoxLayout, QFrame,
                             QFileDialog, QProgressBar, QCheckBox, QTabWidget, QListWidgetItem, QSizePolicy, QGroupBox, QAbstractItemView)  # Importação corrigida
from PyQt6.QtGui import QFont, QDoubleValidator, QColor
from PyQt6.QtCore import Qt

class PersistenceManager:
    @staticmethod
    def get_data_path():
        # Create a data directory if it doesn't exist
        data_dir = os.path.join(os.path.expanduser('~'), '.measurement_app')
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    @staticmethod
    def save_data(filename, data):
        filepath = os.path.join(PersistenceManager.get_data_path(), filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_data(filename, default=None):
        filepath = os.path.join(PersistenceManager.get_data_path(), filename)
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default or {}

class StyleHelper:
    @staticmethod
    def get_stylesheet():
        return """
        /* Global Styling */
        QWidget {
            font-size: 14px;
            background-color: #f4f4f4;
            color: #000000;
        }
        
        /* Tab Widget Styling */
        QTabWidget::pane {
            border: 2px solid #e0e0e0;
            background-color: white;
        }
        
        QTabBar::tab {
            background-color: #f0f0f0;
            color: #000000;
            padding: 10px 20px;
            margin-right: 5px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            min-width: 120px;
        }
        
        QTabBar::tab:selected {
            background-color: #e0e0e0;
            font-weight: bold;
            color: #000000;
        }
        
        /* Input Fields */
        QLineEdit {
            padding: 10px;
            border: 2px solid #ccc;
            border-radius: 8px;
            background-color: white;
            min-height: 40px;
            color: #000000;
        }

        QComboBox {
            color: #000000;
            background-color: white;
            padding: 10px;
            border: 2px solid #ccc;
            border-radius: 8px;
            min-height: 40px;
        }

        QComboBox::drop-down {
            border: none;
            color: #000000;
        }

        QComboBox::down-arrow {
            color: #000000;
        }

        QComboBox QAbstractItemView {
            background-color: white;
            color: #000000;
        }

        QComboBox::item {
            color: #000000;
            background-color: white;
        }

        QComboBox::item:selected {
            color: #000000;
            background-color: #e0e0e0;
        }
        
        QLineEdit:focus, QComboBox:focus {
            border-color: #4A90E2;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #4A90E2;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            min-height: 40px;
        }
        
        QPushButton:hover {
            background-color: #357ABD;
        }
        
        QPushButton:pressed {
            background-color: #2B6AA3;
        }
        
        /* Labels */
        QLabel {
            color: #000000;
            font-weight: bold;
            margin-bottom: 5px;
        }

        /* List Widgets */
        QListWidget {
            background-color: white;
            color: #000000;
            border: 1px solid #ccc;
            border-radius: 4px;
        }

        QListWidget::item {
            color: #000000;
            padding: 5px;
        }

        QListWidget::item:selected {
            background-color: #e0e0e0;
            color: #000000;
        }

        /* Text Browser */
        QTextBrowser {
            background-color: white;
            color: #000000;
            border: 1px solid #ccc;
            border-radius: 4px;
        }

        /* Group Box */
        QGroupBox {
            color: #000000;
            border: 1px solid #ccc;
            margin-top: 1em;
            padding-top: 1em;
        }

        QGroupBox::title {
            color: #000000;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }

        /* Checkboxes */
        QCheckBox {
            color: #000000;
        }

        QCheckBox::indicator {
            width: 13px;
            height: 13px;
        }

        /* Spinboxes */
        QSpinBox {
            color: #000000;
            background-color: white;
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }

        /* Status Bar */
        QStatusBar {
            color: #000000;
            background-color: #f4f4f4;
        }

        /* Menu and Menu Items */
        QMenuBar {
            background-color: #f4f4f4;
            color: #000000;
        }

        QMenuBar::item {
            color: #000000;
        }

        QMenu {
            background-color: white;
            color: #000000;
        }

        QMenu::item {
            color: #000000;
        }

        QMenu::item:selected {
            background-color: #e0e0e0;
            color: #000000;
        }
        """

class EnhancedSection(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 10px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Section Title
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-size: 16px;
            color: #4A90E2;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        """)
        layout.addWidget(title_label)
        
        self.content_layout = QGridLayout()
        layout.addLayout(self.content_layout)
        
        self.setLayout(layout)

    def add_field(self, label_text, input_widget):
        label = QLabel(label_text)
        row = self.content_layout.rowCount()
        self.content_layout.addWidget(label, row, 0)
        self.content_layout.addWidget(input_widget, row + 1, 0)
        return input_widget

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Measurement System")
        self.setGeometry(100, 100, 1000, 800)
        
        # Set global stylesheet
        self.setStyleSheet(StyleHelper.get_stylesheet())
        
        # Main container
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Tab Widget
        self.tabs = QTabWidget()
        
        # Create and add tabs
        self.tube_setup_tab = self.create_tube_setup_tab()
        self.samples_tab = self.create_samples_tab()
        self.test_conditions_tab = self.create_test_conditions_tab()
        self.warnings_tab = self.create_warnings_tab()
        self.documentation_tab = self.create_documentation_tab()
        self.measurements_tab = self.create_measurements_tab()
        self.results_tab = QResultsTab()
        self.export_tab = QReportViewTab()
        
        self.tabs.addTab(self.tube_setup_tab, "Tube Setup")
        self.tabs.addTab(self.samples_tab, "Samples")
        self.tabs.addTab(self.test_conditions_tab, "Test Conditions")
        self.tabs.addTab(self.warnings_tab, "Warnings")
        self.tabs.addTab(self.measurements_tab, "Measuments")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.export_tab, "Report")
        self.tabs.addTab(self.documentation_tab, "Documentation")
        
        main_layout.addWidget(self.tabs)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Load existing data
        self.load_existing_data()

    def load_existing_data(self):
        # Load Tube Setup
        tube_setup_data = PersistenceManager.load_data('tube_setup.json')
        if tube_setup_data:
            self.tube_setup_tab.load_data(tube_setup_data)
        
        # Load Samples
        samples_data = PersistenceManager.load_data('samples.json')
        if samples_data:
            self.samples_tab.load_data(samples_data)
        
        # Load Test Conditions
        test_conditions_data = PersistenceManager.load_data('test_conditions.json')
        if test_conditions_data:
            self.test_conditions_tab.load_data(test_conditions_data)

    def create_tube_setup_tab(self):
        tab = QTubeSetupTab()
        return tab

    def create_samples_tab(self):
        tab = QSamplesTab()
        return tab

    def create_test_conditions_tab(self):
        tab = QTestConditionsTab()
        return tab

    def create_warnings_tab(self):
        tab = QWarningsTab()
        return tab

    def create_documentation_tab(self):
        tab = QDocumentationTab()
        return tab
    
    def create_measurements_tab(self):
        tab = QMeasurementsTab()
        return tab
    
class QReportViewTab(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Client Information Group
        client_group = QGroupBox("Client Information")
        client_layout = QGridLayout()
        
        # Create input fields for client info
        self.client_name = QLineEdit()
        self.company = QLineEdit()
        self.address = QLineEdit()
        self.purchase_order = QLineEdit()
        self.report_number = QLineEdit()
        self.test_date = QLineEdit()
        self.test_requester = QLineEdit()
        self.test_executor = QLineEdit()
        self.test_supervisor = QLineEdit()
        
        # Add fields to layout
        row = 0
        for label, widget in [
            ("Client Name*:", self.client_name),
            ("Company:", self.company),
            ("Address:", self.address),
            ("Purchase Order:", self.purchase_order),
            ("Report Number*:", self.report_number),
            ("Test Date*:", self.test_date),
            ("Test Requester:", self.test_requester),
            ("Test Executor*:", self.test_executor),
            ("Test Supervisor:", self.test_supervisor)
        ]:
            client_layout.addWidget(QLabel(label), row, 0)
            client_layout.addWidget(widget, row, 1)
            row += 1
            
        client_group.setLayout(client_layout)
        
        # Sample Information Group
        sample_group = QGroupBox("Sample Information")
        sample_layout = QGridLayout()
        
        # Create input fields for sample info
        self.product_name = QLineEdit()
        self.manufacturer = QLineEdit()
        self.description = QLineEdit()
        self.layers = QLineEdit()
        self.conditions = QLineEdit()
        self.mounting = QLineEdit()
        
        # Add fields to layout
        row = 0
        for label, widget in [
            ("Product Name:", self.product_name),
            ("Manufacturer:", self.manufacturer),
            ("Description:", self.description),
            ("Layers:", self.layers),
            ("Conditions:", self.conditions),
            ("Mounting:", self.mounting)
        ]:
            sample_layout.addWidget(QLabel(label), row, 0)
            sample_layout.addWidget(widget, row, 1)
            row += 1
            
        sample_group.setLayout(sample_layout)
        
        # Add groups to main layout
        layout.addWidget(client_group)
        layout.addWidget(sample_group)
        
        # Add note about required fields
        note = QLabel("* Required fields")
        note.setStyleSheet("color: red;")
        layout.addWidget(note)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.preview_button = QPushButton("Preview Report")
        self.preview_button.clicked.connect(self.handle_preview)
        
        self.export_button = QPushButton("Export Report")
        self.export_button.clicked.connect(self.handle_export)
        
        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.export_button)
        
        layout.addLayout(button_layout)
        
        # Add status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)

    def set_controller(self, controller):
        self.controller = controller

    def get_client_info(self) -> dict:
        """Get client information from input fields."""
        return {
            'name': self.client_name.text(),
            'company': self.company.text(),
            'address': self.address.text(),
            'purchase_order': self.purchase_order.text(),
            'report_number': self.report_number.text(),
            'test_date': self.test_date.text(),
            'test_requester': self.test_requester.text(),
            'test_executor': self.test_executor.text(),
            'test_supervisor': self.test_supervisor.text()
        }

    def get_sample_info(self) -> dict:
        """Get sample information from input fields."""
        return {
            'product_name': self.product_name.text(),
            'manufacturer': self.manufacturer.text(),
            'description': self.description.text(),
            'layers': self.layers.text(),
            'conditions': self.conditions.text(),
            'mounting': self.mounting.text()
        }

    def set_data(self, client_info: dict, sample_info: dict):
        """Set data in input fields."""
        # Set client info
        self.client_name.setText(client_info.get('name', ''))
        self.company.setText(client_info.get('company', ''))
        self.address.setText(client_info.get('address', ''))
        self.purchase_order.setText(client_info.get('purchase_order', ''))
        self.report_number.setText(client_info.get('report_number', ''))
        self.test_date.setText(str(client_info.get('test_date', '')))
        self.test_requester.setText(client_info.get('test_requester', ''))
        self.test_executor.setText(client_info.get('test_executor', ''))
        self.test_supervisor.setText(client_info.get('test_supervisor', ''))
        
        # Set sample info
        self.product_name.setText(sample_info.get('product_name', ''))
        self.manufacturer.setText(sample_info.get('manufacturer', ''))
        self.description.setText(sample_info.get('description', ''))
        self.layers.setText(sample_info.get('layers', ''))
        self.conditions.setText(sample_info.get('conditions', ''))
        self.mounting.setText(sample_info.get('mounting', ''))

    def update_status(self, message: str):
        """Update status message."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: green;")

    def show_error(self, message: str):
        """Show error message."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: red;")

    def handle_preview(self):
        """Handle preview button click."""
        if self.controller:
            self.controller.handle_preview()

    def handle_export(self):
        """Handle export button click."""
        if self.controller:
            self.controller.handle_export()

class QMeasurementsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.measurements_tab_control = QTabWidget()

        self.measure_tab = QMeasureTab()
        self.processing_tab = QProcessingTab()

        self.measurements_tab_control.addTab(self.measure_tab, "Measure")
        self.measurements_tab_control.addTab(self.processing_tab, "Processing")

        layout.addWidget(self.measurements_tab_control)
        self.setLayout(layout)

class QMeasureTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.controller = None

        measure_control_section = EnhancedSection("Measurements Control")
        measure_control_layout = QVBoxLayout()

        self.amostras_listbox = QListWidget()
        self.amostras_listbox.addItem("Amostra_x")
        self.amostras_listbox.addItem("Amostra_y")
        self.amostras_listbox.setStyleSheet("color : black")
        measure_control_layout.addWidget(QLabel("Samples List"))
        measure_control_layout.addWidget(self.amostras_listbox)

        self.start_measurement_button = QPushButton("Measure")
        measure_control_layout.addWidget(self.start_measurement_button)

        self.resultados_listbox = QListWidget()
        self.resultados_listbox.addItem("Resultado_x")
        self.resultados_listbox.setStyleSheet("color : black")
        measure_control_layout.addWidget(QLabel("Measurements Results"))
        measure_control_layout.addWidget(self.resultados_listbox)

        measure_control_section.content_layout.addLayout(measure_control_layout, 0, 0)
        layout.addWidget(measure_control_section)
        self.setLayout(layout)

        self.start_measurement_button.clicked.connect(self.start_measurement)

    def start_measurement(self):
        """Informa ao controller que o botão de medição foi pressionado."""
        print("buttom clicked")
        selected_item = self.amostras_listbox.currentItem()
        if selected_item:
            sample_name = selected_item.text()
            self.controller.start_measurement(sample_name)

    def set_controller(self, controller):
        self.controller = controller

class QProcessingTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        post_processing_section = EnhancedSection("Post Processing")
        
        # Main horizontal layout for better space utilization
        main_layout = QHBoxLayout()
        
        # Left panel - Measurements list
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Measurements List"))
        
        self.measurements_checklist = QListWidget()
        self.measurements_checklist.setStyleSheet("color: #333; background-color: white; border: 1px solid #ccc;")
        self.measurements_checklist.setMinimumHeight(200)
        self.measurements_checklist.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        left_panel.addWidget(self.measurements_checklist)
        
        # Select/Deselect All buttons
        buttons_layout = QHBoxLayout()
        self.select_all_button = QPushButton("Select All")
        self.deselect_all_button = QPushButton("Deselect All")
        buttons_layout.addWidget(self.select_all_button)
        buttons_layout.addWidget(self.deselect_all_button)
        left_panel.addLayout(buttons_layout)
        
        # Right panel - Operations and preview
        right_panel = QVBoxLayout()
        
        # Operation selection group
        operations_group = QGroupBox("Processing Operations")
        operations_layout = QVBoxLayout()
        
        # Add checkboxes for operations with descriptions
        self.average_checkbox = QCheckBox("Average selected measurements")
        self.combine_checkbox = QCheckBox("Combine selected measurements")
        self.extract_checkbox = QCheckBox("Extract third-octave bands")
        
        operations_layout.addWidget(self.average_checkbox)
        operations_layout.addWidget(self.combine_checkbox)
        operations_layout.addWidget(self.extract_checkbox)
        
        # Add options panel that changes based on selected operation
        self.options_panel = QWidget()
        self.options_layout = QVBoxLayout()
        self.options_panel.setLayout(self.options_layout)
        self.options_panel.setVisible(False)
        operations_layout.addWidget(self.options_panel)
        
        # Add spacer
        operations_layout.addStretch(1)
        
        # Add compute button with styling
        self.compute_button = QPushButton("Compute Processing")
        self.compute_button.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:pressed {
                background-color: #2a66c8;
            }
        """)
        operations_layout.addWidget(self.compute_button)
        
        operations_group.setLayout(operations_layout)
        right_panel.addWidget(operations_group)
        
        # Add the panels to the main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 3)
        
        post_processing_section.content_layout.addLayout(main_layout, 0, 0)
        layout.addWidget(post_processing_section)
        self.setLayout(layout)
        
    def set_controller(self, controller):
        """Store the controller reference and connect signals."""
        self.controller = controller

class QGraphWindow(QWidget):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Graph Display")
        self.setGeometry(100, 100, 800, 600)  # Set window size

        layout = QVBoxLayout()
        self.graph_label = QLabel()
        self.graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.graph_label.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        layout.addWidget(self.graph_label)
        self.setLayout(layout)

class QResultsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Main section
        results_section = EnhancedSection("Post Processed Results")
        
        # Use a horizontal split for better space utilization
        main_layout = QHBoxLayout()
        
        # Left side - Measurements list and controls
        left_panel = QVBoxLayout()
        
        # Measurements list with label
        measurement_panel = QVBoxLayout()
        measurement_panel.addWidget(QLabel("Processed Measurements List"))
        
        self.concluded_measurements = QListWidget()
        self.concluded_measurements.setMinimumHeight(200)
        self.concluded_measurements.setStyleSheet("background-color: white; color: black; border: 1px solid #ccc;")
        
        measurement_panel.addWidget(self.concluded_measurements)
        left_panel.addLayout(measurement_panel)
        
        # Right side - Visualization area
        right_panel = QVBoxLayout()
        
        # Graph display area
        self.graph_label = QLabel()
        self.graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.graph_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.graph_label.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd;")
        self.graph_label.setMinimumHeight(300)  # Set minimum height for better visibility
        right_panel.addWidget(self.graph_label)
        
        # Evaluation metrics section with traditional checkbox layout to maintain controller compatibility
        metrics_box = QGroupBox("Evaluation Metrics")
        metrics_layout = QGridLayout()
        
        # Create all the original checkboxes with the same names for controller compatibility
        self.original_signal = QCheckBox("Original Signal")
        self.fft_signal_graph = QCheckBox("Fourier Transform")
        self.calibration_graph = QCheckBox("Calibration Function")
        self.absorption_coef_graph = QCheckBox("Absorption Coeficient")
        self.reflection_coef_graph = QCheckBox("Reflection Coeficient")
        self.impedance_ratio_graph = QCheckBox("Impedance Ratio")
        self.admittance_ratio_graph = QCheckBox("Admittance Ratio")
        self.transfer_function_graph = QCheckBox("Transfer Function")
        self.impedance_graph = QCheckBox("Impedance")
        self.propagation_constant_graph = QCheckBox("Propagation Constant")
        
        # Add checkboxes to grid layout in pairs
        metrics_layout.addWidget(self.absorption_coef_graph, 0, 0)
        metrics_layout.addWidget(self.reflection_coef_graph, 0, 1)
        metrics_layout.addWidget(self.impedance_ratio_graph, 1, 0)
        metrics_layout.addWidget(self.admittance_ratio_graph, 1, 1)
        metrics_layout.addWidget(self.transfer_function_graph, 2, 0)
        metrics_layout.addWidget(self.impedance_graph, 2, 1)
        metrics_layout.addWidget(self.original_signal, 3, 0)
        metrics_layout.addWidget(self.fft_signal_graph, 3, 1)
        metrics_layout.addWidget(self.calibration_graph, 4, 0)
        metrics_layout.addWidget(self.propagation_constant_graph, 4, 1)
        
        metrics_box.setLayout(metrics_layout)
        right_panel.addWidget(metrics_box)
        
        # Add the panels to the main layout with weight ratio
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)
        
        # Add the main layout to the content section with proper parameters
        results_section.content_layout.addLayout(main_layout, 0, 0)
        layout.addWidget(results_section)
        self.setLayout(layout)

    def set_controller(self, controller):
        self.controller = controller

    def set_graph(self, pixmap):
        if not pixmap.isNull():
            self.graph_window = QGraphWindow(pixmap)  # Create new window instance
            self.graph_window.show()  # Show the new window
            
class QTubeSetupTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # Tube Dimensions Section
        tube_section = EnhancedSection("Tube Dimensions")
        
        self.mic_spacing = QLineEdit()
        tube_section.add_field("Spacing between microphones (mm)", self.mic_spacing)
        
        self.mic1_sample = QLineEdit()
        tube_section.add_field("Distance Mic 1 to sample (mm)", self.mic1_sample)
        
        self.mic2_sample = QLineEdit()
        tube_section.add_field("Distance Mic 2 to sample (mm)", self.mic2_sample)
        
        self.tube_diameter = QLineEdit()
        tube_section.add_field("Tube Diameter (mm)", self.tube_diameter)
        
        layout.addWidget(tube_section)

        # Action Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Measurements")
        self.reset_button = QPushButton("Reset")
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.reset_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def set_controller(self, controller):
        self.controller = controller
        self.save_button.clicked.connect(self.controller.save_measurements)
        self.reset_button.clicked.connect(self.controller.reset_fields)

    def get_input_data(self):
        return {
            'mic_spacing': self.mic_spacing.text(),
            'mic1_sample': self.mic1_sample.text(),
            'mic2_sample': self.mic2_sample.text(),
            'tube_diameter': self.tube_diameter.text()
        }

    def reset_fields(self):
        self.mic_spacing.clear()
        self.mic1_sample.clear()
        self.mic2_sample.clear()
        self.tube_diameter.clear()

    def display_message(self, message):
        QMessageBox.information(self, "Info", message)

    def load_data(self, data):
        self.mic_spacing.setText(data.get('mic_spacing', ''))
        self.mic1_sample.setText(data.get('mic1_sample', ''))
        self.mic2_sample.setText(data.get('mic2_sample', ''))
        self.tube_diameter.setText(data.get('tube_diameter', ''))

class QTestConditionsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Temperature Section
        temp_section = EnhancedSection("Temperature Conditions")
        
        # Temperature Input
        temp_layout = QHBoxLayout()
        self.temp = QLineEdit()
        self.temp.setPlaceholderText("Enter temperature")
        temp_layout.addWidget(self.temp)
        
        # Temperature Unit Dropdown
        self.temp_unit = QComboBox()
        self.temp_unit.addItems([
            "°C (Celsius)", 
            "°F (Fahrenheit)", 
            "K (Kelvin)"
        ])
        self.temp_unit.setStyleSheet("""
            QComboBox { 
                color: black; 
                background-color: white; 
            }
        """)
        temp_layout.addWidget(self.temp_unit)
        
        # Add layout to section
        temp_section.content_layout.addLayout(temp_layout, 1, 0)
        
        layout.addWidget(temp_section)
        
        # Pressure Section
        pressure_section = EnhancedSection("Pressure Conditions")
        
        # Pressure Input
        pressure_layout = QHBoxLayout()
        self.pressure = QLineEdit()
        self.pressure.setPlaceholderText("Enter pressure")
        pressure_layout.addWidget(self.pressure)
        
        # Pressure Unit Dropdown
        self.pressure_unit = QComboBox()
        self.pressure_unit.addItems([
            "Pa (Pascal)", 
            "atm (Atmosphere)", 
            "mmHg (Millimeters of Mercury)", 
            "bar (Bar)"
        ])
        pressure_layout.addWidget(self.pressure_unit)
        
        # Add layout to section
        pressure_section.content_layout.addLayout(pressure_layout, 1, 0)
        
        layout.addWidget(pressure_section)
        
        # Humidity Section
        humidity_section = EnhancedSection("Humidity")
        self.humidity = QLineEdit()
        humidity_section.add_field("Humidity (%)", self.humidity)
        
        layout.addWidget(humidity_section)
        
        # Save Button
        self.save_button = QPushButton("Save Test Conditions")
        #save_button.clicked.connect(self.save_conditions)
        layout.addWidget(self.save_button)
        
        self.setLayout(layout)
    
    def set_controller(self, controller):
        self.controller = controller
        self.save_button.clicked.connect(self.controller.save_measurements)

    def save_conditions(self):
        # Collect data
        data = {
            'temperature': self.temp.text(),
            'temperature_unit': self.temp_unit.currentText(),
            'pressure': self.pressure.text(),
            'pressure_unit': self.pressure_unit.currentText(),
            'humidity': self.humidity.text()
        }
        
        # Save to JSON
        PersistenceManager.save_data('test_conditions.json', data)
        
        # Show success message
        QMessageBox.information(self, "Saved", "Test conditions saved successfully!")

    def load_data(self, data):
        self.temp.setText(data.get('temperature', ''))
        # Find and set the matching unit
        temp_unit_index = self.temp_unit.findText(data.get('temperature_unit', ''))
        if temp_unit_index >= 0:
            self.temp_unit.setCurrentIndex(temp_unit_index)
        
        self.pressure.setText(data.get('pressure', ''))
        # Find and set the matching unit
        pressure_unit_index = self.pressure_unit.findText(data.get('pressure_unit', ''))
        if pressure_unit_index >= 0:
            self.pressure_unit.setCurrentIndex(pressure_unit_index)
        
        self.humidity.setText(data.get('humidity', ''))

class QSamplesTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Samples Section
        samples_section = EnhancedSection("Sample Details")
        self.num_samples = QLineEdit()
        samples_section.add_field("Number of Samples", self.num_samples)
        
        self.sample_thickness = QLineEdit()
        samples_section.add_field("Thickness (mm)", self.sample_thickness)
        
        self.sample_name = QLineEdit()
        samples_section.add_field("Sample Name", self.sample_name)
        
        self.remeasure_times = QLineEdit()
        samples_section.add_field("Times Remeasure", self.remeasure_times)
        
        layout.addWidget(samples_section)
        
        # Sample List
        self.sample_list = QListWidget()
        layout.addWidget(self.sample_list)
        
        # Action Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Sample")
        self.delete_button = QPushButton("Delete Sample")
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.delete_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.controller = None

    def set_controller(self, controller):
        """Set the controller for this view."""
        self.controller = controller
        # Connect buttons to controller methods
        self.save_button.clicked.connect(self.save_samples)
        self.delete_button.clicked.connect(self.delete_sample)

    def save_samples(self):
        """Collect sample data and send to controller."""
        if self.controller:
            data = self.get_sample_data()
            self.controller.save_samples(data)
            QMessageBox.information(self, "Saved", "Sample details saved successfully!")

    def delete_sample(self):
        """Delete selected sample through controller."""
        if self.controller:
            current_item = self.sample_list.currentItem()
            if current_item:
                self.sample_list.takeItem(self.sample_list.row(current_item))
                QMessageBox.information(self, "Deleted", "Sample deleted successfully!")
            else:
                QMessageBox.warning(self, "No Selection", "Please select a sample to delete.")

    def get_sample_data(self):
        """Get the current sample data from input fields."""
        return {
            'num_samples': self.num_samples.text(),
            'sample_thickness': self.sample_thickness.text(),
            'sample_name': self.sample_name.text(),
            'remeasure_times': self.remeasure_times.text()
        }

    def load_data(self, data):
        """Load data into the input fields."""
        self.num_samples.setText(data.get('num_samples', ''))
        self.sample_thickness.setText(data.get('sample_thickness', ''))
        self.sample_name.setText(data.get('sample_name', ''))
        self.remeasure_times.setText(data.get('remeasure_times', ''))

    def display_message(self, message, title="Info"):
        """Display a message to the user."""
        QMessageBox.information(self, title, message)

class QWarningsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Signal-to-Noise Ratio Section
        snr_section = EnhancedSection("Signal Quality")
        
        # Add validators for numeric input
        double_validator = QDoubleValidator()
        
        self.snr = QLineEdit()
        self.snr.setValidator(double_validator)
        snr_section.add_field("Signal-to-Noise Ratio (dB)", self.snr)
        
        self.autospec = QLineEdit()
        self.autospec.setValidator(double_validator)
        snr_section.add_field("AutoSpectrum Max-Min (dB)", self.autospec)
        
        self.califact = QLineEdit()
        self.califact.setValidator(double_validator)
        snr_section.add_field("Calibration Factor (dB)", self.califact)
        
        layout.addWidget(snr_section)
        
        # Save Button
        self.save_button = QPushButton("Save Warning Settings")
        self.save_button.clicked.connect(self.on_save_clicked)
        layout.addWidget(self.save_button)
        
        # Status Label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: green;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)

    def set_controller(self, controller):
        """Set the controller for this view."""
        self.controller = controller
        # Load existing settings when controller is set
        self.load_settings()

    def on_save_clicked(self):
        """Handle save button click."""
        if self.controller:
            self.controller.save_warning_settings()

    def get_warning_settings(self):
        """Get current values from input fields."""
        return {
            'snr': self.snr.text(),
            'auto_spec': self.autospec.text(),
            'calib': self.califact.text()
        }

    def set_warning_settings(self, settings):
        """Set values in input fields."""
        self.snr.setText(str(settings.get('snr', '')))
        self.autospec.setText(str(settings.get('auto_spec', '')))
        self.califact.setText(str(settings.get('calib', '')))

    def load_settings(self):
        """Load existing settings through controller."""
        if self.controller:
            self.controller.load_warning_settings()

    def show_success_message(self, message):
        """Show success message."""
        self.status_label.setStyleSheet("color: green;")
        self.status_label.setText(message)
        QMessageBox.information(self, "Success", message)

    def show_error_message(self, message):
        """Show error message."""
        self.status_label.setStyleSheet("color: red;")
        self.status_label.setText(message)
        QMessageBox.critical(self, "Error", message)

class QDocumentationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = None
        
        # Create layout
        layout = QHBoxLayout()
        
        # Create sections list
        sections_panel = QWidget()
        sections_layout = QVBoxLayout()
        sections_layout.addWidget(QLabel("Documentation Sections"))
        
        self.section_list = QListWidget()
        sections_layout.addWidget(self.section_list)
        sections_panel.setLayout(sections_layout)
        
        # Create content area
        content_panel = QWidget()
        content_layout = QVBoxLayout()
        
        self.title_label = QLabel()
        self.title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #4A90E2;
            padding: 10px 0;
        """)
        content_layout.addWidget(self.title_label)
        
        self.content_browser = QTextBrowser()
        self.content_browser.setStyleSheet("""
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 10px;
        """)
        content_layout.addWidget(self.content_browser)
        
        content_panel.setLayout(content_layout)
        
        # Add panels to main layout
        layout.addWidget(sections_panel, 1)
        layout.addWidget(content_panel, 3)
        
        self.setLayout(layout)
    
    def set_controller(self, controller):
        self.controller = controller
    
    def clear_sections(self):
        """Clear all sections from the list."""
        self.section_list.clear()
    
    def add_section(self, section_name, title):
        """Add a section to the list."""
        item = QListWidgetItem(title)
        item.setData(Qt.ItemDataRole.UserRole, section_name)
        self.section_list.addItem(item)
    
    def display_section(self, title, content):
        """Display a section's content."""
        self.title_label.setText(title)
        self.content_browser.setPlainText(content)

class QReportGeneratorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Client Information Group
        client_group = QGroupBox("Client Information")
        client_layout = QGridLayout()
        
        # Create input fields for client info
        self.client_name = QLineEdit()
        self.company = QLineEdit()
        self.address = QLineEdit()
        self.purchase_order = QLineEdit()
        self.report_number = QLineEdit()
        self.test_date = QLineEdit()
        self.test_requester = QLineEdit()
        self.test_executor = QLineEdit()
        self.test_supervisor = QLineEdit()
        
        # Add fields to layout
        row = 0
        for label, widget in [
            ("Client Name*:", self.client_name),
            ("Company:", self.company),
            ("Address:", self.address),
            ("Purchase Order:", self.purchase_order),
            ("Report Number*:", self.report_number),
            ("Test Date*:", self.test_date),
            ("Test Requester:", self.test_requester),
            ("Test Executor*:", self.test_executor),
            ("Test Supervisor:", self.test_supervisor)
        ]:
            client_layout.addWidget(QLabel(label), row, 0)
            client_layout.addWidget(widget, row, 1)
            row += 1
            
        client_group.setLayout(client_layout)
        
        # Sample Information Group
        sample_group = QGroupBox("Sample Information")
        sample_layout = QGridLayout()
        
        # Create input fields for sample info
        self.product_name = QLineEdit()
        self.manufacturer = QLineEdit()
        self.description = QLineEdit()
        self.layers = QLineEdit()
        self.conditions = QLineEdit()
        self.mounting = QLineEdit()
        
        # Add fields to layout
        row = 0
        for label, widget in [
            ("Product Name:", self.product_name),
            ("Manufacturer:", self.manufacturer),
            ("Description:", self.description),
            ("Layers:", self.layers),
            ("Conditions:", self.conditions),
            ("Mounting:", self.mounting)
        ]:
            sample_layout.addWidget(QLabel(label), row, 0)
            sample_layout.addWidget(widget, row, 1)
            row += 1
            
        sample_group.setLayout(sample_layout)
        
        # Add groups to main layout
        layout.addWidget(client_group)
        layout.addWidget(sample_group)
        
        # Add note about required fields
        note = QLabel("* Required fields")
        note.setStyleSheet("color: red;")
        layout.addWidget(note)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.preview_button = QPushButton("Preview Report")
        self.preview_button.clicked.connect(self.handle_preview)
        
        self.export_button = QPushButton("Export Report")
        self.export_button.clicked.connect(self.handle_export)
        
        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.export_button)
        
        layout.addLayout(button_layout)
        
        # Add status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)

    def set_controller(self, controller):
        self.controller = controller

    def get_client_info(self) -> dict:
        """Get client information from input fields."""
        return {
            'name': self.client_name.text(),
            'company': self.company.text(),
            'address': self.address.text(),
            'purchase_order': self.purchase_order.text(),
            'report_number': self.report_number.text(),
            'test_date': self.test_date.text(),
            'test_requester': self.test_requester.text(),
            'test_executor': self.test_executor.text(),
            'test_supervisor': self.test_supervisor.text()
        }

    def get_sample_info(self) -> dict:
        """Get sample information from input fields."""
        return {
            'product_name': self.product_name.text(),
            'manufacturer': self.manufacturer.text(),
            'description': self.description.text(),
            'layers': self.layers.text(),
            'conditions': self.conditions.text(),
            'mounting': self.mounting.text()
        }

    def update_status(self, message: str):
        """Update status message."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: green;")

    def show_error(self, message: str):
        """Show error message."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: red;")

    def handle_preview(self):
        """Handle preview button click."""
        if self.controller:
            self.controller.handle_preview()

    def handle_export(self):
        """Handle export button click."""
        if self.controller:
            self.controller.handle_export()

    def set_data(self, client_info: dict, sample_info: dict):
        """Set data in input fields."""
        # Set client info
        self.client_name.setText(client_info.get('name', ''))
        self.company.setText(client_info.get('company', ''))
        self.address.setText(client_info.get('address', ''))
        self.purchase_order.setText(client_info.get('purchase_order', ''))
        self.report_number.setText(client_info.get('report_number', ''))
        self.test_date.setText(str(client_info.get('test_date', '')))
        self.test_requester.setText(client_info.get('test_requester', ''))
        self.test_executor.setText(client_info.get('test_executor', ''))
        self.test_supervisor.setText(client_info.get('test_supervisor', ''))
        
        # Set sample info
        self.product_name.setText(sample_info.get('product_name', ''))
        self.manufacturer.setText(sample_info.get('manufacturer', ''))
        self.description.setText(sample_info.get('description', ''))
        self.layers.setText(sample_info.get('layers', ''))
        self.conditions.setText(sample_info.get('conditions', ''))
        self.mounting.setText(sample_info.get('mounting', ''))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())