import sys
import json
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                             QLabel, QLineEdit, QPushButton, QGridLayout, QListWidget,
                             QComboBox, QTextBrowser, QMessageBox, QStatusBar, QHBoxLayout, QFrame,
                             QFileDialog, QProgressBar, QCheckBox, QTabWidget, QListWidgetItem, QSizePolicy)  # Importação corrigida
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
        }
        
        /* Tab Widget Styling */
        QTabWidget::pane {
            border: 2px solid #e0e0e0;
            background-color: white;
        }
        
        QTabBar::tab {
            background-color: #f0f0f0;
            color: #333;
            padding: 10px 20px;
            margin-right: 5px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            min-width: 120px;
        }
        
        QTabBar::tab:selected {
            background-color: #e0e0e0;
            font-weight: bold;
        }
        
        /* Input Fields */
        QLineEdi {
            padding: 10px;
            border: 2px solid #ccc;
            border-radius: 8px;
            background-color: white;
            min-height: 40px;
        }

        QComboBox {
            color: #333; /* Default text color */
            background-color: white;
            padding: 10px;
            border: 2px solid #ccc;
            border-radius: 8px;
            min-height: 40px;
        }

        QComboBox::down-arrow {
            color: #4A90E2; /* Color of the dropdown arrow */
        }

        QComboBox::item {
            color: black; /* Color of items in the dropdown list */
        }

        QComboBox::item:selected {
            color: balck; /* Color of selected item in dropdown */
            background-color: #4A90E2; /* Background of selected item */
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
            color: #333;
            font-weight: bold;
            margin-bottom: 5px;
        }

        /* Checkboxes */
        QCheckBox {
            color: #333; /* Cor do texto do checkbox */
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
        
        self.tabs.addTab(self.tube_setup_tab, "Tube Setup")
        self.tabs.addTab(self.samples_tab, "Samples")
        self.tabs.addTab(self.test_conditions_tab, "Test Conditions")
        self.tabs.addTab(self.warnings_tab, "Warnings")
        self.tabs.addTab(self.documentation_tab, "Documentation")
        self.tabs.addTab(self.measurements_tab, "Measuments")
        self.tabs.addTab(self.results_tab, "Results")
        
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
        post_processing_layout = QGridLayout()

        self.measurements_checklist = QListWidget()
        self.measurements_checklist.setStyleSheet("color: #333;")

        item1 = QListWidgetItem("Test")
        item1.setFlags(item1.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item1.setCheckState(Qt.CheckState.Unchecked)
        self.measurements_checklist.addItem(item1)

        item2 = QListWidgetItem("Test")
        item2.setFlags(item2.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item2.setCheckState(Qt.CheckState.Checked)
        self.measurements_checklist.addItem(item2)

        item3 = QListWidgetItem("123")
        item3.setFlags(item3.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item3.setCheckState(Qt.CheckState.Checked)
        self.measurements_checklist.addItem(item3)

        post_processing_layout.addWidget(QLabel("Measurements List"), 0, 0)
        post_processing_layout.addWidget(self.measurements_checklist, 1, 0)

        operations_layout = QVBoxLayout()
        operations_layout.addWidget(QLabel("Operations"))

        self.avarege_checkbox = QCheckBox("Avarege")
        self.combine_checkbox = QCheckBox("Combine")
        self.extract_checkbox = QCheckBox("Extract third-octave")

        operations_layout.addWidget(self.avarege_checkbox)
        operations_layout.addWidget(self.combine_checkbox)
        operations_layout.addWidget(self.extract_checkbox)

        self.compute_button = QPushButton("Compute")
        operations_layout.addWidget(self.compute_button)

        post_processing_layout.addLayout(operations_layout, 1, 2)

        post_processing_section.content_layout.addLayout(post_processing_layout, 0, 0)
        layout.addWidget(post_processing_section)
        self.setLayout(layout)

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

        results_section = EnhancedSection("Post Processed Results")
        results_layout = QGridLayout()

        self.concluded_measurements = QListWidget()
        self.concluded_measurements.setStyleSheet("color: black;")

        item1 = QListWidgetItem("Measurement-1")
        item1.setFlags(item1.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item1.setCheckState(Qt.CheckState.Unchecked)
        item1.setForeground(QColor("black"))
        self.concluded_measurements.addItem(item1)

        item2 = QListWidgetItem("Measurement-2")
        item2.setFlags(item2.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item2.setCheckState(Qt.CheckState.Checked)
        self.concluded_measurements.addItem(item2)

        item3 = QListWidgetItem("Measurement-3")
        item3.setFlags(item3.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item3.setCheckState(Qt.CheckState.Checked)
        self.concluded_measurements.addItem(item3)

        results_layout.addWidget(QLabel("Processed Measurements List"), 0, 0)
        results_layout.addWidget(self.concluded_measurements, 1, 0)

        evaluation_layout = QVBoxLayout()
        evaluation_layout.addWidget(QLabel("Evaluation Metrics"))

        self.original_signal = QCheckBox("Original Signal")
        self.fft_signal_graph = QCheckBox("Fourier Transform")
        self.absorption_coef_graph = QCheckBox("Absorption Coeficient")
        self.reflection_coef_graph = QCheckBox("Reflection Coeficient")
        self.impedance_ratio_graph = QCheckBox("Impedance Ratio")
        self.admittance_ratio_graph = QCheckBox("Admittance Ratio")
        self.transfer_function_graph = QCheckBox("Transfer Function")
        self.impedance_graph = QCheckBox("Impedance")
        self.propagation_constant_graph = QCheckBox("Propagation Constant")

        horizontal_layout1 = QHBoxLayout()
        horizontal_layout1.addWidget(self.absorption_coef_graph)
        horizontal_layout1.addWidget(self.reflection_coef_graph)
        evaluation_layout.addLayout(horizontal_layout1)

        horizontal_layout2 = QHBoxLayout()
        horizontal_layout2.addWidget(self.impedance_ratio_graph)
        horizontal_layout2.addWidget(self.admittance_ratio_graph)
        evaluation_layout.addLayout(horizontal_layout2)

        horizontal_layout3 = QHBoxLayout()
        horizontal_layout3.addWidget(self.transfer_function_graph)
        horizontal_layout3.addWidget(self.impedance_graph)
        evaluation_layout.addLayout(horizontal_layout3)

        horizontal_layout4 = QHBoxLayout()
        horizontal_layout4.addWidget(self.original_signal)
        horizontal_layout4.addWidget(self.fft_signal_graph)
        evaluation_layout.addLayout(horizontal_layout4)

        evaluation_layout.addWidget(self.propagation_constant_graph)

        results_layout.addLayout(evaluation_layout, 1, 2)

        # Adiciona o QLabel para exibir os gráficos
        self.graph_label = QLabel()
        self.graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.graph_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Allow expansion
        self.graph_label.setScaledContents(True)  # Scale pixmap to fit the label

        results_section.content_layout.addLayout(results_layout, 0, 0)
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
        save_button = QPushButton("Save Test Conditions")
        save_button.clicked.connect(self.save_conditions)
        layout.addWidget(save_button)
        
        self.setLayout(layout)

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

# Placeholder classes for other tabs (you can implement these similarly)
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
        save_button = QPushButton("Save Sample")
        save_button.clicked.connect(self.save_samples)
        delete_button = QPushButton("Delete Sample")
        delete_button.clicked.connect(self.delete_sample)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(delete_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def save_samples(self):
        # Collect data
        data = {
            'num_samples': self.num_samples.text(),
            'sample_thickness': self.sample_thickness.text(),
            'sample_name': self.sample_name.text(),
            'remeasure_times': self.remeasure_times.text()
        }
        
        # Save to JSON
        PersistenceManager.save_data('samples.json', data)
        
        # Show success message
        QMessageBox.information(self, "Saved", "Sample details saved successfully!")

    def delete_sample(self):
        # Get current selected item
        current_item = self.sample_list.currentItem()
        if current_item:
            self.sample_list.takeItem(self.sample_list.row(current_item))

    def load_data(self, data):
        self.num_samples.setText(data.get('num_samples', ''))
        self.sample_thickness.setText(data.get('sample_thickness', ''))
        self.sample_name.setText(data.get('sample_name', ''))
        self.remeasure_times.setText(data.get('remeasure_times', ''))

class QWarningsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Signal-to-Noise Ratio Section
        snr_section = EnhancedSection("Signal Quality")
        self.snr = QLineEdit()
        snr_section.add_field("Signal-to-Noise Ratio (dB)", self.snr)
        
        self.autospec = QLineEdit()
        snr_section.add_field("AutoSpectrum Max-Min (dB)", self.autospec)
        
        self.califact = QLineEdit()
        snr_section.add_field("Calibration Factor (dB)", self.califact)
        
        layout.addWidget(snr_section)
        
        # Save Button
        save_button = QPushButton("Save Warnings")
        layout.addWidget(save_button)
        
        self.setLayout(layout)

class QDocumentationTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Documentation Browser
        self.doc_browser = QTextBrowser()
        layout.addWidget(self.doc_browser)
        
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())