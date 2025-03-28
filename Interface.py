import sys
import json
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, 
                             QLabel, QLineEdit, QPushButton, QGridLayout, QListWidget, 
                             QComboBox, QTextBrowser, QMessageBox, QStatusBar, QHBoxLayout, QFrame)
from PyQt6.QtGui import QFont
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
        
        self.tabs.addTab(self.tube_setup_tab, "Tube Setup")
        self.tabs.addTab(self.samples_tab, "Samples")
        self.tabs.addTab(self.test_conditions_tab, "Test Conditions")
        self.tabs.addTab(self.warnings_tab, "Warnings")
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
        save_button = QPushButton("Save Measurements")
        save_button.clicked.connect(self.save_measurements)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_fields)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(reset_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def save_measurements(self):
        # Collect data
        data = {
            'mic_spacing': self.mic_spacing.text(),
            'mic1_sample': self.mic1_sample.text(),
            'mic2_sample': self.mic2_sample.text(),
            'tube_diameter': self.tube_diameter.text()
        }
        
        # Save to JSON
        PersistenceManager.save_data('tube_setup.json', data)
        
        # Show success message
        QMessageBox.information(self, "Saved", "Tube setup measurements saved successfully!")

    def reset_fields(self):
        self.mic_spacing.clear()
        self.mic1_sample.clear()
        self.mic2_sample.clear()
        self.tube_diameter.clear()

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