from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem
from PyQt6.QtCore import Qt
from DataReader import DWDataReader
from model import Dewesoft, TubeSetupModel, ResultsModel, MeasurementModel, ProcessingModel, DataStore, ReportModel
import matplotlib.pyplot as plt
from PyQt6.QtGui import QPixmap
from io import BytesIO
from Interface import QResultsTab
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from docx import Document
from docx.shared import Inches

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                             QLabel, QLineEdit, QPushButton, QGridLayout, QListWidget,
                             QComboBox, QTextBrowser, QMessageBox, QStatusBar, QHBoxLayout, QFrame,
                             QFileDialog, QProgressBar, QCheckBox, QTabWidget, QListWidgetItem, QSizePolicy, QGroupBox, QAbstractItemView, QSpinBox)  # Importação corrigida
from PyQt6.QtGui import QFont, QDoubleValidator, QColor
from PyQt6.QtCore import Qt

class TubeSetupController:
    def __init__(self, model, view):
        self.model = model  # TubeSetupModel, for example
        self.view = view

        # Connect the view to the controller
        self.view.set_controller(self)

    def save_measurements(self, data):
        # Save the data to the model (this could be saving to a file, database, etc.)
        self.model.save_data(data)
        # Optionally, you could give feedback to the view (e.g., success message)
        QMessageBox.information(self.view, "Saved", "Tube setup measurements saved successfully!")
    
    def reset_fields(self):
        # Reset logic (if necessary)
        self.view.mic_spacing.clear()
        self.view.mic1_sample.clear()
        self.view.mic2_sample.clear()
        self.view.tube_diameter.clear()


class ResultsController:
    def __init__(self, model : ResultsModel, view : QResultsTab, signals):
        self.model = model
        self.view = view
        self.view.set_controller(self)
        self.selected_metrics = set()
        self.selected_measurement = None

        # Populate the view with existing processed measurements
        self.populate_measurements()

        # Connect metric checkboxes to the toggle function
        self.setup_metric_connections()

        signals.measurement_results_updated.connect(self.populate_measurements)

    def populate_measurements(self):
        """Populates the QListWidget with processed measurements from the model."""
        print("Populated Called")
        self.view.concluded_measurements.clear()
        for measurement in self.model.get_processed_measurements():
            print(measurement)
            item = QListWidgetItem(measurement)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.view.concluded_measurements.addItem(item)

    def setup_metric_connections(self):
        """Connects checkboxes to toggle_metric function."""
        self.view.original_signal.stateChanged.connect(lambda state: self.toggle_metric("Original Signal", state))
        self.view.fft_signal_graph.stateChanged.connect(lambda state: self.toggle_metric("Fourier Transform", state))
        self.view.calibration_graph.stateChanged.connect(lambda state: self.toggle_metric("Calibration Function", state))
        self.view.absorption_coef_graph.stateChanged.connect(lambda state: self.toggle_metric("Absorption Coefficient", state))
        self.view.reflection_coef_graph.stateChanged.connect(lambda state: self.toggle_metric("Reflection Coefficient", state))
        self.view.impedance_ratio_graph.stateChanged.connect(lambda state: self.toggle_metric("Impedance Ratio", state))
        self.view.admittance_ratio_graph.stateChanged.connect(lambda state: self.toggle_metric("Admittance Ratio", state))
        self.view.transfer_function_graph.stateChanged.connect(lambda state: self.toggle_metric("Transfer Function", state))
        self.view.impedance_graph.stateChanged.connect(lambda state: self.toggle_metric("Impedance", state))
        self.view.propagation_constant_graph.stateChanged.connect(lambda state: self.toggle_metric("Propagation Constant", state))

    def toggle_metric(self, metric_name, state):
        """Handles the selection/deselection of evaluation metrics and updates the graph."""
        print("metric toggled")
        print(state)
        if state == Qt.CheckState.Checked.value:
            self.selected_metrics.add(metric_name)
            print("checked checkbox")
            print(self.selected_metrics)
        else:
            self.selected_metrics.discard(metric_name)
        
        self.display_graph(metric_name)

    def display_graph(self, name):
        """Gera e exibe um gráfico para as métricas selecionadas."""
        print(f"Display Graph called, for metric {name}")
        if not self.selected_metrics:
            print(f"Selected metrics")
            print(self.selected_metrics)
            self.view.set_graph(QPixmap())  # Limpa o gráfico caso não haja métricas selecionadas
            return
        
        self.selected_measurement = self.view.concluded_measurements.selectedItems()
        if len(self.selected_measurement) == 0:
            self.selected_measurement = "TestAbsorcao_Medicao"

        # Obter dados correspondentes do modelo
        fig = self.model.generate_plot(name,self.selected_measurement)

        # Converter gráfico em QPixmap
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue())

        # Atualizar a view com o gráfico
        self.view.set_graph(pixmap)

        # Fechar a figura do Matplotlib para liberar memória
        plt.close(fig)

class MeasurementsSignalEmitter(QObject):
    measurement_results_updated = pyqtSignal()

class MeasurementController():
    def __init__(self, model : MeasurementModel, view):
        self.model = model
        self.view = view
        self.view.set_controller(self)
        self.dreader = DWDataReader()
        self.dewesoft = Dewesoft()
        self.dewesoft.set_sample_rate(1000) 
        self.dewesoft.set_dimensions(800, 600)
        self.dewesoft.load_setup("C:\\Users\\jvv20\\Vibra\\DeweSoftData\\Setups\\test.dxs")

        self.signals = MeasurementsSignalEmitter()
            
        # Populate the view with initial data
        self.populate_samples()
        self.populate_results()


    def populate_samples(self):
        """Populates the QListWidget with available samples."""
        self.view.amostras_listbox.clear()
        for sample in self.model.get_samples():
            self.view.amostras_listbox.addItem(sample)

    def populate_results(self):
        """Populates the QListWidget with measurement results."""
        self.view.resultados_listbox.clear()
        for result in self.model.get_measurement_results():
            self.view.resultados_listbox.addItem(result)

        self.signals.measurement_results_updated.emit()  # Notify others
        

    def start_measurement(self, sample_name: str):
        """Handles the measurement process."""
        print("Buttom StartMeasurement Clicked")
        selected_item = self.view.amostras_listbox.currentItem()
        if selected_item:
            sample_name = selected_item.text()
            #self.dewesoft.measure(2, "orginal_signal")
            #self.dewesoft.close()

            self.dreader.open_data_file("TestAbsorcao_Medicao")
            data = self.dreader.get_measurements_as_dataframe()
            print("measurements read")
            self.model.add_measurement_result(data,"TestAbsorcao_Medicao")
            self.dreader.close()

            self.dreader.open_data_file("TestAbsorcao_MicTrocado")
            data = self.dreader.get_measurements_as_dataframe()
            print("measurements read")
            self.model.add_measurement_result(data,"TestAbsorcao_MicTrocado")
            self.dreader.close()

            self.populate_results()


class ProcessingController:
    def __init__(self, model, view, measurements_signals):
        self.model = model
        self.view = view
        self.view.set_controller(self)
        
        # Connect signals to slots
        self.connect_signals(measurements_signals)
        
        # Populate the view with available measurements
        self.populate_measurements()
        
        # Initialize options panels
        self.setup_options_panels()
        
    def connect_signals(self, measurement_signals):
        """Connect UI signals to controller methods."""
        # Connect checkbox signals
        self.view.average_checkbox.stateChanged.connect(self.on_operation_changed)
        self.view.combine_checkbox.stateChanged.connect(self.on_operation_changed)
        self.view.extract_checkbox.stateChanged.connect(self.on_operation_changed)
        
        # Connect buttons
        self.view.compute_button.clicked.connect(self.on_compute_clicked)
        self.view.select_all_button.clicked.connect(self.select_all_measurements)
        self.view.deselect_all_button.clicked.connect(self.deselect_all_measurements)
        
        # Connect list widget item changed signal
        self.view.measurements_checklist.itemChanged.connect(self.on_measurement_selection_changed)

        measurement_signals.measurement_results_updated.connect(self.populate_measurements)
    
    def populate_measurements(self):
        """Populate the list with available measurements from the model."""
        self.view.measurements_checklist.clear()
        for measurement in self.model.get_available_measurements():
            item = QListWidgetItem(measurement)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.view.measurements_checklist.addItem(item)
    
    def setup_options_panels(self):
        """Create operation-specific option panels."""
        # Clear existing options
        while self.view.options_layout.count():
            item = self.view.options_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Create panels for each operation type
        self.average_options = self.create_average_options()
        self.combine_options = self.create_combine_options()
        self.extract_options = self.create_extract_options()
        
        # Hide all panels initially
        self.average_options.setVisible(False)
        self.combine_options.setVisible(False)
        self.extract_options.setVisible(False)
        
        # Add panels to options layout
        self.view.options_layout.addWidget(self.average_options)
        self.view.options_layout.addWidget(self.combine_options)
        self.view.options_layout.addWidget(self.extract_options)
    
    def create_average_options(self):
        """Create options panel for averaging operation."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Averaging Options"))
        
        # Add options specific to averaging
        self.average_method = QComboBox()
        self.average_method.addItems(["Arithmetic Mean", "Geometric Mean", "Weighted Average"])
        
        layout.addWidget(QLabel("Method:"))
        layout.addWidget(self.average_method)
        
        panel.setLayout(layout)
        return panel
    
    def create_combine_options(self):
        """Create options panel for combining operation."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Combination Options"))
        
        # Add options specific to combining
        self.combine_method = QComboBox()
        self.combine_method.addItems(["Sequential", "Parallel", "Custom"])
        
        layout.addWidget(QLabel("Method:"))
        layout.addWidget(self.combine_method)
        
        panel.setLayout(layout)
        return panel
    
    def create_extract_options(self):
        """Create options panel for third-octave extraction."""
        panel = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Third-Octave Options"))
        
        # Add options for third-octave extraction
        self.freq_range_min = QSpinBox()
        self.freq_range_min.setRange(20, 20000)
        self.freq_range_min.setValue(125)
        
        self.freq_range_max = QSpinBox()
        self.freq_range_max.setRange(20, 20000)
        self.freq_range_max.setValue(4000)
        
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Range:"))
        freq_layout.addWidget(self.freq_range_min)
        freq_layout.addWidget(QLabel("Hz to"))
        freq_layout.addWidget(self.freq_range_max)
        freq_layout.addWidget(QLabel("Hz"))
        
        layout.addLayout(freq_layout)
        
        panel.setLayout(layout)
        return panel
    
    def on_operation_changed(self):
        """Handle operation selection changes."""
        # Show/hide option panels based on checkbox selection
        self.view.options_panel.setVisible(True)
        
        self.average_options.setVisible(self.view.average_checkbox.isChecked())
        self.combine_options.setVisible(self.view.combine_checkbox.isChecked())
        self.extract_options.setVisible(self.view.extract_checkbox.isChecked())
        
        # Enable compute button if at least one operation is selected
        has_operations = (self.view.average_checkbox.isChecked() or 
                          self.view.combine_checkbox.isChecked() or 
                          self.view.extract_checkbox.isChecked())
        
        self.view.compute_button.setEnabled(has_operations)
        
        # Hide options panel if no operations selected
        if not has_operations:
            self.view.options_panel.setVisible(False)
    
    def on_compute_clicked(self):
        """Handle compute button click."""
        # Get selected measurements
        selected_measurements = self.get_selected_measurements()
        
        if not selected_measurements:
            QMessageBox.warning(self.view, "No Selection", 
                               "Please select at least one measurement to process.")
            return
        
        # Get selected operations
        operations = []
        if self.view.average_checkbox.isChecked():
            operations.append({
                'type': 'average',
                'method': self.average_method.currentText()
            })
            
        if self.view.combine_checkbox.isChecked():
            operations.append({
                'type': 'combine',
                'method': self.combine_method.currentText()
            })
            
        if self.view.extract_checkbox.isChecked():
            operations.append({
                'type': 'extract',
                'min_freq': self.freq_range_min.value(),
                'max_freq': self.freq_range_max.value()
            })
        
        # Call model method to process the data
        try:
            result = self.model.process_measurements(selected_measurements, operations)
            QMessageBox.information(self.view, "Processing Complete", 
                                   f"Successfully processed {len(selected_measurements)} measurements.")
        except Exception as e:
            QMessageBox.critical(self.view, "Processing Error", 
                                f"An error occurred during processing: {str(e)}")
    
    def get_selected_measurements(self):
        """Get list of selected measurements from checklist."""
        selected = []
        for i in range(self.view.measurements_checklist.count()):
            item = self.view.measurements_checklist.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected
    
    def select_all_measurements(self):
        """Select all measurements in the list."""
        for i in range(self.view.measurements_checklist.count()):
            item = self.view.measurements_checklist.item(i)
            item.setCheckState(Qt.CheckState.Checked)
    
    def deselect_all_measurements(self):
        """Deselect all measurements in the list."""
        for i in range(self.view.measurements_checklist.count()):
            item = self.view.measurements_checklist.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)
    
    def on_measurement_selection_changed(self, item):
        """Handle changes in measurement selection."""
        # Update UI based on selection if needed
        selected_count = len(self.get_selected_measurements())
        self.view.compute_button.setEnabled(selected_count > 0 and 
                                          (self.view.average_checkbox.isChecked() or 
                                           self.view.combine_checkbox.isChecked() or 
                                           self.view.extract_checkbox.isChecked()))


class ReportGeneratorController:
    def __init__(self, model, view):
        self.view = view
        self.model = model
        self.view.set_controller(self)

    def handle_preview(self):
        self.generate_report("preview_report.docx")
        self.view.update_status("Preview ready: preview_report.docx")

    def handle_export(self):
        self.generate_report("final_report.docx")
        self.view.update_status("Export complete: final_report.docx")

    def generate_report(self, filename="report.docx"):
        doc = Document()
        doc.add_heading("Measurement Report", 0)

        # Add summary
        doc.add_paragraph("This report contains results from the conducted measurements including visual graphs and mathematical analysis.")

        doc.save(filename)

class MainAppController:
    def __init__(self, main_app_view):
        self.main_app_view = main_app_view

        self.data_store = DataStore()  # Shared storage
        
        # Initialize controllers for each tab and connect them to views
        self.tube_setup_controller = self.create_tube_setup_controller()
        self.samples_controller = self.create_samples_controller()
        self.test_conditions_controller = self.create_test_conditions_controller()
        self.measurements_controller = self.create_measurements_controller()
        self.processing_controller = self.create_post_processing_controller(self.measurements_controller.signals)
        self.results_controller = self.create_results_controller(self.measurements_controller.signals)
        self.report_controller = self.create_report_controller()

    def create_tube_setup_controller(self):
        tube_setup_model = TubeSetupModel()  # Assuming the TubeSetupModel is defined earlier
        tube_setup_view = self.main_app_view.tube_setup_tab
        controller = TubeSetupController(tube_setup_model, tube_setup_view)
        return controller

    def create_samples_controller(self):
        # Similar logic for other controllers
        pass

    def create_test_conditions_controller(self):
        # Similar logic for other controllers
        pass

    def create_report_controller(self):
        report_model = ReportModel(self.data_store)
        report_view = self.main_app_view.export_tab
        controller = ReportGeneratorController(report_model, report_view)
        return controller

    def create_measurements_controller(self):
        self.measurement_model = MeasurementModel(self.data_store)
        self.measurement_view = self.main_app_view.measurements_tab.measure_tab
        measurement_controller = MeasurementController(self.measurement_model, self.measurement_view)
        return measurement_controller

    def create_results_controller(self, signals):
        self.results_model = ResultsModel(self.data_store)
        self.results_view = self.main_app_view.results_tab
        results_controller = ResultsController(self.results_model, self.results_view, signals)
        return results_controller

    def create_post_processing_controller(self, signals):
        self.post_processing_model = ProcessingModel(self.data_store)
        self.processing_view = self.main_app_view.measurements_tab.processing_tab
        processing_controller = ProcessingController(self.post_processing_model, self.processing_view, signals)
        return processing_controller

    def load_existing_data(self):
        # Load data from models (if necessary)
        pass

