from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem
from PyQt6.QtCore import Qt
from DataReader import DWDataReader
from model import Dewesoft, TubeSetupModel, ResultsModel, MeasurementModel, DataStore
import matplotlib.pyplot as plt
from PyQt6.QtGui import QPixmap
from io import BytesIO
from Interface import QResultsTab

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
    def __init__(self, model : ResultsModel, view : QResultsTab):
        self.model = model
        self.view = view
        self.view.set_controller(self)
        self.selected_metrics = set()
        self.selected_measurement = None

        # Populate the view with existing processed measurements
        self.populate_measurements()

        # Connect metric checkboxes to the toggle function
        self.setup_metric_connections()

    def populate_measurements(self):
        """Populates the QListWidget with processed measurements from the model."""
        self.view.concluded_measurements.clear()
        for measurement in self.model.get_processed_measurements():
            item = QListWidgetItem(measurement)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.view.concluded_measurements.addItem(item)

    def setup_metric_connections(self):
        """Connects checkboxes to toggle_metric function."""
        self.view.original_signal.stateChanged.connect(lambda state: self.toggle_metric("original_data", state))
        self.view.fft_signal_graph.stateChanged.connect(lambda state: self.toggle_metric("Fourier Transform", state))
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
        
        # Redraw graph
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
            self.selected_measurement = "TestFundo"

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


class MeasurementController:
    def __init__(self, model : MeasurementModel, view):
        self.model = model
        self.view = view
        self.view.set_controller(self)
        self.dreader = DWDataReader()
        self.dewesoft = Dewesoft()
        self.dewesoft.set_sample_rate(1000) 
        self.dewesoft.set_dimensions(800, 600)
        self.dewesoft.load_setup("C:\\Users\\jvv20\\Vibra\\DeweSoftData\\Setups\\test.dxs")
            
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

    def start_measurement(self, sample_name: str):
        """Handles the measurement process."""
        print("Buttom StartMeasurement Clicked")
        selected_item = self.view.amostras_listbox.currentItem()
        if selected_item:
            sample_name = selected_item.text()
            #self.dewesoft.measure(2, "orginal_signal")
            #self.dewesoft.close()
            self.dreader.open_data_file("TestFundo")
            data = self.dreader.get_measurements_as_dataframe()
            print("measurements read")
            self.model.add_measurement_result(data,"TestFundo")
            self.populate_results()

class MainAppController:
    def __init__(self, main_app_view):
        self.main_app_view = main_app_view

        self.data_store = DataStore()  # Shared storage
        
        # Initialize controllers for each tab and connect them to views
        self.tube_setup_controller = self.create_tube_setup_controller()
        self.samples_controller = self.create_samples_controller()
        self.test_conditions_controller = self.create_test_conditions_controller()
        self.measurements_controller = self.create_measurements_controller()
        self.results_controller = self.create_results_controller()

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

    def create_measurements_controller(self):
        self.measurement_model = MeasurementModel(self.data_store)
        self.measurement_view = self.main_app_view.measurements_tab.measure_tab
        self.measurement_controller = MeasurementController(self.measurement_model, self.measurement_view)

    def create_results_controller(self):
        self.results_model = ResultsModel(self.data_store)
        self.results_view = self.main_app_view.results_tab
        self.results_controller = ResultsController(self.results_model, self.results_view)

    def load_existing_data(self):
        # Load data from models (if necessary)
        pass

