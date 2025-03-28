from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from DataReader import DWDataReader
from model import Dewesoft, TubeSetupModel

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

class MainAppController:
    def __init__(self, main_app_view):
        self.main_app_view = main_app_view
        
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
        # Create and set up controller for measurements tab
        pass

    def create_results_controller(self):
        # Create and set up controller for results tab
        pass

    def load_existing_data(self):
        # Load data from models (if necessary)
        pass

