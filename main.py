import sys
from PyQt6.QtWidgets import QApplication
from model import Dewesoft  # Model
from Interface import MainApp  # View
from controller import MainAppController  # Controller

def main():
    app = QApplication(sys.argv)
    
    # Create the view (MainApp)
    main_window = MainApp()
    
    # Instantiate the controller and pass both the view and the model
    app_controller = MainAppController(main_window)
    
    # Show the main window
    main_window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()