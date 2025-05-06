from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem
from PyQt6.QtCore import Qt
from DataReader import DWDataReader
from model import Dewesoft, TubeSetupModel, ResultsModel, MeasurementModel, ProcessingModel, DataStore, ReportModel, TestConditionsModel
import matplotlib.pyplot as plt
from PyQt6.QtGui import QPixmap
from io import BytesIO
from Interface import QResultsTab
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from datetime import date
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT
import os
from lxml import etree
from docx.oxml import OxmlElement

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

class TestConditionsController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.set_controller(self)

    def save_measurements(self):
        temp = self.view.temp.text()
        temp_unit = self.view.temp_unit.currentText()
        self.model.set_temperature(temp,temp_unit)
        pressure = self.view.pressure.text()
        pressure_unit = self.view.pressure_unit.currentText()
        self.model.set_pressure(pressure, pressure_unit)
        humidity = self.view.humidity.text()
        self.model.set_humidity(humidity)
        self.model.save_all()
        


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
        sections = doc.sections
        for section in sections:
            section.top_margin = Cm(2.5)
            section.bottom_margin = Cm(2.5)
            section.left_margin = Cm(2.5)
            section.right_margin = Cm(2.5)

        # Create a 3x2 table for the header (no borders)
        header_table = doc.add_table(rows=3, cols=2)
        header_table.alignment = WD_TABLE_ALIGNMENT.LEFT
        header_table.autofit = False

        # Merge the first two cells for the university logo
        header_cell_1 = header_table.cell(0, 0)
        header_cell_1.merge(header_table.cell(2, 0))

        # Add logo image
        # Note: Replace 'logo.png' with the path to your actual logo file
        logo_paragraph = header_cell_1.paragraphs[0]
        logo_run = logo_paragraph.add_run()
        logo_run.add_picture('logo1.png', width=Inches(2.0))  # Adjust width as needed

        # Add department info in the second column
        acoustics_info = header_table.cell(0, 1).paragraphs[0]
        acoustics_info.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        acoustics_text = acoustics_info.add_run("VibroAcustica Acoustics Laboratories\n Joinville – SC, 89219-600, Brazil")
        acoustics_text.font.name = 'Arial'
        acoustics_text.font.size = Pt(8)

        # Add phone number
        phone_info = header_table.cell(1, 1).paragraphs[0]
        phone_info.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        phone_text = phone_info.add_run("+55 (47) 3440-1787")
        phone_text.font.name = 'Arial'
        phone_text.font.size = Pt(8)

        # Add page number
        page_info = header_table.cell(2, 1).paragraphs[0]
        page_info.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        page_text = page_info.add_run("Page 1 of 9")
        page_text.font.name = 'Arial'
        page_text.font.size = Pt(8)

        # Make the table borders invisible
        for row in header_table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        if hasattr(run, 'font'):  # Skip runs that don't have font attribute (like image runs)
                            run.font.name = 'Arial'
                # Remove borders more simply
                cell._element.tcPr.tcBorders = None

        # Add some vertical space
        doc.add_paragraph()
        doc.add_paragraph()
        doc.add_paragraph()

        # Add the LABORATORY TEST REPORT heading
        lab_report_heading = doc.add_paragraph()
        lab_report_heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
        lab_report_text = lab_report_heading.add_run("LABORATORY TEST REPORT")
        lab_report_text.font.name = 'Arial'
        lab_report_text.font.size = Pt(12)
        lab_report_text.font.bold = True
        lab_report_text.font.color.rgb = RGBColor(220, 0, 0)

        # Add the standard reference
        standard_ref = doc.add_paragraph()
        standard_ref.alignment = WD_ALIGN_PARAGRAPH.LEFT
        standard_text = standard_ref.add_run("BS EN ISO 10534-2: 2001. DETERMINATION OF SOUND ABSORPTION COEFFICIENT\nIN IMPEDANCE TUBES. TRANSFER-FUNCTION METHOD.")
        standard_text.font.name = 'Arial'
        standard_text.font.size = Pt(10)
        standard_text.font.bold = True
        standard_text.font.color.rgb = RGBColor(220, 0, 0)

        # Add vertical space
        doc.add_paragraph()
        doc.add_paragraph()

        # Create a table for client and product info
        info_table = doc.add_table(rows=4, cols=2)
        info_table.autofit = False

        # Set column widths
        for cell in info_table.columns[0].cells:
            cell.width = Cm(9)
        for cell in info_table.columns[1].cells:
            cell.width = Cm(9)

        # Client info
        client_label = info_table.cell(0, 0).paragraphs[0].add_run("CLIENT:")
        client_label.font.name = 'Arial'
        client_label.font.size = Pt(10)
        client_label.font.bold = True

        client_info = info_table.cell(0, 1).paragraphs[0].add_run("Client")
        client_info.font.name = 'Arial'
        client_info.font.size = Pt(10)

        # Product info
        product_label = info_table.cell(1, 0).paragraphs[0].add_run("PRODUCT(S):")
        product_label.font.name = 'Arial'
        product_label.font.size = Pt(10)
        product_label.font.bold = True

        product_info = info_table.cell(1, 1).paragraphs[0].add_run("")
        product_info.font.name = 'Arial'
        product_info.font.size = Pt(10)

        # Test date
        test_date_label = info_table.cell(2, 0).paragraphs[0].add_run("TEST DATE:")
        test_date_label.font.name = 'Arial'
        test_date_label.font.size = Pt(10)
        test_date_label.font.bold = True

        test_date_info = info_table.cell(2, 1).paragraphs[0].add_run(str(date.today))
        test_date_info.font.name = 'Arial'
        test_date_info.font.size = Pt(10)

        # Add a blank row for spacing
        info_table.cell(3, 0).text = ""
        info_table.cell(3, 1).text = ""

        # Make the table borders invisible
        for row in info_table.rows:
            for cell in row.cells:
                cell._element.tcPr.tcBorders = None

        # Add test report number and date of issue in a table
        report_info_table = doc.add_table(rows=1, cols=2)
        report_info_table.autofit = False

        # Set column widths
        for cell in report_info_table.columns[0].cells:
            cell.width = Cm(9)
        for cell in report_info_table.columns[1].cells:
            cell.width = Cm(9)

        # Test Report No.
        report_num_label = report_info_table.cell(0, 0).paragraphs[0].add_run("TEST REPORT No.: ")
        report_num_label.font.name = 'Arial'
        report_num_label.font.size = Pt(10)
        report_num_label.font.bold = True

        report_num = report_info_table.cell(0, 0).paragraphs[0].add_run("Test No")
        report_num.font.name = 'Arial'
        report_num.font.size = Pt(10)

        # Date of Issue
        date_issue_label = report_info_table.cell(0, 1).paragraphs[0].add_run("DATE OF ISSUE: ")
        date_issue_label.font.name = 'Arial'
        date_issue_label.font.size = Pt(10)
        date_issue_label.font.bold = True

        date_issue = report_info_table.cell(0, 1).paragraphs[0].add_run(str(date.today))
        date_issue.font.name = 'Arial'
        date_issue.font.size = Pt(10)

        # Make the table borders invisible
        for row in report_info_table.rows:
            for cell in row.cells:
                cell._element.tcPr.tcBorders = None

        # Add signature section
        doc.add_paragraph()
        signature_table = doc.add_table(rows=2, cols=2)
        signature_table.autofit = False

        # Add "Signed:" text
        signed_text = signature_table.cell(0, 0).paragraphs[0].add_run("Signed: ")
        signed_text.font.name = 'Arial'
        signed_text.font.size = Pt(10)
        signed_text.font.bold = True

        # Add "Approved:" text
        approved_text = signature_table.cell(0, 1).paragraphs[0].add_run("Approved: ")
        approved_text.font.name = 'Arial'
        approved_text.font.size = Pt(10)
        approved_text.font.bold = True

        # Add Leonardo Weber name
        name1 = signature_table.cell(1, 0).paragraphs[0].add_run("Pessoa 1\n")
        name1.font.name = 'Arial'
        name1.font.size = Pt(10)

        # Add Leonardo Weber title in red
        title1 = signature_table.cell(1, 0).paragraphs[0].add_run("Cargo 1")
        title1.font.name = 'Arial'
        title1.font.size = Pt(10)
        title1.font.color.rgb = RGBColor(220, 0, 0)

        # Add Andy Moorhouse name
        name2 = signature_table.cell(1, 1).paragraphs[0].add_run("Pessoa 2\n")
        name2.font.name = 'Arial'
        name2.font.size = Pt(10)

        # Add Andy Moorhouse title in red
        title2 = signature_table.cell(1, 1).paragraphs[0].add_run("Cargo2")
        title2.font.name = 'Arial'
        title2.font.size = Pt(10)
        title2.font.color.rgb = RGBColor(220, 0, 0)

        # Make the table borders invisible
        for row in signature_table.rows:
            for cell in row.cells:
                cell._element.tcPr.tcBorders = None

                # Helper function to create styled headings
        def add_styled_heading(doc, text, level):
            heading = doc.add_paragraph()
            heading_text = heading.add_run(text)
            heading_text.font.name = 'Arial'
            
            if level == 1:
                heading_text.font.size = Pt(12)
                heading_text.font.bold = True
            else:
                heading_text.font.size = Pt(11)
                heading_text.font.bold = True
            
            return heading

        # Helper function to create styled paragraphs
        def add_styled_paragraph(doc, text):
            para = doc.add_paragraph()
            para_text = para.add_run(text)
            para_text.font.name = 'Arial'
            para_text.font.size = Pt(10)
            return para

        # Helper function to create styled bullet points
        def add_styled_bullet_point(doc, text):
            para = doc.add_paragraph(style='List Bullet')
            para_text = para.add_run(text)
            para_text.font.name = 'Arial'
            para_text.font.size = Pt(10)
            return para

        # Add page break before starting content
        doc.add_page_break()

        # 1. TEST SAMPLES AND CONDITIONS
        add_styled_heading(doc, "1. TEST SAMPLES AND CONDITIONS", 1)

        # 1.1. Description of Test Samples
        add_styled_heading(doc, "1.1. Description of Test Samples", 2)
        add_styled_paragraph(doc, "Product Identification: {self.model.product_name}")
        add_styled_paragraph(doc, "Manufacturer: {self.model.manufacturer}")
        add_styled_paragraph(doc, "Description: {self.model.description}")
        add_styled_paragraph(doc, "Layers: {self.model.layers}")
        add_styled_paragraph(doc, "Sample Conditions: {self.model.sample_conditions}")
        add_styled_paragraph(doc, "Mounting: {self.model.mounting_description}")

        # 1.2. Test Conditions
        add_styled_heading(doc, "1.2. Test Conditions", 2)
        add_styled_paragraph(doc, "Temperature: {self.model.temperature} °C")
        add_styled_paragraph(doc, "Humidity: {self.model.humidity} %")
        add_styled_paragraph(doc, "Pressure: {self.model.pressure} hPa")

        # 2. MEASUREMENT DETAILS
        add_styled_heading(doc, "2. MEASUREMENT DETAILS", 1)

        # 2.1. Equipment
        add_styled_heading(doc, "2.1. Equipment", 2)
        for item in range(2):  # self.model.equipment:
            add_styled_bullet_point(doc, "Equipment {item+1}: Serial {1000+item}")

        # 2.2. Procedure
        add_styled_heading(doc, "2.2. Procedure", 2)
        add_styled_paragraph(doc, "Sound absorption measurements were conducted according to the standard BS EN ISO 10534-2: 2001 using impedance tubes.")
        it_text = '''For this particular test, a B&K Type 4206 Impedance Tube was used (Figure 2). This 
                    impedance tube is in accordance with BS EN ISO 10534-2 and consists of an adjustable signal 
                    filter, a loudspeaker, a sound propagation tube, microphone holders, a large sample tube (100 
                    mm diameter), and a small sample tube (29 mm diameter). Each sample tube contains an 
                    adjustable plunger for positioning the test sample and creating air gaps behind it if desired. '''

        add_styled_paragraph(doc, it_text)

        # Add image centered
        img_paragraph = doc.add_paragraph()
        img_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        img_run = img_paragraph.add_run()
        img_run.add_picture('it.png', width=Inches(4.5))  # Adjust path and size as needed

        # Add caption below image
        caption_paragraph = doc.add_paragraph("Figure 1: B&K Type 4206 Impedance Tube Setup")
        caption_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        caption_run = caption_paragraph.runs[0]
        caption_run.font.name = 'Arial'
        caption_run.font.size = Pt(10)
        caption_run.italic = True

        pt = """The test sample was mounted at 
                the end of the impedance tube by means of the sample holder, which is assumed to behave as 
                a rigid termination, with no gaps between the sample and the termination. The sample holder 
                was then mounted to the end of the tube and the microphones placed at measurement positions 
                following the characteristics described in Section 2.1 for samples with 29mm and 100mm 
                diameter. The impedance tube was mounted vertically on the wall, to allow the thin fabric to be 
                placed on top of the melamine foam surface. Finally, a broadband stationary random signal 
                was generated from B&K Pulse through a power amplifier and into the loudspeaker mounted 
                in the impedance tube. 
                For each one of the test samples, the normal incidence sound absorption coefficient was then 
                determined by decomposing the incident and reflected components of the sound field within 
                the tube, which were measured by the two separated microphones along the tube length. The 
                incident and reflected components of the sound pressure level, at the two microphone positions, 
                were then used to calculate three frequency response functions, from which the reflection and 
                absorption coefficients can be calculated."""
        
        add_styled_paragraph(doc, pt)

        doc.add_paragraph()

        def set_cell_background(cell, color):
            """Set background shading for a table cell (color as hex, e.g., 'C00000')"""
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:fill'), color)
            tcPr.append(shd)

        add_styled_paragraph(doc, "A full list of the equipment used during the tests is presented in Table 1 below.")

        # Equipment list as (Name, Serial)
        equipment_list = [
            ("DewesoftX", "2477213 / PULSE N.2"),
            ("Impedance Tube Type 4206", "2681869"),
            ("Desktop PC with Software and Peripheral Equipment", "21329"),
            ("Two ¼” Condenser Microphones Type 4187", "2677390 & 2677391"),
            ("Power Amplifier", "129003364"),
        ]

        # Create table with 2 columns
        table = doc.add_table(rows=1, cols=2)
        table.style = 'Table Grid'

        # Header row
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Equipment"
        hdr_cells[1].text = "Serial/Ref No."

        for cell in hdr_cells:
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = para.runs[0]
            run.font.name = 'Arial'
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)  # White text

            set_cell_background(cell, "C00000")  # Red background

        # Add equipment data
        for name, serial in equipment_list:
            row_cells = table.add_row().cells
            for idx, text in enumerate((name, serial)):
                para = row_cells[idx].paragraphs[0]
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                run = para.add_run(text)
                run.font.name = 'Arial'
                run.font.size = Pt(10)

        # Caption
        caption = doc.add_paragraph("Table 1: List of equipment.")
        caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        caption.runs[0].font.name = 'Arial'
        caption.runs[0].font.size = Pt(9)
        caption.runs[0].italic = True

        # 2.3. Calculations
        add_styled_heading(doc, "2.3. Calculations", 2)
        add_styled_paragraph(doc, "Absorption coefficient α is calculated from reflection coefficient R:")
        
        # Create a styled quote
        quote_para = doc.add_paragraph()
        quote_run = quote_para.add_run("α = 1 - |R|²")
        quote_run.font.name = 'Arial'
        quote_run.font.size = Pt(10)
        quote_run.font.italic = True
        quote_para.paragraph_format.left_indent = Cm(1.0)
        
        add_styled_paragraph(doc, "R is derived from the measured transfer function H₁₂ of the two microphones.")

        # 3. RESULTS
        add_styled_heading(doc, "3. RESULTS", 1)

        table = doc.add_table(rows=1, cols=4)
        table.style = 'Table Grid'

        # Set header text
        headers = ["Frequency\nHz", "αₙ (tube Ø100 mm)", "αₙ (tube Ø29 mm)", "αₙ (combined)"]
        for idx, text in enumerate(headers):
            cell = table.cell(0, idx)
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run(text)
            run.font.name = 'Arial'
            run.font.size = Pt(10)
            cell.vertical_alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Example data - replace with real self.model.results
        example_data = [
            (80, 0.07, 0.00, 0.07),
            (100, 0.08, 0.00, 0.08),
            (125, 0.30, 0.30, 0.30),
            (160, 0.38, 0.45, 0.39),
            (200, 0.50, 0.55, 0.51),
            # Add more as needed...
        ]

        # Add data rows
        for freq, alpha100, alpha29, combined in example_data:
            row_cells = table.add_row().cells
            values = [freq, alpha100, alpha29, combined]

            for idx, val in enumerate(values):
                text = "-" if val == 0.00 else f"{val:.2f}" if isinstance(val, float) else str(val)
                para = row_cells[idx].paragraphs[0]
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = para.add_run(text)
                run.font.name = 'Arial'
                run.font.size = Pt(10)

        # Add image acoustic abs coef
        img_paragraph = doc.add_paragraph()
        img_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        img_run = img_paragraph.add_run()
        img_run.add_picture('ac.png', width=Inches(4.5))  # Adjust path and size as needed

        # Add caption below image
        caption_paragraph = doc.add_paragraph("Figure 1: Absorption Coeficent vs Frequency")
        caption_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        caption_run = caption_paragraph.runs[0]
        caption_run.font.name = 'Arial'
        caption_run.font.size = Pt(10)
        caption_run.italic = True

        # Save the document
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
        tube_setup_model = TubeSetupModel(self.data_store)  # Assuming the TubeSetupModel is defined earlier
        tube_setup_view = self.main_app_view.tube_setup_tab
        controller = TubeSetupController(tube_setup_model, tube_setup_view)
        return controller

    def create_samples_controller(self):
        # Similar logic for other controllers
        pass

    def create_test_conditions_controller(self):
        conditions_model = TestConditionsModel(self.data_store)
        conditions_view = self.main_app_view.test_conditions_tab
        controller = TestConditionsController(conditions_model, conditions_view)
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

