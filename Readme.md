# Impedance Tube Automation

A Python application to automate Dewesoft for conducting impedance tube measurements according to ISO 10534-2 standard.

## Prerequisites

- Windows 10/11
- Python 3.9 or higher
- Dewesoft software installed
- Access to impedance tube hardware

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Python-IT
```

### 2. Create and Activate Virtual Environment

Create a new virtual environment:
```bash
python -m venv impedance-tube
```

Activate the virtual environment:

**For PowerShell:**
```powershell
impedance-tube\Scripts\Activate.ps1
```

**For Command Prompt:**
```cmd
impedance-tube\Scripts\activate.bat
```

**For Git Bash:**
```bash
source impedance-tube/Scripts/activate
```

### 3. Install Dependencies

Install all project dependencies:
```bash
pip install -e .
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Python-IT/
├── main.py                 # Main application entry point
├── Interface.py            # PyQt6 GUI interface
├── controller.py           # Application controller logic
├── model.py               # Data models and calculations
├── DataReader.py          # Dewesoft data reading utilities
├── DWDataReaderLib.dll    # Dewesoft library
├── DWDataReaderLib64.dll  # Dewesoft 64-bit library
├── DWDataReaderHeader.py  # Dewesoft header definitions
├── iso_calculations.py    # ISO 10534-2 calculations
├── calibration_calculations.py  # Calibration procedures
├── absorption.py          # Absorption coefficient calculations
├── example.py             # Example usage
├── pyproject.toml         # Project configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Usage

### Running the Application

1. Ensure your virtual environment is activated:

   **PowerShell:**
   ```powershell
   impedance-tube\Scripts\Activate.ps1
   ```

   **Command Prompt:**
   ```cmd
   impedance-tube\Scripts\activate.bat
   ```

2. Run the main application:
   ```bash
   python main.py
   ```

### Features

- **Automated Measurements**: Conduct impedance tube measurements according to ISO 10534-2
- **Data Processing**: Process and analyze measurement data
- **Report Generation**: Generate automated reports in DOCX format
- **Calibration**: Automated calibration procedures
- **GUI Interface**: User-friendly PyQt6 interface

### Configuration

The application uses several configuration files and libraries:

- **Dewesoft Integration**: Uses DWDataReader libraries for hardware communication
- **Data Processing**: NumPy and SciPy for scientific calculations
- **GUI**: PyQt6 for the user interface
- **Reports**: python-docx for document generation

## Development

### Adding New Dependencies

To add new dependencies, edit `pyproject.toml`:

```toml
[project]
dependencies = [
    "existing-package==1.0.0",
    "new-package==2.0.0"
]
```

Then reinstall:
```bash
pip install -e .
```

### Virtual Environment Management

- **Activate (PowerShell)**: `impedance-tube\Scripts\Activate.ps1`
- **Activate (CMD)**: `impedance-tube\Scripts\activate.bat`
- **Deactivate**: `deactivate`
- **Remove**: Delete the `impedance-tube` folder

## Troubleshooting

### Common Issues

1. **DLL Load Failed (PyQt6)**
   - Install Microsoft Visual C++ Redistributable from Microsoft's website
   - Reinstall PyQt6: `pip install --force-reinstall PyQt6`

2. **PowerShell Execution Policy**
   - Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

3. **Virtual Environment Not Found**
   - Ensure you're in the correct directory
   - Check that the virtual environment was created successfully
   - Try recreating: `python -m venv impedance-tube`

4. **Dewesoft Connection Issues**
   - Ensure Dewesoft is running
   - Check hardware connections
   - Verify DLL files are in the project directory

5. **pip Install Errors**
   - Upgrade pip: `python -m pip install --upgrade pip`
   - Install wheel: `pip install wheel`
   - Try installing with verbose output: `pip install -v -e .`

### Getting Help

If you encounter issues:

1. Check that all dependencies are installed correctly
2. Verify your virtual environment is activated (you should see `(impedance-tube)` in your prompt)
3. Ensure Dewesoft software is properly configured
4. Check the console output for error messages
5. Try running with Python directly: `python main.py`

## License

This project is proprietary software developed for Vibroacústica.

## Contact

- **Author**: João Volpato
- **Email**: joao.volp@vibroacustica.com
- **Company**: Vibroacústica

## Version

Current version: 0.1.0