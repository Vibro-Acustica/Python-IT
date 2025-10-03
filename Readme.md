# Impedance Tube Automation

A Python application to automate Dewesoft for conducting impedance tube measurements according to ISO 10534-2 standard.

## Requirements

- Python 3.12 or newer
- [uv](https://github.com/astral-sh/uv) (a fast Python package/dependency manager)

## Installation

1. **Unzip repository** (if you haven't already)

2. **Install [uv](https://github.com/astral-sh/uv):**

   On Windows (PowerShell):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Or via pip:
   ```sh
   pip install uv
   ```

3. **Create virtual environment with Python 3.12:**
   ```sh
   uv venv --python 3.12
   ```

4. **Install all dependencies:**
   ```sh
   uv sync
   ```

   This will automatically:
   - Create a virtual environment in `.venv`
   - Install all dependencies from the lockfile

5. **Set up your environment variables:**

   Create a `.env` file in the project root with content like, use the path in your machine, this is just an example:

   ```
   DEWESOFT_SETUP_PATH=C:\Users\jvv20\Vibra\DeweSoftData\Setups\test.dxs
   DEWESOFT_READ_FILES_PATH=C:\Users\jvv20\Vibra\DeweSoftData\ReadFiles
   ```

## Running the Project

1. **Make sure the virtual environment is activated:**
   ```sh
   .venv\Scripts\activate
   ```

2. **Run the main application:**
   ```sh
   python main.py
   ```


## Troubleshooting

### ModuleNotFoundError: No module named 'win32com' or 'PyQt6'

If you encounter this error, it means the dependencies are not installed in the active Python environment. Follow these steps:

1. **Delete any existing `.venv` folder** in the project directory
2. **Run `uv sync` again** to create a fresh virtual environment with all dependencies
3. **Use `uv run python main.py`** to run the application, which ensures the correct environment is used

Alternatively, if you prefer manual activation:
```sh
.venv\Scripts\activate
python main.py
```

Make sure you see `(.venv)` at the beginning of your command prompt, indicating the virtual environment is active.

## License

This project is proprietary software developed for Vibroacústica.

## Contact

- **Author**: João Volpato
- **Email**: joao.volp@vibroacustica.com
- **Company**: Vibroacústica

## Version

Current version: 0.1.0
