# Impedance Tube Automation

A Python application to automate Dewesoft for conducting impedance tube measurements according to ISO 10534-2 standard.

## Requirements

- Python 3.12 or newer
- [uv](https://github.com/astral-sh/uv) (a fast Python package/dependency manager)

## Installation

1. **Unzip repository** (if you haven’t already):

2. **Install [uv](https://github.com/astral-sh/uv):**

   ```sh
   pip install uv
   ```

3. **Create virtual environment with Python 3.12:**
   ```sh
   uv venv --python 3.12
   ```

4. **Activate the virtual environment:**
   ```sh
   .venv\Scripts\activate
   ```

5. **Install all dependencies:**
   ```sh
   uv sync
   ```



6. **Set up your environment variables:**

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


## License

This project is proprietary software developed for Vibroacústica.

## Contact

- **Author**: João Volpato
- **Email**: joao.volp@vibroacustica.com
- **Company**: Vibroacústica

## Version

Current version: 0.1.0
