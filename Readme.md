# Impedance Tube Automation

A Python application to automate Dewesoft for conducting impedance tube measurements according to ISO 10534-2 standard.

## Requirements

- Python 3.9 or newer
- [uv](https://github.com/astral-sh/uv) (a fast Python package/dependency manager)

## Installation

1. **Clone the repository** (if you haven’t already):

   ```sh
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Install [uv](https://github.com/astral-sh/uv):**

   ```sh
   pip install uv
   ```

3. **Create and Activate a virtual environment**
   ```sh
   python -m venv .venv

   .venv\Scripts\activate
   ```

3. **Install all dependencies (from lockfile):**

   ```sh
   uv pip install -r uv.lock
   ```



4. **Set up your environment variables:**

   Create a `.env` file in the project root with content like:

   ```
   DEWESOFT_SETUP_PATH=C:\Users\jvv20\Vibra\DeweSoftData\Setups\test.dxs
   DEWESOFT_READ_FILES_PATH=C:\Users\jvv20\Vibra\DeweSoftData\ReadFiles
   ```

## Running the Project

1. **Run the main application:**

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