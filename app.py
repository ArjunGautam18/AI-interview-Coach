"""Launch the Streamlit app stored in the existing virtual-environment folder."""

from pathlib import Path
import runpy


runpy.run_path(Path(__file__).parent / "venv" / "app.py", run_name="__main__")