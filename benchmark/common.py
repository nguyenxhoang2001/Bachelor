import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

def ensure_pkg_path():
    # Make project root importable when running as a module/script
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def load_yaml(path: str) -> Dict:
    try:
        import yaml
    except Exception:
        raise RuntimeError("PyYAML is required to load YAML configs. Install with `pip install pyyaml`.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Timer:
    """Simple wall-clock timer."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0
    
    def start(self):
        self.start_time = time.time()
        return self
    
    def stop(self):
        if self.start_time:
            self.elapsed = time.time() - self.start_time
        return self.elapsed
    
    def get_elapsed(self) -> float:
        if self.start_time:
            return time.time() - self.start_time
        return self.elapsed


class CSVWriter:
    """Helper for writing benchmark results to CSV."""
    
    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.rows = []
    
    def add_row(self, row: Dict):
        """Add a row (will be written on flush)."""
        self.rows.append(row)
    
    def flush(self):
        """Write all rows to CSV file."""
        write_csv(self.filepath, self.rows, self.fieldnames)
        print(f"Written {len(self.rows)} rows to {self.filepath}")


def format_value(value, precision: int = 4) -> str:
    """Format numeric values for CSV output."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)
