"""Reproduce the full D=1e-15, 0-600s demonstration using this interpreter."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    commands = [("run_demo.py", "all"), ("run_demo.py", "validate"),
                ("time_sensitivity.py",), ("mesh_check.py",)]
    for command in commands:
        subprocess.run([sys.executable, str(ROOT/command[0]), *command[1:]], check=True, cwd=ROOT)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT), "-p", "test_demo.py", "-v"],
                   check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT/"create_guide.py")], check=True, cwd=ROOT)
