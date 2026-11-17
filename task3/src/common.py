"""Task 3 paths and shared engine imports; no Task 4 outputs are modified."""
from pathlib import Path
import os
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
ENGINE = Path(os.environ.get('HIMCM_ENGINE', WORKSPACE / 'task4' / 'src')).resolve()
OF = Path(os.environ.get('HIMCM_OF', WORKSPACE / 'tools' / 'of.cmd')).resolve()
SCRATCH = Path(tempfile.gettempdir()) / 'himcm-task3'
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(SCRATCH / 'mpl'))
sys.path.insert(0, str(ENGINE))
