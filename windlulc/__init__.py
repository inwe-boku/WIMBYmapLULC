from pathlib import Path

__all__ = ("__version__", "PACKAGE_DIR")
__version__ = (0, 0, 1)
PACKAGE_DIR = Path(__file__).resolve().parent

from .windlulc import main
