"""VidFit's small public API. Importing does not load PyTorch or model weights."""

from vidfit.analyzer import VideoAnalyzer
from vidfit.core.sampling import SelectionSpec

__all__ = ["VideoAnalyzer", "SelectionSpec"]
__version__ = "0.2.0"
