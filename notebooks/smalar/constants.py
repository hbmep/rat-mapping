import os

from paper.util import get_paths
from paper.constants.smalar import (
    EXPERIMENT,
    DATA_PATH_FILTERED,
    # Laterality
    GROUND_BIG,
    GROUND_SMALL,
    NO_GROUND_BIG,
    NO_GROUND_SMALL,
    # Size
    GROUND,
    NO_GROUND,
)

BUILD_DIR, TOML_PATH, DATA_PATH, MEP_MATRIX_PATH = get_paths(EXPERIMENT)
