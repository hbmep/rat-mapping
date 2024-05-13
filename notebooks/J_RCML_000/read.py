import os
import logging

import pandas as pd

from paper.utils import setup_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    setup_logging(dir="/Users/andres/repos/rat-mapping-paper/reports/logs", fname="read.log")

    src = "/Users/andres/data/hbmep-processed/J_RCML_000/data.csv"
    df = pd.read_csv(src)

    logger.info(type(df))
    logger.info(df.shape)

    logger.info(df.head(10))

    logger.info(df.columns)

    logger.info(
        df["participant"].unique()
    )

    logger.info(
        df["compound_position"].unique()
    )