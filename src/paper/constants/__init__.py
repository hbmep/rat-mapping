import os

HOME = os.getenv("HOME")
# Point this to directory where this code repository is present
REPO = os.path.join(HOME, "repos", "rat-mapping")
# Point this to directory containing rat dataset
DATA = os.path.join(HOME, "data", "rat-dataset")
# Point this to directory where output should be saved
REPORTS = os.path.join(HOME, "reports", "rat_mapping")
