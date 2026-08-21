"""
pylidc (last released ~2020) predates several since-removed stdlib/numpy/setuptools
APIs. Import this module before importing pylidc anywhere in this codebase.
"""

import configparser
import numpy as np

configparser.SafeConfigParser = configparser.ConfigParser  # removed in Python 3.12
np.int = int  # removed in numpy>=1.24
