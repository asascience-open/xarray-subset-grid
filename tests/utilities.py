"""
Assorted utilities useful for the tests.
"""

import os
import contextlib

try:
    import urllib.request as urllib_request  # for python 3
except ImportError:
    import urllib2 as urllib_request  # for python 2

import pytest
import glob

from .get_remote_data import get_datafile


def get_test_file_dir():
    """
    returns the test file dir path
    """
    test_file_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    return test_file_dir