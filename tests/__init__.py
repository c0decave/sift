"""Test package setup.

Point the cache at a throwaway directory before anything imports sift. Without
this the suite writes into the developer's real ~/.cache/sift, where entries
from one test are visible to the next — which is exactly how a cache-key
collision hid here once, and it would equally hide a stale summary in day-to-day
use.
"""
import atexit
import os
import shutil
import tempfile

_CACHE = tempfile.mkdtemp(prefix="sift-test-cache-")
os.environ["SIFT_CACHE_DIR"] = _CACHE
atexit.register(shutil.rmtree, _CACHE, True)
