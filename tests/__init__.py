"""Offline test-suite for the chatbot.

Every test runs without a network: the model client and the search HTTP layer
are replaced with mocks, so nothing reaches an API provider, Tavily or the
internet, and no API key is ever read from .env or printed.

Run them with:  python -m unittest discover -v
"""

import os
import sys

# Make ``import main`` / ``import web_search`` work no matter where unittest is
# started from, without requiring the project to be installed as a package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
