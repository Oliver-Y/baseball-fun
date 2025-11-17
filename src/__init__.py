"""
Baseball trajectory visualization package.

Logging is configured here once for the entire package.
Individual modules should use: logger = logging.getLogger(__name__)
"""
import logging

# Configure root logger once at package import
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

__all__ = []

