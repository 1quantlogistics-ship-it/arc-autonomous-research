"""
ARC Configuration Package
=========================

Provides YAML-based configuration loading for agents, models, and consensus.
Uses config/loader.py for configuration management.

Also re-exports settings from root config.py for convenience.
"""

from config.loader import ConfigLoader, get_config_loader

# Re-export from root config.py (parent directory)
import sys
from pathlib import Path
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Import from root config.py module
import importlib.util
_spec = importlib.util.spec_from_file_location("config_root", _root / "config.py")
_config_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_module)

ARCSettings = _config_module.ARCSettings
get_settings = _config_module.get_settings
reset_settings_cache = _config_module.reset_settings_cache

__all__ = [
    'ConfigLoader', 'get_config_loader',
    'ARCSettings', 'get_settings', 'reset_settings_cache'
]
