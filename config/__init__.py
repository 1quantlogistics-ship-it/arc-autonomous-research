"""
ARC Configuration Package
=========================

Provides YAML-based configuration loading for agents, models, and consensus.
Uses config/loader.py for configuration management.
"""

from config.loader import ConfigLoader, get_config_loader

__all__ = ['ConfigLoader', 'get_config_loader']
