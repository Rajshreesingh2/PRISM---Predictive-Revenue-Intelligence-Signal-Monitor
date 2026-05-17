"""Configuration management for PRISM"""

import yaml
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class Config:
    """Load and manage configuration"""
    
    def __init__(self, config_file='config.yaml'):
        """Initialize config from YAML file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create directories
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
        for subdir in ['raw', 'processed']:
            (self.data_dir / subdir).mkdir(exist_ok=True)
        
        (Path('logs')).mkdir(exist_ok=True)
        (Path('models')).mkdir(exist_ok=True)
        
        logger.info(f"✅ Configuration loaded from {config_file}")
    
    def get(self, key, default=None):
        """Get config value by dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
