from pathlib import Path
import json
import os
import shutil
from typing import Dict, Any
from lyrics_aligner.src.core.models import TimingConfig

class ConfigLoader:
    @staticmethod
    def load_config() -> TimingConfig:
        """Load configuration from default_config.json."""
        config_path = Path(__file__).parent / "default_config.json"
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return TimingConfig(**config_data)
        except Exception as e:
            print(f"Error loading config: {e}")
            return TimingConfig()  # Return default values if loading fails

    @staticmethod
    def read_settings() -> Dict[str, Any]:
        """Read settings from settings.txt file."""
        settings_path = Path(os.getcwd()) / "settings.txt"
        settings = {}
        
        try:
            with open(settings_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, value = [x.strip() for x in line.split("=", 1)]
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        settings[key] = value
        except FileNotFoundError:
            settings_template = Path(__file__).parent / "settings_template.txt"
            if settings_template.exists():
                shutil.copy(str(settings_template), str(settings_path))
                print("Created new settings.txt file. Please edit it with your settings.")
            else:
                print("Error: settings.txt and template not found!")
        except Exception as e:
            print(f"Error reading settings: {e}")
        
        return settings