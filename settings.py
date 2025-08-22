import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Settings object that loads secrets and personalized configurations from .env file.
    
    This class automatically loads environment variables from a .env file in the project root
    and provides typed access to all configuration values used throughout the project.
    """
    
    model_config = ConfigDict(extra="forbid", frozen=True)
    hf_access_token: Optional[str] = Field(
        default=None,
        description="Hugging Face token for accessing models and datasets"
    )

    def __init__(self, env_file: Optional[str | Path] = None, **kwargs):
        """Initialize settings by loading from .env file and environment variables.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in project root.
            **kwargs: Additional keyword arguments to override settings.
        """
        # Load environment variables from .env file
        if env_file is None:
            # Look for .env in the project root (same directory as this settings.py file)
            project_root = Path(__file__).parent
            env_file = project_root / ".env"
        
        # Load .env file if it exists
        if Path(env_file).exists():
            load_dotenv(env_file, override=True)
        else:
            # Try to load any .env file found
            load_dotenv(override=True)
        
        # Extract values from environment variables
        env_values = {
            "hf_access_token": os.getenv("HF_ACCESS_TOKEN") or os.getenv("HF_TOKEN"),
        }
        
        # Remove None values and update with any provided kwargs
        env_values = {k: v for k, v in env_values.items() if v is not None}
        env_values.update(kwargs)
        
        super().__init__(**env_values)

# Create a global settings instance
settings = Settings()
