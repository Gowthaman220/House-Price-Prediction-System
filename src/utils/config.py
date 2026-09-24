"""Configuration loader and structured logging utility for MLOps pipeline."""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Optional relative or absolute path to config.yaml.
        
    Returns:
        Dictionary containing configuration values.
        
    Raises:
        FileNotFoundError: If the config file cannot be found.
    """
    root = get_project_root()
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
    
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
        
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config or {}


def load_params(params_path: Optional[str] = None) -> Dict[str, Any]:
    """Load parameters from params.yaml.
    
    Args:
        params_path: Optional relative or absolute path to params.yaml.
        
    Returns:
        Dictionary containing parameters.
        
    Raises:
        FileNotFoundError: If params.yaml cannot be found.
    """
    root = get_project_root()
    if params_path is None:
        params_path = os.getenv("PARAMS_PATH", "params.yaml")
        
    path = Path(params_path)
    if not path.is_absolute():
        path = root / path
        
    if not path.exists():
        raise FileNotFoundError(f"Params file not found at: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
        
    return params or {}


def get_logger(name: str = "mlops_pipeline") -> logging.Logger:
    """Configure and return a structured logger.
    
    Args:
        name: Name of the logger, typically __name__.
        
    Returns:
        Standard library Logger instance.
    """
    logger = logging.getLogger(name)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler (store in logs/ directory)
        try:
            root = get_project_root()
            logs_dir = root / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(logs_dir / "app.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            # Fallback if filesystem write is restricted
            pass
            
    return logger
