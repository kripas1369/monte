import json
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Load configuration with fallback
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')

DEFAULT_CONFIG = {
    "simulation_params": {
        "A": 1e10,
        "E_a": 1.0,
        "k_B": 8.617e-5,
        "base_temp": 260,
        "critical_size": 4
    },
    "states": {
        "EMPTY": 0,
        "SUBSTRATE": 1,
        "MOBILE": 2,
        "STABLE": 3,
        "DEFECT": 4,
        "NUCLEATION": 5,
        "CLUSTER": 6
    },
    "visualization": {
        "colors": ["#FFFFFF", "#4E79A7", "#E15759", "#59A14F", "#F28E2B", "#EDC948", "#B07AA1"],
        "view_angle": [30, 45],
        "voxel_alpha": 0.85
    },
    "nucleation": {
        "T_m": 1700.0,
        "L": 1.0e9,
        "gamma": 0.3,
        "theta_deg": 60.0,
        "A": 1e10
    }
}

try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    logger.warning(f"config.json not found at {CONFIG_PATH}. Using default configuration.")
    CONFIG = DEFAULT_CONFIG
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in config.json: {str(e)}. Using default configuration.")
    CONFIG = DEFAULT_CONFIG

SIMULATION_PARAMS = CONFIG['simulation_params']
STATES = CONFIG['states']
VISUALIZATION = CONFIG['visualization']
NUCLEATION = CONFIG['nucleation']
STRUCTURE_3D = np.ones((3,3,3), dtype=bool)