"""Shared constants, logging setup, and device configuration."""

import warnings
import logging

import torch

warnings.filterwarnings("ignore")

# Extra (molecular) feature column templates and dimensions

EXTRA_COLS = [
    "Alkyl_{s}",
    "TailSym_{s}",
    "TailPlanarity_{s}",
    "NumHAcceptors_{s}",
    "NumHDonors_{s}",
    "TPSA_{s}",
    "MolLogP_{s}",
    "HOMO_{s}",
    "dipole_{s}",
    "MPI_{s}",
    "surface_min_{s}",
    "surface_max_{s}",
    "PSA_{s}",
]
EXTRA_DIM = len(EXTRA_COLS)

# Global feature column names and dimensions

GLOBAL_COLS = [
    "MO_ITO",
]
GLOBAL_DIM = len(GLOBAL_COLS)

# Task configuration

TASK_NAMES = ["PCE"]
NUM_TASKS = len(TASK_NAMES)

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Device

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")
