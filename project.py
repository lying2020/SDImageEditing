import torch
import os
import sys
import argparse
import logging
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.image as image
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import matplotlib.text as text
import matplotlib.font_manager as font_manager
import matplotlib.colors as colors

current_dir = os.path.dirname(os.path.abspath(__file__))
# INPAINTING_MODEL_PATH = "/home/liying/Documents/stable-diffusion-inpainting"
INPAINTING_MODEL_PATH = "/home/liying/Desktop/stable-diffusion-inpainting"

STABLE_DIFFUSION_V1_5_MODEL_PATH = "/home/liying/Documents/stable-diffusion-v1-5"

EDITING_RESULTS_DIR = os.path.join(current_dir, "editing_results")
os.makedirs(EDITING_RESULTS_DIR, exist_ok=True)

CHECKPOINTS_DIR = os.path.join(current_dir, "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

IMAGES_DIR = os.path.join(current_dir, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

IMAGES_JSON_FILE = os.path.join(IMAGES_DIR, "images.json")
if not os.path.exists(IMAGES_JSON_FILE):
    raise FileNotFoundError(f"Images JSON file not found at {IMAGES_JSON_FILE}")

OUTPUT_DIR = os.path.join(current_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
