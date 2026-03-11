import os
from datetime import datetime


# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')


def get_timestamped_filename(base_filename, extension=None):
    """
    Generate a filename with timestamp to avoid overwriting previous results.
    
    Args:
        base_filename: Base name of the file (without extension)
        extension: File extension (e.g., 'csv', 'json', 'png')
    
    Returns:
        Filename with timestamp (e.g., 'results_20250306_143052.csv')
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if extension:
        return f"{base_filename}_{timestamp}.{extension}"
    return f"{base_filename}_{timestamp}"


def get_results_path(filename=None):
    """
    Get the path to the results directory or a specific file in it.
    
    Args:
        filename: Optional filename to append to the results path
    
    Returns:
        Path to results directory or specific file
    """
    os.makedirs(RESULTS_PATH, exist_ok=True)
    if filename:
        return os.path.join(RESULTS_PATH, filename)
    return RESULTS_PATH


os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)


FS = 250  
LOW_FREQ = 4  
HIGH_FREQ = 40  
TMIN = 0.5 # 事件发生后0.5秒开始
TMAX = 3.0 # 事件发生后3.0秒结束


RANDOM_STATE = 42
N_SPLITS = 5  
