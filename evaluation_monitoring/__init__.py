import sys
import os
from pathlib import Path

# Get the directory of the current script
current_script_path = Path(__file__).parent

# Construct the path to the resources directory relative to the current script
resources_path = current_script_path / '../resources'

# Resolve to an absolute path and add it to sys.path
sys.path.append(str(resources_path.resolve()))