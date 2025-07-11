#!/bin/bash

# This script is the new ENTRYPOINT for the container.

# 1. Add the root of the MuseTalk project to the Python path.
#    This makes 'from musetalk...' imports work everywhere.
export PYTHONPATH="/app/MuseTalk:${PYTHONPATH}"

# 2. Add the directory containing the 'face_detection' module to the path.
#    This solves the nested import problem.
export PYTHONPATH="/app/MuseTalk/musetalk/utils:${PYTHONPATH}"

# 3. Print the final path for debugging purposes.
echo "---"
echo "BOOTSTRAP: Launching API with PYTHONPATH=${PYTHONPATH}"
echo "---"

# 4. Execute the Uvicorn server command.
#    The 'exec' command replaces this script with the uvicorn process.
exec uvicorn musetalk_api:app --host 0.0.0.0 --port 8000