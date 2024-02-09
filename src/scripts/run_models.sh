#!/bin/bash

# define the python script path
python_script_path="./run_models.py"

# define the maximum number of attempts
max_attempts=10

# initialize attempt counter
attempt=1

# run the python script
while true; do
    echo "Attempt: $attempt"
    python3 $python_script_path

    # check the exit code of the python script
    if [ $? -eq 0 ]; then
        echo "Python script ran successfully."
        break
    else
        echo "Python script failed."

        # increment attempt counter
        ((attempt++))

        # check if max attempts reached
        if [ $attempt -gt $max_attempts ]; then
            echo "Max attempts reached. Exiting."
            break
        fi

        echo "Retrying..."
    fi
done
