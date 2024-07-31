#!/bin/bash

# List of Python scripts to run
scripts=(
    "deepsort.py"
    "ACTINN.py"
    "Celltypist.py"
    "singlecellnet.py"
    "svm.py"
    "WordSageEmbed.py"
    "deepsort_w2v.py"
    "ACTINN_w2v.py"
    "CellTypist_w2v.py"
    "singlecellnet_w2v.py"
    "svm_w2v.py"
    "WordSageRun.py"
)

for script in "${scripts[@]}"
do
    echo "Running $script..."
    python "$script" >> output.txt 2>&1
    echo "Finished running $script."
done
