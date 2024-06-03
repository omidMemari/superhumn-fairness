#!/bin/bash

# Run the commands 3 times
for i in {1..3}
do
    # Run the main.py script with the specified config and append output to log.txt
    echo "Running main.py iteration $i" >> small_nn_wo_counts.txt
    python3 main.py -c config_train_nn.json >> small_nn_wo_counts.txt 2>&1

    # Run the plot.py script with the specified config and append output to log.txt
    echo "Running plot.py iteration $i" >> small_nn_wo_counts.txt
    python3 plot.py -c config_test_nn.json >> small_nn_wo_counts.txt 2>&1
done