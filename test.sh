#!/bin/bash

# Run the commands 3 times
for i in {1..3}
do
    # Run the main.py script with the specified config and append output to log.txt
    echo "Running main.py iteration $i" >> large_nn_wo_counts_2.txt
    python3 main.py -c config_train_nn.json >> large_nn_wo_counts_2.txt 2>&1

    echo "Running main.py test $i" >> large_nn_wo_counts_2.txt
    python3 main.py -c config_test_nn.json >> large_nn_wo_counts_2.txt 2>&1

    # Run the plot.py script with the specified config and append output to log.txt
    echo "Running plot.py iteration $i" >> large_nn_wo_counts_2.txt
    python3 plot.py -c config_test_nn.json >> large_nn_wo_counts_2.txt 2>&1

    echo "Running download_plots.py iteration $i" >> large_nn_wo_counts_2.txt
    python3 download_plots.py large_nn_wo_counts_2_$i >> large_nn_wo_counts_2.txt 2>&1

done