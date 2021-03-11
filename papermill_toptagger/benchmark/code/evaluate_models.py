# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB) - Top Tagger Benchmark Demo.
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

"""Default implementation for the model evaluation step in the Top Tagger
workflow.

usage: evaluate-models.py [-h] [-a ARCHITECTURE] [-r RESTORE] [-s N_START]
                          [-f N_FINISH]
                          tree_file data_dir output_dir
"""

import argparse
import logging
import numpy as np
import os
import pickle
import sys
import time

import files as fn
import code.recnn.evaluate as eval
import code.recnn.utils as utils


# -- Helper Functions ---------------------------------------------------------

def load_results(data_dir, result_dir, n_start, n_finish):
    """Load all the info for each run in a dictionary (hyperparameteres, auc,
    fpr, tpr, output prob, etc).
    """
    # For each folder in the result directory that starts with the prefix
    # 'run_' and contains a metrics resutl file we read the results as well as
    # the parameters from the respective folder in the data directory.
    results = list()
    for subdir in os.listdir(result_dir):
        if subdir.startswith(fn.RUN_DIR_PREFIX):
            run_id = int(subdir[subdir.rfind('_') + 1])
            if n_start <= run_id < n_finish:
                run_dir = os.path.join(result_dir, subdir)
                result_file = os.path.join(run_dir, fn.Y_PROB_TRUE_FILE)
                if os.path.isfile(result_file):
                    with open(result_file, 'rb') as f:
                        y_prob_true = list(pickle.load(f))
                    results.append(np.asarray(y_prob_true))
    return np.asarray(results)


# -- Main Function ------------------------------------------------------------

def main(tree_file, output_dir, algorithm, restore, data_dir):
    # Ensure that the output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # Initialize the logger
    log = logging.getLogger()
    log.addHandler(logging.StreamHandler(sys.stdout))
    log_file = os.path.join(output_dir, fn.EVAL_LOG_FILE)
    log.addHandler(logging.FileHandler(log_file))
    logging.getLogger().setLevel(logging.DEBUG)
    # Call evaluation function for each model
    start_time = time.time()
    for n_run in np.arange(n_start, n_finish):
        run_id = '{}{}'.format(fn.RUN_DIR_PREFIX, n_run)
        run_dir = os.path.join(data_dir, run_id)
        restore_file = os.path.join(run_dir, '{}.pth.tar'.format(restore))
        eval.run(
            tree_file=tree_file,
            params=utils.Params(os.path.join(run_dir, fn.PARAMS_FILE)),
            architecture=algorithm,
            restore_file=restore_file,
            output_dir=output_dir
        )
    # Combine result files to generate output file
    logging.info('Combine runs {}-{}'.format(n_start, n_finish))
    results = load_results(
        data_dir=data_dir,
        result_dir=output_dir,
        n_start=n_start,
        n_finish=n_finish
    )
    # Saving output probabilities and true values
    output_file = os.path.join(output_dir, fn.Y_PROB_BEST_FILE)
    msg = 'Saving output probabilities and true values to {}'
    logging.info(msg.format(output_file))
    with open(output_file, 'wb') as f:
        pickle.dump(results[:, :, 0], f)
    # Log runtime information
    exec_time = time.time() - start_time
    logging.info('Preprocessing time (minutes) = {}'.format(exec_time / 60))


