import math

import algorithms.dqn as alg
import resources.util as util
import resources.data_gen_util as dg_util


def get_test_loss_and_accuracy(test_dir_name, env, model):
    if model.compiled_loss is None or model.compiled_metrics is None:
        print("Model is not compiled. compiling...")
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    X_test, y_test = util.preprocess_data(env, dg_util.read_labeled_dataset_from_pkl_file(test_dir_name))
    result = model.evaluate(X_test, y_test, verbose=0)
    return result[0], result[1]
