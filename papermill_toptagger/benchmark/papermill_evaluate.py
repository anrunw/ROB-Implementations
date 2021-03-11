import papermill as pm
pm.execute_notebook('Preprocess.ipynb', 'output_preprocess_notebook.ipynb', parameters = dict(tree_file = 'data/results/tree_test_jets.pkl',
output_dir = 'data/evaluate/ results/',
algorithm = 'NiNRecNNReLU',
restore = 'best',
data_dir = 'data/evaluate',
n_start = 3,
n_finish = 9))