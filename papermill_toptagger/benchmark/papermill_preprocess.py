import papermill as pm
pm.execute_notebook('Preprocess.ipynb', 'output_preprocess_notebook.ipynb', parameters = dict(input_jets_file = 'data/test_jets.pkl',
card_file = 'data/preprocess/jet_image_trim_pt800-900_card.dat',
out_dir = 'data/results',
transformer_file = 'data/preprocess/transformer.pkl'))