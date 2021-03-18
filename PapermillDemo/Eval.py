import papermill as pm
pm.execute_notebook('ROB notebook.ipynb', 'output notebook.ipynb', parameters = dict(model = 'model.h5',
datafile = 'robfile.pkl',
labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']))