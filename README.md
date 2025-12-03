Setup

1. Create a Virtual environment - python -m venv .venv
2. Activate the virtual environment - .venv\Scripts\activate (Windows) or source .venv/bin/activate (Mac/Linux)
3. Install the libraries from requirements.txt - pip install -r requirements.txt
4. Specifically install the CPU version of PyTorch - pip install torch --index-url https://download.pytorch.org/whl/cpu
5. Select the Python Interpreter and choose the venv (you just created)


Seqeuence of scripts

1. Run Inspect_grid.py to observe the grid structure
2. Run generate_data.py to create the datasets once again and visualize them
3. Run visualize_data.py to analyse the training data
4. Run train_mlp.py and train_sr.py to create the surrogates
5. Run the following command to activate the dashboard - streamlit run dashboard.py
