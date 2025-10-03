# DVC-Pipeline-for-Efficient-Hyperparameter-Tuning
 The key focus here is on **automating and tracking hyperparameter tuning experiments**. This project demonstrates how to create a reproducible DVC pipeline that not only trains a model but also systematically finds the best hyperparameters for it, ensuring peak performance.


## Key Feature: Automated Hyperparameter Tuning 🔬

Finding the optimal model settings manually is time-consuming and inefficient. This project solves that by integrating a hyperparameter tuning library (e.g., Optuna, Hyperopt) into the DVC pipeline.

* **Automation**: The pipeline automatically searches through various combinations of parameters.
* **Reproducibility**: DVC tracks the parameters, code, and data for each experiment, making results easy to reproduce.
* **Efficiency**: It quickly identifies the best-performing model configuration without manual intervention.

## MLOps Tech Stack

This project uses a combination of tools focused on experimentation and reproducibility:

* **Experimentation**: `scikit-learn`, `pandas`
* **Hyperparameter Tuning**: `yaml.safe_load()` ,`params.yaml`, `n_estimators`
* **Data & Pipeline Versioning**: `dvc`, `git`
* **Environment Management**: `virtualenv`, `pip`

# How to Execute This Project

Follow these steps to reproduce the pipeline and the tuning experiments.

# Step 1: Clone the Repository**

git clone https://github.com/DibyashaSahu/DVC-Pipeline-for-Efficient-Hyperparameter-Tuning.git
cd DVC-Pipeline-for-Efficient-Hyperparameter-Tuning

# Step 2: Install Dependencies

# Create and activate a virtual environment

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install requirements

pip install -r requirements.txt

# Step 3: Reproduce the Pipeline

This command will pull the data and run the entire pipeline, including the hyperparameter search, to produce the final tuned model.

dvc repro

# Step 4: View Results

The best hyperparameters and the performance metrics are saved. You can view them by checking the generated output files.

# Example command to see DVC-tracked metrics

dvc metrics show

# Project Outcome

The pipeline successfully identified the optimal set of hyperparameters, resulting in a model with improved accuracy. The entire process is versioned and reproducible, showcasing a mature approach to the machine learning experimentation phase.
