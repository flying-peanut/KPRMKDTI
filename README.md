# KPRMKDTI: A Framework for Predicting Drug-Target Interactions Integrating Feature Extraction from Large Language Models and RBMO-Based Optimal Feature Selection

## Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)

## About The Project

**KPRMKDTI** is a highly accurate model for predicting drug-target interactions. 
This work provides an online prediction platform that relevant researchers can use. The platform is: **http://www.kprmkdti.com**.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

This project requires Python 3.12 or later. The necessary Python packages can be installed via `pip` using the provided `requirements.txt` file.

* **Python 3.12+**
* **pip**

### Environment Configuration

1. **Clone the repository:**

   ```sh
   git clone [https://github.com/flying-peanut/KPRMKDTI.git]
   cd ProtFPreDTI
   ```

2. **Install the required packages:**
   The dependencies are listed in the `requirements.txt` file.

   ```sh
   pip install -r requirements.txt
   ```

   **Key Dependencies:**

   * `pytorch`
   * `numpy`
   * `pandas`
   * `scikit-learn`
   * `rdkit-pypi` 
   * `joblib` 
   * `shap`

   *Note: Please refer to the `requirements.txt` file for a complete and version-specific list of dependencies.*

## Usage

Once the environment is set up, you can use the model for predictions.

**1. Prepare your input data:**

   -   Ensure your target sequences.
   -   Ensure your drug sequences are in SMILES format.
   -   Prepare a CSV file containing smiles sequences, target sequences, and labels.

**2. Run a prediction:**

   ```sh
python main.py
   ```
## Introduction to the document

- Dataset: stores the original data
- model: includes the described model structure
- trained_model: stores the trained K-BERT model
- utils: stores feature extraction methods, feature selection methods and basic data processing methods
- tips: The ProstT5 model can be obtained from huggingface
