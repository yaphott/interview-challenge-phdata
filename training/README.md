# Training

## Setup Instructions

1. Clone the repository and navigate to the project root directory:

    ```bash
    git clone https://github.com/yaphott/interview-challenge-phdata.git
    cd interview-challenge-phdata
    ```

2. Create and activate a Python virtual environment using Conda:

    ```bash
    conda env create -n housing-challenge-train -f training/environment.yml
    conda activate housing-challenge-train
    ```

## Train the Original Model

To train the original housing price prediction model, run the following command:

```bash
python3 -m 'training.train_model_original'
```

Trained model artifacts will be saved in the [../models/](../models/) directory.

## Train the Improved Model

To train an improved version of the housing price prediction model, run the following command:

```bash
python3 -m 'training.train_model_updated'
```

Trained model artifacts will be saved in the [../models/](../models/) directory.
