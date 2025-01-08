# G0-LM

### Author: Shi Pan, UCL Genetics Institute

G0-LM is a foundation model to classify G0, slow and fast cycling states in single cell RNA-seq cancer data. This model has currently been trained on breast cancer only.

![G0-LM classification pipeline and performance. a G0 arrest and proliferation state classification pipeline in scRNA-seq data. The first training phase involves binary classification training for each state separately using the Parameter-Efficient Fine-Tuning (PEFT) method. This allows the original pre-trained model’s feature space to better represent G0 arrest/proliferation. The final G0-LM model is obtained by fusing the three binary classification models for the individual states (G0 arrest, slow cycling and fast cycling). b Receiver operating characteristic (ROC) curves per cell cycle category displaying performances of G0-LM. c-d PCA (c) and UMAP (d) embedding spaces of the cell cycle categories.![image](https://github.com/user-attachments/assets/5469f770-f098-4741-a172-545266189062)
](G0-LM.png)

Related publications:

## Repository Structure
```
G0-LM/
├── src/                      # Source code directory
│   ├── DLM_exp_helpers/     # Helper functions for experiments
│   ├── scLLM/               # Main model implementation
│   └── scLLM_support_data/  # Supporting data and utilities
├── Data/                    # Data directory for training and testing
├── Exp/                     # Experiment results and configurations
├── Outputs/                 # Model outputs and predictions
└── LICENSE                  # License information
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA compatible GPU (recommended)
- Conda (recommended for environment management)

### Environment Setup
1. Create and activate a new conda environment:
```bash
conda create -n g0lm python=3.8
conda activate g0lm
```

2. Install required packages:
```bash
pip install torch torchvision
pip install scanpy anndata
pip install scikit-learn pandas numpy matplotlib
pip install lifelines dill tqdm
```


## Usage

The project workflow consists of several key steps, each with its specific purpose and outputs:

### Step 0: Data Preprocessing
Located in `Exp/step0_preprocess/`
- Processes raw single-cell RNA sequencing data
- Performs quality control and filtering
- Normalizes data and prepares it for model input
- Outputs preprocessed data in AnnData (.h5ad) format

### Step 1: Phase 1 Training
Located in `Exp/step1_train_phase1/`
- Initial model training phase
- Trains the model on breast cancer dataset
- Focuses on learning cell cycle state patterns
- Includes model checkpoints and training logs

### Step 2: Phase 2 Training
Located in `Exp/step2_train_phase2/`
- Fine-tuning phase of the model
- Adapts the model for specific cancer types
- Optimizes performance on G0 state classification
- Generates refined model weights

### Step 3: Embedding Space Analysis
Located in `Exp/step3_embedding_space/`
- Analyzes the learned cell state representations
- Visualizes embedding space using dimensionality reduction
- Identifies cell state clusters
- Provides insights into model's internal representations

### Step 4: Model Evaluation
Located in `Exp/step4_eval/`
- Comprehensive model evaluation
- Performs cross-validation
- Generates performance metrics
- Creates visualization of results

Each step contains detailed Jupyter notebooks and scripts with specific parameters and configurations. To reproduce results:

1. Start with preprocessing your data following notebooks in step0
2. Follow the sequential steps (1-4) in the `Exp/` directory
3. Each step's directory contains README files with specific instructions
4. Results and outputs will be saved in the `Outputs/` directory

## Experiments
Check the `Exp/` directory for example notebooks and scripts demonstrating various use cases and experiments.

## License
This project is licensed under the terms specified in the LICENSE file.

## Citation
If you use this model in your research, please cite:
[Citation information to be added]

## Contact
For questions and feedback, please contact Shi Pan at UCL Genetics Institute.
