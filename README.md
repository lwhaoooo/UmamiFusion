# UmamiFusion
UmamiFusion is a state‑of‑the‑art computational tool designed for predicting umami peptides using multimodal deep learning. By integrating sequence‑based features with ESM2‑derived graph structural representations through a bidirectional cross‑attention mechanism, UmamiFusion enables accurate and interpretable prediction of umami taste from peptide amino acid sequences.

To prioritize accessibility, the model supports end‑to‑end inference from raw peptide sequences without requiring experimental structural determination. The source code and pre‑trained models are publicly available to facilitate reproducibility and further research in peptide bioinformatics and food flavor optimization.

# Source codes:
- `umami_create_data.py`: Process raw peptide sequences and ESM2‑derived graph features into PyTorch Geometric data format.
- `models/gcn_attention.py`: Definition of the UmamiFusion model (GCNConvNet with bidirectional cross‑attention).
- `models/gin_attention.py`: GIN variant with cross‑attention for comparison.
- `models/gat_attention.py`: GAT variant with cross‑attention for comparison.
- `models/gat_gcn_attention.py`: Hybrid GAT‑GCN variant with cross‑attention for comparison.
- `umami_train_validation.py`: Training and validation script for UmamiFusion.
- `umami_eval.py`: Unified evaluation script for testing the trained model on benchmark datasets (UMP442, UMP614), cross‑dataset validation, and the independent BIOPEP‑UWM external test set.
- `umami_t_sne.py`: t‑SNE visualization of fused multimodal features across training epochs to assess class separability.
- `umami_saliency.py`: Saliency‑based residue‑level interpretability analysis using Captum.
- `umami_auc_figure.py`: ROC curve generation for comparing single‑modality, concatenated, and attention‑based fusion architectures on UMP442 and UMP614 datasets.
- `utils.py`: Includes TestbedDataset for data loading, evaluation metrics, and utility functions.
- `umami_data_distribution.py`: Illustrating amino acid frequency distributions and sequence length distributions for the UMP442 and UMP614 datasets.
- `umami_creat_BIOPEP.py`: Construction of the independent BIOPEP‑UWM external test set, including peptide extraction, de‑duplication against UMP442 and UMP614, and filtering of non‑standard amino acid residues.
- `models/gcn.py`: Baseline GCN model processing graph features without cross‑attention.
- `models/gat.py`: Baseline GAT model processing graph features without cross‑attention.
- `models/gat_gcn.py`: Baseline hybrid GAT‑GCN model processing graph features without cross‑attention.
- `models/ginconv.py`: Baseline GIN model processing graph features without cross‑attention.

# Step-by-step running:
##  1.Install Python libraries needed
- Install rdkit: conda install -y -c conda-forge rdkit
- pip install numpy pandas scikit-learn matplotlib seaborn tqdm
- pip install shap captum
- Install esm: pip install fair-esm, esm needs to download esm2_t33_650M_UR50D.pt and esm2_t33_650M_UR50D-contact-regression.pt
```python
conda create -n umamifusion python=3
conda activate umamifusion
conda install -y -c conda-forge rdkit
conda install pytorch torchvision cudatoolkit -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
```
## 2.Prepare Data
Place the raw dataset files (CSV format with SEQUENCE and TASTE columns) in data/umami/.

Generate the pre‑processed PyTorch Geometric data files:
```python
umami_create_data.py
umami_create_BIOPEP.py
```

## 3.Train the UmamiFusion Model
To train a model using training data. The model is chosen if it gains the best MSE for testing data.
Running
```python
conda activate engci
python train_validation_GPCR.py 0 0 0
```
where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively; the second argument is for the index of the models, 0/1/2/3 for GINConvNet, GATNet, GAT_GCN, or GCNNet, respectively; and the third argument is for the index of the cuda, 0/1 for 'cuda:0' or 'cuda:1', respectively.

## 4.Evaluate the Trained Model
To evaluate the trained model on the test set:
```python
python umami_eval.py 0 0 0
```
This script reports accuracy, MCC, AUC, precision, recall, and F1 score.

## 5.Saliency‑Based Residue‑Level Analysis
To perform residue‑level interpretability on individual peptides using Captum Saliency:
```python
python umami_saliency.py
```
Outputs are saved as PDF/PNG figures showing importance scores for each residue position.
