# UmamiFusion
UmamiFusion is a state‑of‑the‑art computational tool designed for predicting umami peptides using multimodal deep learning. By integrating sequence‑based features with ESM2‑derived graph structural representations through a bidirectional cross‑attention mechanism, UmamiFusion enables accurate and interpretable prediction of umami taste from peptide amino acid sequences.

To prioritize accessibility, the model supports end‑to‑end inference from raw peptide sequences without requiring experimental structural determination. The source code and pre‑trained models are publicly available to facilitate reproducibility and further research in peptide bioinformatics and food flavor optimization.

# Source codes:
- umami_create_data.py: Process raw peptide sequences and ESM2‑derived graph features into PyTorch Geometric data format.
- models/gcn_attention.py: Definition of the UmamiFusion model (GCNConvNet with bidirectional cross‑attention).
- models/gin_attention.py: GIN variant with cross‑attention for comparison.
- models/gat_attention.py: GAT variant with cross‑attention for comparison.
- models/gat_gcn_attention.py: Hybrid GAT‑GCN variant with cross‑attention for comparison.
- umami_train_validation.py: Training and validation script for UmamiFusion.
- umami_eval.py: Unified evaluation script for testing the trained model on benchmark datasets (UMP442, UMP614), cross‑dataset validation, and the independent BIOPEP‑UWM external test set.
- umami_t_sne.py:t‑SNE visualization of fused multimodal features across training epochs to assess class separability.
- umami_saliency.py: Saliency‑based residue‑level interpretability analysis using Captum.
- umami_auc_figure.py: ROC curve generation for comparing single‑modality, concatenated, and attention‑based fusion architectures on UMP442 and UMP614 datasets.

# Step-by-step running:
##  1.Install Python libraries needed
- Install rdkit: conda install -y -c conda-forge rdkit
- Install uni-mol: pip install unimol_tools, pip install huggingface_hub
- Install esm: pip install fair-esm, esm needs to download esm2_t33_650M_UR50D.pt and esm2_t33_650M_UR50D-contact-regression.pt
```python
conda create -n engci python=3
conda activate engci
conda install -y -c conda-forge rdkit
conda install pytorch torchvision cudatoolkit -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
```
## 2.Create data for model 1 and model 2
Running
```python
create_GPCR_data.py
create_data_esm.py
esm_dict.py
creat_data_uni-mol.py
```
## 3.Train and verify model 1
To train a model using training data. The model is chosen if it gains the best MSE for testing data.
Running
```python
conda activate engci
python train_validation_GPCR.py 0 0 0
```
where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively; the second argument is for the index of the models, 0/1/2/3 for GINConvNet, GATNet, GAT_GCN, or GCNNet, respectively; and the third argument is for the index of the cuda, 0/1 for 'cuda:0' or 'cuda:1', respectively.

## 4.Train and verify model 2
```python
python esm_uni_mol_train_validation.py
```
## 5.Test model 1, model 2 individually and test the integrated model
```python
python only_GNN_eval.py
python only_esm_eval
python esm_uni_mol_GIN_eval.py
```
