# Deep Learning-Based Classification of Oral Cancer Cells

## Research Paper Supplementary Materials

This repository contains the code and supplementary materials for our research paper on automated classification of oral cancer cells using deep learning techniques.

### Abstract

We present a deep learning approach for automated classification of oral cancer cells into three distinct categories: 5nMTG (treated cancer cells), Control (untreated cancer cells), and NonCancerOral (non-cancerous oral cells). Our methodology employs state-of-the-art convolutional neural networks to identify cellular morphological features that distinguish between these classes with high accuracy. The results demonstrate the potential of machine learning techniques for aiding in cancer diagnostics and treatment response monitoring.

### Dataset Description

The dataset consists of microscopic images of oral cells collected during the period of May 21 to June 18, organized as follows:

```
Data/
├── oral cancer 0521-0618_tag300_Test/
│   ├── 5nMTG/         # Cancer cells treated with 5nM TG
│   ├── Control/        # Untreated cancer cells
│   └── NonCancerOral/  # Non-cancerous oral cells
└── oral cancer 0521-0618_tag300_Val/
    ├── 5nMTG/
    ├── Control/
    └── NonCancerOral/
```

### Methodology

Our study compared multiple deep learning architectures, including:

- ResNet50
- VGG16
- InceptionV3
- InceptionResNetV2
- EfficientNetB0
- DenseNet121
- MobileNetV2

Each model was evaluated using rigorous cross-validation techniques and performance metrics.

### Reproduction Instructions

To reproduce our results:

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the primary classification script:

   ```bash
   python cell_classification.py
   ```

3. For ensemble model evaluation, use:

   ```bash
   python ensemble_cell_classification.py
   ```

4. To analyze model performance across architectures:
   ```bash
   python analyze_completed_models.py
   ```

### Key Findings

Our experiments demonstrate that deep learning models can distinguish between treated cancer cells, untreated cancer cells, and non-cancerous cells with high accuracy. The best-performing models achieved:

- Macro-average precision: >90%
- Macro-average recall: >89%
- Macro-average F1 score: >89%
- Matthews Correlation Coefficient: >0.85

### Visualizations and Analysis

The implementation generates comprehensive performance visualizations:

- ROC curves with AUC scores for each class
- Confusion matrices showing classification performance
- Learning curves showing training and validation metrics
- Model comparison charts for architecture performance analysis

### Citation

If you use this code or methodology in your research, please cite our paper:

```
@article{AuthorLastName2023,
  title={Deep Learning-Based Classification of Oral Cancer Cells},
  author={Author1, Author2, et al.},
  journal={Journal Name},
  year={2023},
  volume={},
  pages={}
}
```

### License

This research code is provided for academic and research purposes only.
