# Assignment 5: From Supervised Learning to Self-Supervised Learning

## Overview

This assignment implements SimCLR for self supervised representation learning on CIFAR10 dataset using PyTorch.

The assignment includes:

- Supervised baseline training using 10% labeled data
- SimCLR data augmentation
- Similarity analysis before & after training
- SimCLR contrastive pretraining
- Linear probe evaluation
- Fine-tuning evaluation
- PCA feature visualization
- Confusion matrix generation
- Metrics and prediction export

## Environment

- Python 3.10+
- torch
- torchvision
- torchaudio
- numpy
- matplotlib
- scikit-learn
- pandas
- tqdm

Install dependencies using:

```
pip install -r requirements.txt
```

## Running the Experiments

### Supervised Baseline

```
python msds24031_05_task1_supervised.py
```

### Augmentation Examples

```
python msds24031_05_task2_augmentations.py
```

### Similarity Analysis Before Training

```
python msds24031_05_task3_similarity.py
```

### SimCLR Pretraining

```
python msds24031_05_task5_simclr_pretraining.py
```

### Similarity Analysis After Training

```
python msds24031_05_task5_simclr_evaluating.py
```

### Linear Probe Evaluation

```
python msds24031_05_task6_linear_probe.py
```

### Fine-Tuning

```
python msds24031_05_task7_finetune.py
```

### Feature Visualization

```
python msds24031_05_task8_visualization.py
```

## Summary

Results demonstrate that the SimCLR successfully learns meaningful visual representations from the unlabeled CIFAR10 images. The pretrained encoder achieved 74.66% accuracy using frozen linear classifier, and 81.15% accuracy after fine-tuning, significantly outperforming supervised baseline trained with only 10% labeled data (45.87%).

Experiments show that self supervised contrastive learning can learn useful feature representations without labels, and provide strong initialization for downstream classification tasks.