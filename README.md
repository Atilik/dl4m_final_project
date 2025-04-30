# Hit Song Decade Prediction Project _ DL4M-S25
Group10 Yujin Kim, Danielle Jung, Atilay Kucukoglu, Sai Padmapriya Korrapati
____

This project builds a multi-class classification model to predict the decade (1980s, 1990s, 2000s, or 2010s) in which a given song would most likely have been a hit on the Billboard Hot 100 chart, using its audio feature analysis.

## Overview

*The project consists of the following steps:*

- Data collection and preprocessing from popular music datasets (e.g., Billboard Hot 100)
- Feature extraction focusing on audio attributes (e.g., danceability, energy, loudness)
- Multi-class classification model development to predict the decade in which a song would most likely have been a hit
- Evaluation of model performance and interpretation of classification results


## Components
- model.py: Defines the deep learning model architecture.
- utils.py: Utility functions for data processing, model evaluation, etc. Script for extracting audio features from song data.
- Final_roject.ipynb: Main notebook that executes the full workflow (data loading → preprocessing → model training and evaluation).

### Model Structure

### Model Architectures

|---------------------|------------------------------------------------------------------|-----------|----------------------------------|
| Model               | Structure                                                        | Optimizer | Loss Function                    |
|---------------------|------------------------------------------------------------------|-----------|----------------------------------|
| Baseline            | 2 Dense layer                                                    | Adam      | Sparse Categorical Cross-entropy |
| Model 1             | 4 Dense + Batch Normalization + Dropouts (_lr_ = 0.001)          | Adam      | Sparse Categorical Cross-entropy |
| Model 2             | 8 Dense layer + Batch Normalization + Dropouts                   | Adam      | Sparse Categorical Cross-entropy |
| Model 3             | 4 Dense + Batch Normalization + Skip + Dropouts                  | Adam      | Sparse Categorical Cross-entropy |
| Transformer Encoder | Dense + Multi-Head Attention + Layer Normalization + Dropout     | Adam      | Sparse Categorical Cross-entropy |
|---------------------|------------------------------------------------------------------|-----------|----------------------------------|

## Datasets

Dataset Download Link: https://drive.google.com/drive/folders/1CgxukTqugMtWjx4q9T0J3ALrfYSodwVP?usp=sharing

The dataset used in this project was manually created by collecting and combining information from the Billboard Hot 100 charts.
We curated a list of songs and extracted their corresponding audio features, such as danceability, energy, loudness, and others, using public music data sources.

To ensure accurate song identification, we retrieved the MusicBrainz Identifiers (MBIDs) for each unique track using the MusicBrainz API.


Each song is labeled according to the decade or period during which it became a hit.
Instead of a simple binary hit/not-hit label, this task is formulated as a multi-class classification problem, predicting whether a song would most likely succeed in the 1980s, 1990s, 2000s, or 2010s.
 

## Installation
To run this project, you will need Python and a few essential libraries. Install the dependencies with:


1. Clone the repository:

   ```bash
pip install -r requirements.txt
    ```
    or manually install the following packages:
    
    ```bash
    pip install numpy pandas scikit-learn tensorflow matplotlib

    ```
    
**Note:**  
If you are running this notebook on Google Colab, you may need to mount your Google Drive to access external files:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Results

### Baseline
|--------|-----------|--------|----------|---------|
| Decade | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1980s  | 0.57      | 0.82   | 0.67     | 157     |
| 1990s  | 0.43      | 0.21   | 0.28     | 115     |
| 2000s  | 0.40      | 0.19   | 0.25     | 150     |
| 2010s  | 0.62      | 0.80   | 0.70     | 245     |
|--------|-----------|--------|----------|---------|

|--------------|-----------|--------|----------|---------|
| Metric       | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Accuracy     |           |        | 0.57     | 667     |
| Macro Avg    | 0.51      | 0.50   | 0.48     | 667     |
| Weighted Avg | 0.53      | 0.57   | 0.52     | 667     |
|--------------|-----------|--------|----------|---------|

### Model_1
|--------|-----------|--------|----------|---------|
| Decade | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1980s  | 0.69      | 0.62   | 0.65     | 157     |
| 1990s  | 0.40      | 0.38   | 0.39     | 115     |
| 2000s  | 0.37      | 0.64   | 0.47     | 150     |
| 2010s  | 0.82      | 0.53   | 0.65     | 245     |
|--------|-----------|--------|----------|---------|

|--------------|-----------|--------|----------|---------|
| Metric       | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Accuracy     |           |        | 0.55     | 667     |
| Macro Avg    | 0.57      | 0.54   | 0.54     | 667     |
| Weighted Avg | 0.62      | 0.55   | 0.56     | 667     |
|--------------|-----------|--------|----------|---------|

### Model_2
|--------|-----------|--------|----------|---------|
| Decade | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1980s  | 0.60      | 0.73   | 0.66     | 157     |
| 1990s  | 0.41      | 0.46   | 0.43     | 115     |
| 2000s  | 0.41      | 0.61   | 0.49     | 150     |
| 2010s  | 0.87      | 0.45   | 0.59     | 245     |
|--------|-----------|--------|----------|---------|

|--------------|-----------|--------|----------|---------|
| Metric       | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Accuracy     |           |        | 0.55     | 667     |
| Macro Avg    | 0.57      | 0.56   | 0.54     | 667     |
| Weighted Avg | 0.63      | 0.55   | 0.56     | 667     |
|--------------|-----------|--------|----------|---------|

### Model_3
|--------|-----------|--------|----------|---------|
| Decade | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1980s  | 0.65      | 0.64   | 0.64     | 157     |
| 1990s  | 0.38      | 0.47   | 0.42     | 115     |
| 2000s  | 0.42      | 0.52   | 0.47     | 150     |
| 2010s  | 0.78      | 0.59   | 0.67     | 245     |
|--------|-----------|--------|----------|---------|

|--------------|-----------|--------|----------|---------|
| Metric       | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Accuracy     |           |        | 0.57     | 667     |
| Macro Avg    | 0.56      | 0.55   | 0.55     | 667     |
| Weighted Avg | 0.60      | 0.57   | 0.58     | 667     |
|--------------|-----------|--------|----------|---------|

### Transformer Encoder
|--------|-----------|--------|----------|---------|
| Decade | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1980s  | 0.72      | 0.52   | 0.60     | 157     |
| 1990s  | 0.37      | 0.40   | 0.38     | 115     |
| 2000s  | 0.35      | 0.69   | 0.47     | 150     |
| 2010s  | 0.86      | 0.47   | 0.61     | 245     |
|--------|-----------|--------|----------|---------|

|--------------|-----------|--------|----------|---------|
| Metric       | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Accuracy     |           |        | 0.52     | 667     |
| Macro Avg    | 0.57      | 0.52   | 0.52     | 667     |
| Weighted Avg | 0.63      | 0.52   | 0.54     | 667     |
|--------------|-----------|--------|----------|---------|


### Key Findings
- The features used did not offer enough discriminative power. This indicates the model perfomance depends on the data quality.
- Meaningful data is more important than the amount of the data.
- Feature normaliation or scaling affects learning.

## Role (Alphabetic Order)

- Dataset collection : Atilay, Yujin
- Data Preprocessing : Atilay, Yujin
- Build Model : Danielle, Priya
- Evaluation / Demo : Atilay, Danielle, Priya, Yujin
- Documentation : Danielle, Priya


## Acknowledgements

- [Billboard Hot 100 Chart](https://www.billboard.com/charts/hot-100/{date_str}) – Used for collecting hit song data.
- [MusicBrainz API Documentation](https://musicbrainz.org/doc/MusicBrainz_API) – Used for retrieving MBIDs (MusicBrainz Identifiers) of each song.
- [Essentia Streaming Extractor Music](https://essentia.upf.edu/streaming_extractor_music.html) – Used for extracting detailed audio features from music tracks.

We gratefully acknowledge these publicly available resources, which made this project possible.
