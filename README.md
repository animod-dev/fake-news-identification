# Fake News Identification

A machine-learning project that classifies news articles as **real** or **fake** using Natural Language Processing (NLP). It uses text preprocessing, stemming, TF-IDF feature extraction, and a Logistic Regression classifier.

The repository includes both a Python script and a Jupyter Notebook so the workflow can be run as a script or explored step by step.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [Results](#results)
- [Project Structure](#project-structure)
- [Limitations and Notes](#limitations-and-notes)
- [Possible Improvements](#possible-improvements)
- [Dataset Attribution](#dataset-attribution)

## Overview

Fake news can spread quickly online, making it difficult for readers to distinguish reliable reporting from misleading information. This project demonstrates a basic text-classification approach to fake-news identification using a labelled news dataset.

The model learns patterns from the **author and title** of each news item. It does not currently use the full article body, even though the dataset includes a `text` column.

### Objectives

- Prepare and clean news-related text for machine learning.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classifier on labelled examples.
- Evaluate the model using training and test accuracy.
- Demonstrate a prediction for an example from the held-out test set.

## How It Works

The current implementation follows this workflow:

1. **Load the dataset** — read `train.csv` into a pandas DataFrame.
2. **Inspect and clean data** — review the dataset shape and missing values, then replace missing values with empty strings.
3. **Build the text feature** — combine each record's `author` and `title` into one `content` field.
4. **Preprocess the text** — remove non-alphabetic characters, convert text to lowercase, split it into words, remove English stopwords, and apply Porter stemming.
5. **Vectorize the text** — use `TfidfVectorizer` to convert preprocessed text into numerical features.
6. **Split the data** — create training and test sets with a 60/40 split, stratified by label, using `random_state=2`.
7. **Train the model** — fit a `LogisticRegression` classifier on the training data.
8. **Evaluate the model** — calculate accuracy on the training and test sets.
9. **Demonstrate a prediction** — classify one example already present in the test set and print its actual label for comparison.

## Technology Stack

- **Python** — programming language
- **pandas** — dataset loading and manipulation
- **NumPy** — numerical computing
- **NLTK** — English stopwords and Porter stemming
- **scikit-learn** — TF-IDF vectorization, train/test split, Logistic Regression, and accuracy evaluation
- **Matplotlib** — imported in the existing code; no plot is currently produced
- **Jupyter Notebook** — interactive execution and inspection of the workflow

## Dataset

This project is designed for the `train.csv` file from Kaggle's **Fake News** competition dataset:

**Dataset page:** https://www.kaggle.com/competitions/fake-news/data?select=train.csv

The notebook records a dataset size of **20,800 rows and 5 columns**.

| Column | Description |
|---|---|
| `id` | Unique identifier for a news article |
| `title` | Headline of the article |
| `author` | Author of the article |
| `text` | Article text; some entries may be incomplete |
| `label` | Target class: `1` = fake news, `0` = real news |

The code uses the `author`, `title`, and `label` columns. The `text` and `id` columns are not used to train the current model.

### Prepare the dataset

1. Download `train.csv` from the Kaggle page above. You may need to sign in to Kaggle and accept the dataset/competition terms.
2. Place `train.csv` in the project directory.
3. Update the CSV path in the Python script and notebook as described in [Getting Started](#getting-started).

The dataset is not included in this repository; download it from its source and follow the source's terms of use.

## Getting Started

### Prerequisites

- Python 3.10 or later is recommended.
- `pip` for installing Python packages.
- The `train.csv` dataset described above.

### 1. Clone the repository

```bash
git clone https://github.com/anmo-02/fake-news-identification.git
cd fake-news-identification
```

### 2. Create and activate a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install numpy pandas nltk scikit-learn matplotlib jupyter
```

### 4. Download NLTK stopwords

The code accesses the NLTK English stopwords corpus. Download it once in your environment:

```bash
python -m nltk.downloader stopwords
```

### 5. Configure the dataset path

The current script and notebook contain a machine-specific Windows path similar to:

```python
news_dataset = pd.read_csv(r'C:\Users\msani\Downloads\train.csv')
```

Change that line in **both** `fake-news-identification.py` and `fake-news-identification.ipynb` to point to the location of your downloaded dataset. If `train.csv` is in the repository directory, use:

```python
news_dataset = pd.read_csv("train.csv")
```

## Running the Project

### Option A: Run the Python script

From the project directory, after completing the setup above, run:

```bash
python fake-news-identification.py
```

The script prints dataset/preprocessing information, the training and test accuracy scores, and a prediction for one sample from the test set alongside that sample's actual label.

### Option B: Run the Jupyter Notebook

Start Jupyter from the project directory:

```bash
jupyter notebook
```

Open `fake-news-identification.ipynb` and run the cells from top to bottom. Ensure the dataset path is correct before running the data-loading cell.

## Results

The saved notebook output reports the following accuracy values:

| Evaluation set | Accuracy shown in the notebook |
|---|---:|
| Training set | 98.49% |
| Test set | 97.12% |

These numbers are the outputs stored in the notebook, not a guarantee of performance on new data. Results can differ when the dataset, package versions, or implementation changes. See [Limitations and Notes](#limitations-and-notes), especially the note about fitting TF-IDF before splitting the data.

## Project Structure

```text
fake-news-identification/
├── README.md
├── fake-news-identification.py
└── fake-news-identification.ipynb
```

`train.csv` must be downloaded separately and placed in the project directory (or referenced by an updated path).

## Limitations and Notes

- **Uses headlines and author names only.** The model does not analyse the article body, source reputation, cited evidence, or corroborating reports.
- **Potential data leakage in the current implementation.** `TfidfVectorizer` is fitted on the full dataset before the train/test split. This allows information from the test set to influence the vocabulary/IDF statistics and can make the reported test score optimistic. For a more reliable evaluation, split the raw text first and fit the vectorizer on the training text only, ideally by using a scikit-learn `Pipeline`.
- **The prediction example is fixed to a test sample.** The existing code predicts `X_test[3]`; it does not currently provide a function or command-line interface for entering an arbitrary news headline.
- **Accuracy is only one metric.** Precision, recall, F1-score, a confusion matrix, and evaluation on a separate dataset would provide a more complete picture.
- **This is a classification demonstration, not a fact-checking service.** A model prediction is not proof that a claim is true or false. Verify important claims using trustworthy, independent sources.

## Possible Improvements

- Fit TF-IDF only on the training split to prevent data leakage.
- Use a reusable preprocessing and prediction function that accepts new headlines or article text.
- Compare headline-only features with headline-plus-body features.
- Evaluate with precision, recall, F1-score, and a confusion matrix.
- Try alternative models and tune hyperparameters using cross-validation.
- Save the trained model and vectorizer for reuse without retraining each time.
- Add input validation, a dependency file, and automated tests.

## Dataset Attribution

The project is intended to use the [Kaggle Fake News competition dataset](https://www.kaggle.com/competitions/fake-news/data?select=train.csv). Please review and comply with Kaggle's applicable terms and dataset conditions. No separate software license is specified here; add a `LICENSE` file if you intend to publish the project under a particular open-source license.
