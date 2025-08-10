


# Detecting AI-Generated Text – SEA820 NLP Final Project

## 📌 Project Overview
This project tackles the challenge of distinguishing between **human-written** and **AI-generated** text, a problem with important implications for **academic integrity**, **misinformation detection**, and **content moderation**.  
We implemented and compared:
1. **Baseline Model** – Logistic Regression with TF-IDF features.
2. **Transformer Model** – Fine-tuned DistilBERT using Hugging Face Transformers.

---

## 🎯 Objectives
- **Implement and Compare Models**: Build a baseline classic ML model and a modern Transformer model.
- **Conduct Rigorous Evaluation**: Go beyond accuracy by using **precision, recall, and F1-score**.
- **Perform Detailed Error Analysis**: Investigate misclassifications and identify improvement areas.
- **Analyze Ethical Implications**: Discuss potential misuse and bias.
- **Time & Resource Considerations**: Compare training times and scalability.

---

## 📂 Repository Structure
```

.
├── data/                          # Train, validation, and test CSV files
├── scripts/                       # Python scripts for training, evaluation, and error analysis
│   ├── train\_baseline.py
│   ├── train\_transformer.py
│   ├── report\_metrics.py
│   ├── error\_analysis.py
├── outputs/                       # Saved models, metrics, and plots
│   ├── baseline\_confusion\_valid.png
│   ├── transformer\_confusion\_validation.png
│   ├── transformer\_confusion\_test.png
│   ├── transformer\_metrics.csv
│   ├── metrics\_comparison.csv
│   ├── validation\_misclassified.csv
│   ├── validation\_top50\_confident\_wrongs.csv
├── final\_project\_report\_full.md    # Final detailed report
├── requirements.txt                # Python dependencies
└── README.md                       # This file

````

---

## ⚙️ Setup & Installation
### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-text-detection.git
cd ai-text-detection
````

### 2. Create a virtual environment

```bash
conda create -n sea820 python=3.9
conda activate sea820
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. 📥 Dataset Setup (Kaggle)

This project uses the **AI vs Human Text** dataset from Kaggle.  
To download it, you need a Kaggle account and an API key.

### 1. Get your Kaggle API key
1. Log in to [Kaggle](https://www.kaggle.com/).
2. Go to **Account Settings** → **Create New API Token**.
3. This will download a file called `kaggle.json`.

### 2. Place the API key
Move the `kaggle.json` file to your system's Kaggle configuration folder:
```bash
# Linux / macOS
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/

# Windows (PowerShell)
mkdir $HOME\.kaggle
mv kaggle.json $HOME\.kaggle\

---

## 🚀 Usage

### Train the Baseline Model

```bash
python scripts/train_baseline.py
```

### Train the Transformer Model

```bash
python scripts/train_transformer.py
```

### Evaluate Models & Generate Metrics

```bash
python scripts/report_metrics.py
```

### Perform Error Analysis

```bash
python scripts/error_analysis.py
```

---

## 📊 Results Summary

| Model           | Accuracy | Precision | Recall | F1-Score | Training Time |
| --------------- | -------- | --------- | ------ | -------- | ------------- |
| **Baseline**    | X.XX     | X.XX      | X.XX   | X.XX     | **\~2-3 min** |
| **Transformer** | X.XX     | X.XX      | X.XX   | X.XX     | **\~1.5 hrs** |

*The Transformer significantly outperforms the baseline across all metrics, but requires more computational resources.*

---

## 🔍 Error Analysis

* **False Positives**: AI text predicted as human-written (often very well-written AI essays).
* **False Negatives**: Human text predicted as AI-generated (common for short or repetitive writing).
* **Examples**: See [`validation_misclassified.csv`](outputs/validation_misclassified.csv).

---

## ⚖️ Ethical Considerations

* **Bias Risk**: May flag non-native English speakers unfairly.
* **Misuse Risk**: Could be used to censor legitimate content.
* **Transparency**: Clear disclaimers needed for deployment in academic and journalistic settings.

---

## 🛠️ Technologies Used

* **Python 3.9**
* **scikit-learn** for baseline model
* **Hugging Face Transformers** for DistilBERT
* **Pandas**, **Matplotlib**, **Seaborn** for analysis & visualization

---

## 👥 Authors

* **Fahad Ali Khan**

---

