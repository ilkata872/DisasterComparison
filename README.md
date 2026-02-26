# Disaster Tweets Analysis: Traditional ML vs. Deep Learning (BERT) 🌪️📱

This Natural Language Processing (NLP) project classifies tweets to determine whether they are announcing a real disaster or not. The main objective is to compare the performance of a traditional Machine Learning baseline against a modern, contextual Deep Learning approach.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Hugging Face `transformers` (BERT)
* **Traditional ML:** Scikit-Learn (`LogisticRegression`, `TfidfVectorizer`, `RandomizedSearchCV`)
* **Data Processing:** Pandas, NumPy

## 🧠 Models Compared

1. **TF-IDF + Logistic Regression:** * A strong, lightweight baseline model.
   * Uses term frequency-inverse document frequency to represent text numerically.
   * Tuned using `RandomizedSearchCV` to find the optimal C parameter.
2. **Fine-Tuned BERT (`bert-base-uncased`):** * A pre-trained Transformer model from Hugging Face.
   * Fine-tuned for 3 epochs using PyTorch to understand the deep semantic context of the tweets rather than just word frequencies.

## 📊 Results & Key Insights

While the overall accuracy difference might seem small, the way the models handle the minority class (actual disasters) is drastically different. 

| Metric (Disaster Class) | TF-IDF + LogReg | Fine-Tuned BERT |
| :--- | :--- | :--- |
| **Overall Accuracy** | 90% | **91%** |
| **Disaster Recall** | 0.61 | **0.78** |
| **Disaster F1-Score** | 0.69 | **0.77** |

**Conclusion:** The TF-IDF model heavily relies on specific keywords and misses many real disaster tweets (low recall of 61%). BERT, by understanding the *context* of words in a sentence, significantly reduces False Negatives, boosting the recall for real disasters to 78%. When human lives are on the line (e.g., real-time disaster monitoring), BERT is the clear winner.

## 💻 How to Run
1. Clone the repository.
2. Install dependencies: `pip install torch transformers scikit-learn pandas kagglehub`
3. Run `logregnlp.ipynb` for the traditional ML baseline.
4. Run `bertnlp.ipynb` (preferably with GPU enabled) for the BERT fine-tuning.