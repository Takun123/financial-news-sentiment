# Financial News Sentiment & Market Signal
 
**Domain:** NLP | Financial Markets | Sentiment Analysis  
**Stack:** Python, scikit-learn, XGBoost, NLTK, TF-IDF  
**Status:** Complete
 
---
 
## Business Problem
 
Traders and financial analysts process hundreds of news headlines daily. Manually assessing the sentiment of each headline is slow, inconsistent, and subject to emotional bias. This project builds a sentiment classification model that automatically labels financial news headlines as **Bullish (Positive)**, **Bearish (Negative)**, or **Neutral** — providing a supplementary signal alongside traditional financial indicators to inform buy, hold, or sell decisions.
 
The goal is not to replace human judgment but to give analysts a fast, explainable first-pass signal they can act on or override based on their own conviction.
 
---
 
## Dataset
 
Two sources were combined to build the training dataset:
 
| Source | Rows | Labeling Method |
|--------|------|-----------------|
| Financial PhraseBank (sentences_allagree) | 4,846 | Human expert labels (full annotator agreement) |
| Yahoo Finance RSS Headlines | 87 | FinBERT auto-labeling (confidence threshold ≥ 0.75) |
| **Merged & deduplicated** | **4,922** | — |
 
**Financial PhraseBank** was chosen as the primary source for its label quality — only sentences where all annotators agreed were included, making it the highest-confidence subset available.
 
**Yahoo Finance RSS** was scraped using feedparser across four tickers (AAPL, TSLA, MSFT, General). FinBERT was used to auto-label headlines; rows below 0.75 confidence were excluded to maintain label integrity.
 
### Label Distribution (post-deduplication)
 
| Label | Count | % |
|-------|-------|---|
| Neutral | 2,924 | 59% |
| Positive | 1,370 | 28% |
| Negative | 621 | 13% |
 
Class imbalance confirmed — negative class is underrepresented. Handled via class weights in modeling.
 
### Data Quality Issues Identified & Resolved
- 11 duplicate rows removed
- 4 fragment titles (fewer than 4 words) dropped — insufficient context for sentiment signal
- No null values or empty strings in final dataset
---
 
## Methodology
 
### Text Preprocessing (`03_preprocessing.ipynb`)
1. Drop fragment titles (< 4 words)
2. Lowercase all text
3. Remove punctuation and special characters
4. Keep numbers — financial figures carry sentiment signal (e.g. "down 12%")
5. Keep stop words — removing them risks destroying negation context ("not growing" → "growing")
6. Lemmatization — reduces word variants to base form without producing invalid tokens
7. TF-IDF vectorization (5,000 features, unigrams + bigrams) — rewards rare, domain-specific financial terms over generic vocabulary
### Modeling (`04_modeling.ipynb`)
 
Two models trained with class weights to address negative class imbalance:
 
| Model | Negative Recall | Macro F1 | Macro ROC-AUC |
|-------|----------------|----------|----------------|
| Logistic Regression | **0.726** | 0.720 | 0.883 |
| XGBoost | 0.694 | 0.725 | 0.883 |
 
**Selected model: Logistic Regression**
 
Reasoning: Both models achieve identical ROC-AUC (0.883). Logistic Regression catches 4 more bearish signals out of 124 (90 vs 86). In a trading context, missing a bearish signal means buying into a losing position — a recoverable false alarm is less costly than an undetected loss signal. Logistic Regression also provides full coefficient-level explainability, which is critical for regulatory compliance and trader trust.
 
---
 
## Explainability (`05_explainability.ipynb`)
 
Because this model serves financial professionals, explainability is not optional. Traders need to understand *why* a headline was flagged as bearish — not just that it was.
 
Using logistic regression coefficients, the top words driving each class are:
 
**Bearish signals:** down, decreased, fell, cut, lower, layoff, loss, declined  
**Bullish signals:** increased, up, rose, grew, signed, improved, growth, agreement  
**Neutral signals:** approximately, range, stake, development, business, include
 
These word lists are financially coherent and interpretable to any analyst without technical background.
 
---
 
## Known Limitations
 
| Limitation | Impact | Potential Fix |
|------------|--------|---------------|
| Only 621 negative training samples | Model misses informal bearish vocabulary ("losses", "layoffs" underweighted) | Expand dataset with more bearish financial news |
| Context-blind TF-IDF | "Cut costs" (positive) treated same as "cut dividend" (negative) | Sentence-level embeddings (FinBERT fine-tuned end-to-end) |
| Negation partially unresolved | "Not growing" still partially misclassified | Negation handling — convert "not X" to "not_X" before vectorization |
| RSS data only 87 rows | Model underexposed to short headline style | Expand live RSS collection over weeks, not a single scrape |
 
---
 
## Project Structure
 
```
financial-news-sentiment/
├── data/
│   ├── raw/                  # Original scraped and downloaded data
│   └── processed/            # Cleaned and merged datasets
├── models/                   # Saved model files (.pkl)
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_explainability.ipynb
├── reports/                  # Visualizations and output charts
└── README.md
```
 
---
 
## Results Summary
 
- **Best model:** Logistic Regression with TF-IDF features
- **Negative (Bearish) Recall: 0.726** — catches 90 out of 124 bearish headlines
- **Macro ROC-AUC: 0.883** — strong discriminative ability across all three classes
- **Explainability:** Full coefficient-level word importance available per prediction
- **Production gap:** Negative recall at 0.726 means 1 in 4 bearish signals is missed — acceptable as a supplementary signal, not as a standalone trading system