# Mini-Projet IR: Reuters-21578 Search Engine

[![Python](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-orange)](https://www.nltk.org)
[![Scikit-learn](https://img.shields.io/badge/scikit-learn-1.3-blue)](https://scikit-learn.org)
[![ISAMM Manouba](https://img.shields.io/badge/ISAMM-Manouba-blue)](https://isa2m.rnu.tn)

**Information Retrieval (IR) system** built for ISAMM Manouba's Indexing & Referencing course. Processes the Reuters-21578 corpus (7,770 economic news articles across 90 categories like earn, acq, crude, grain) to build an inverted index, TF-IDF vectors, and ranking pipeline [web:486][web:3].

**Key Results**: MAP = 0.848 (outperforms typical Reuters baselines ~0.4-0.6), P@10 = 0.77. Ablation study shows stemming + stopwords optimal (+0.4% MAP, -15% vocabulary). Includes Rocchio feedback and evaluation on 8 queries (cocoa, oil, earnings, etc.) [web:73].

## Project Overview

- **Inverted Index**: 5,895 economic terms → (doc_id, TF) postings (JSON, 2.1MB)
- **TF-IDF Model**: 500 docs × 2,828 features, L2-normalized, min_df=2
- **Ranking**: Cosine similarity for top-k results (e.g., "cocoa prices" P@10=1.00)
- **Evaluation**: Precision@10, Recall@10, AP, MAP on 7 queries (perfect on cocoa/acq)
- **Ablation**: 4 configs tested (stops ON/OFF, stemming ON/OFF) – stops+stemming best
- **Feedback**: Rocchio algorithm (α=1.0, β=0.75, γ=0.15) – minor AP improvement (-4.4% on oil query)

**Performance Highlights**:
| Query | P@10 | Relevant Docs | Top Categories |
|-------|------|---------------|----------------|
| cocoa prices | 1.00 | 15 | cocoa |
| acquisition | 1.00 | 81 | acq |
| oil market | 0.90 | 33 | crude, oilseed |
| earnings | 0.80 | 167 | earn |
| **MAP** | **0.848** | - | - |

![Ablation Results](ablation_analysis.png)
*Stemming + stopwords: +0.4% MAP, -15% vocab [chart:466]*

## Project Structure

mini-projet-ir/
├── mini_projet1.py # Main IR pipeline (indexing, TF-IDF, ranking, eval)
├── MiniProjet1_IR.ipynb # Jupyter notebook with analysis & plots
├── inverted_index.json # 5,895 terms index
├── evaluation_metrics.csv # P@10=0.77, MAP=0.848 results
├── ablation_results.csv # Preprocessing ablation (4 configs)
├── requirements.txt # Dependencies (NLTK, scikit-learn, pandas)
└── .gitignore # Excludes .idea, pycache


## Setup & Run

### Prerequisites
- Python 3.12+

### Installation

Clone repo
git clone https://github.com/mouinbkr/mini-projet-ir.git
cd mini-projet-ir

Virtual environment
python -m venv venv
source venv/bin/activate # Linux/Mac

venv\Scripts\activate # Windows
Install dependencies
pip install -r requirements.txt

Download NLTK data (Reuters corpus, stopwords, tokenizer)
python -c "import nltk; nltk.download(['reuters', 'punkt_tab', 'stopwords'])"


### Execution

Run main script (full pipeline, ~5min for 500 docs)
python mini_projet1.py

Or open Jupyter notebook for interactive analysis
jupyter notebook MiniProjet1_IR.ipynb


**Expected Output**:
- Loaded 500 docs, 60 categories
- Built index: 5,895 terms
- TF-IDF matrix: 500 × 2,828 (98% sparse)
- Results: MAP=0.848, ablation analysis saved
- Plots: Evaluation metrics & preprocessing comparison

**Full run time**: 5min (500 docs). For complete corpus (7.7k docs): ~30min.

## Usage Example
In mini_projet1.py or notebook
query = "oil market" # Example query
results = rank_documents(query, tfidf_model, index, top_k=10)
print(f"Top result: {results['title']} (score: {results['score']:.3f})")

Evaluate: P@10=0.90 for oil


## Author & Contact
**Mouin Boubakri** - Final-year CS student, ISAMM Manouba  
Portfolio: [mouinboubakri.tech](https://mouinboubakri.tech)  
GitHub: [mouin-bkr](https://github.com/mouin-bkr)  
LinkedIn: [Mohamed Mouin Boubakri](https://linkedin.com/in/mohamed-mouin-boubakri)

*Academic project for ISAMM Manouba - MAP=0.848 on Reuters-21578. Open to collaborations in IR/ML/DevOps.*
