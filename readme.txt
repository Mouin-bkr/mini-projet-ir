Information Retrieval (IR) system. Processes the Reuters-21578 corpus (7,770 economic news articles across 90 categories like earn, acq, crude, grain) to build an inverted index, TF-IDF vectors, and ranking pipeline.​

Key Results: MAP = 0.848 (outperforms typical Reuters baselines ~0.4-0.6), P@10 = 0.77. Ablation study shows stemming + stopwords optimal (+0.4% MAP, -15% vocabulary). Includes Rocchio feedback and evaluation on 8 queries (cocoa, oil, earnings, etc.).​

Project Overview
Inverted Index: 5,895 economic terms → (doc_id, TF) postings (JSON, 2.1MB)

TF-IDF Model: 500 docs × 2,828 features, L2-normalized, min_df=2

Ranking: Cosine similarity for top-k results (e.g., "cocoa prices" P@10=1.00)

Evaluation: Precision@10, Recall@10, AP, MAP on 7 queries (perfect on cocoa/acq)

Ablation: 4 configs tested (stops ON/OFF, stemming ON/OFF) – stops+stemming best

Feedback: Rocchio algorithm (α=1.0, β=0.75, γ=0.15) – minor AP improvement (-4.4% on oil query)

Project Structure 

  mini-projet-ir/
  ├── mini_projet1.py          # Main IR pipeline (indexing, TF-IDF, ranking, eval)
  ├── MiniProjet1_IR.ipynb     # Jupyter notebook with analysis & plots
  ├── inverted_index.json      # 5,895 terms index
  ├── evaluation_metrics.csv   # P@10=0.77, MAP=0.848 results
  ├── ablation_results.csv     # Preprocessing ablation (4 configs)
  ├── requirements.txt         # Dependencies (NLTK, scikit-learn, pandas)
  └── .gitignore               # Excludes .idea, __pycache__

Installation

  # Clone repo
  git clone https://github.com/mouinbkr/mini-projet-ir.git
  cd mini-projet-ir
  
  # Virtual environment
  python -m venv venv
  source venv/bin/activate  # Linux/Mac
  # venv\Scripts\activate    # Windows
  
  # Install dependencies
  pip install -r requirements.txt
  
  # Download NLTK data (Reuters corpus, stopwords, tokenizer)
  python -c "import nltk; nltk.download(['reuters', 'punkt_tab', 'stopwords'])"

Execution 
  
  # Run main script (full pipeline, ~5min for 500 docs)
  python mini_projet1.py
  
  # Or open Jupyter notebook for interactive analysis
  jupyter notebook MiniProjet1_IR.ipynb


Author & Contact

  Mouin Boubakri - Final-year CS student, ISAMM Manouba
  Portfolio: mouinboubakri.tech
  GitHub: mouin-bkr
  LinkedIn: Mohamed Mouin Boubakri
