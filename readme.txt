# 📊 Mini-Projet 1: Système de Recherche d'Information Économique
## Reuters-21578 IR Pipeline - ISAMM Manouba (MAP = 0.848)

[![ISAMM Manouba](https://img.shields.io/badge/ISAMM-Manouba-blue)](https://isa2m.rnu.tn)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-orange)](https://www.nltk.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-blue)](https://scikit-learn.org)

Système de Recherche d'Information (RI) complet sur le corpus Reuters-21578, implémentant indexation inversée, modélisation TF-IDF, classement cosinus, évaluation IR (P@10, MAP), feedback Rocchio, et étude d'ablation des prétraitements. Projet universitaire ISAMM Manouba - Techniques d'Indexation et de Référencement (Pr. Chiraz Trabelsi).

## 🎯 Objectifs Pédagogiques
- [x] **Index Inversé** : 5,895 termes économiques → (doc_id, TF) postings (JSON)
- [x] **TF-IDF** : 500 docs × 2,828 features, L2-normalisé, min_df=2
- [x] **Requêtes** : 8 thématiques économiques (cocoa, oil, earn, acq...)
- [x] **Évaluation** : **MAP = 0.848**, P@10 = 0.77 (supérieur baseline Reuters ~0.6)
- [x] **Rocchio** : Feedback pertinence (+/-0.037 AP sur "oil market")
- [x] **Ablation** : Stops + stemming optimal (MAP +0.4%, vocab -15%)

## 📈 Résultats Clés
| Requête | Pertinents | P@10 | R@10 | AP | Top-3 Docs |
|---------|------------|------|------|----|------------|
| cocoa prices | 15 | **1.00** | 0.67 | 1.00 | [318,489,319] |
| acquisition deal | 81 | **1.00** | 0.12 | 1.00 | [323,69,430] |
| oil market | 33 | 0.90 | 0.27 | 0.84 | [49,208,122] |
| company earnings | 167 | 0.80 | 0.05 | 0.64 | [382,381,285] |
| **MOYENNE (MAP)** | | **0.848** | | | |

**Ablation Study**: Stops + stemming = MAP 0.762 (meilleur), vocab 1,086 vs 1,282 baseline (-15%) [chart:184].

## 🛠️ Technologies
- **Corpus**: Reuters-21578 (NLTK) - 7.7k docs économiques, 90 catégories
- **Indexation**: Python defaultdict, NLTK tokenization
- **Vectorisation**: scikit-learn TF-IDF (L2-norm, min_df=2)
- **Ranking**: Cosine similarity, NumPy arg-sort
- **Évaluation**: P@10, R@10, AP, MAP (pandas)
- **Visualisation**: matplotlib (ablation plots)
- **Gestion**: Git, GitHub, Jupyter Notebook

## 📁 Structure du Projet
