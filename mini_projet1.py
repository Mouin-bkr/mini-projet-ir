"""
 Mini-Projet 1: Economic News Retrieval - Reuters-21578 IR Pipeline
Implements: Inverted Index, TF-IDF, Cosine Ranking, Precision/Recall/MAP, Rocchio Feedback, Ablation Study
Author: Mohamed mouin boubakri - Mahdi ben Ali
Date: November 20, 2025
"""

import nltk
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, reuters
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# One-time downloads
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('reuters', quiet=True)

# Global preprocessing setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# =============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_reuters_corpus(n_docs=500):
    """Load Reuters-21578 training documents and categories."""
    train_ids = [fid for fid in reuters.fileids() if not fid.startswith('test')][:n_docs]
    docs = [reuters.raw(fid) for fid in train_ids]
    all_categories = [reuters.categories(fid) for fid in train_ids]
    print(f"Loaded {len(docs)} documents with {len(set(sum(all_categories, [])))} unique categories")
    return docs, train_ids, all_categories


def preprocess_document(doc, remove_stops=True, apply_stemming=False):
    """Clean and tokenize document using specified preprocessing options."""
    tokens = word_tokenize(doc.lower())
    tokens = [token for token in tokens if token.isalpha() and len(token) >= 3]
    if remove_stops:
        tokens = [token for token in tokens if token not in stop_words]
    if apply_stemming:
        tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)


# Load corpus (scale to 500 docs for realistic evaluation)
docs, train_ids, all_categories = load_reuters_corpus(n_docs=500)
clean_docs = [preprocess_document(doc) for doc in docs]


# =============================================================================
# SECTION 2: INVERTED INDEX CONSTRUCTION (TASK 1)
# =============================================================================

def build_inverted_index(documents, doc_ids):
    """Build inverted index mapping terms to (doc_id, term_frequency) postings."""
    index = defaultdict(list)
    for doc_idx, doc in enumerate(documents):
        doc_id = doc_ids[doc_idx]
        tokens = word_tokenize(doc.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) >= 3]
        term_freq = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1
        for term, freq in term_freq.items():
            index[term].append((doc_id, freq))
    return index


# Build and save inverted index
inverted_index = build_inverted_index(docs, train_ids)
json_index = {term: [(doc_id, freq) for doc_id, freq in postings]
              for term, postings in inverted_index.items()}
with open('inverted_index.json', 'w') as f:
    json.dump(json_index, f, indent=2)
print(f"Inverted index saved: {len(inverted_index)} unique terms")


# =============================================================================
# SECTION 3: TF-IDF VECTORIZATION (TASK 2)
# =============================================================================

def create_tfidf_model(clean_documents, max_features=5000, min_df=2):
    """Create TF-IDF vectorizer with baseline preprocessing."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        min_df=min_df,
        max_features=max_features,
        norm='l2'
    )
    tfidf_matrix = vectorizer.fit_transform(clean_documents)
    return vectorizer, tfidf_matrix


# Baseline TF-IDF model
baseline_vectorizer, X_baseline = create_tfidf_model(clean_docs)
print(f"Baseline TF-IDF: {X_baseline.shape[0]} docs, {X_baseline.shape[1]} features")

# Save vocabulary
vocab = baseline_vectorizer.get_feature_names_out()
with open('tfidf_vocabulary.txt', 'w') as f:
    f.write('\n'.join(vocab))
print("TF-IDF vocabulary saved")

# =============================================================================
# SECTION 4: QUERY PROCESSING AND RANKING (TASK 3)
# =============================================================================

# Define 8 representative economic queries mapped to Reuters categories
QUERIES = [
    "cocoa prices", "oil market", "company earnings", "acquisition deal",
    "trade balance", "foreign exchange", "grain crop", "interest rates"
]

# Category mappings for relevance evaluation
QUERY_TO_CATEGORIES = {
    "cocoa prices": ['cocoa'],
    "oil market": ['crude', 'oilseed'],
    "company earnings": ['earn'],
    "acquisition deal": ['acq'],
    "trade balance": ['trade'],
    "foreign exchange": ['fx', 'currency'],
    "grain crop": ['grain', 'wheat', 'corn', 'soybean'],
    "interest rates": ['money-supply', 'interest', 'livestock']
}


def rank_documents(query, vectorizer, tfidf_matrix, k=10):
    """Rank documents by cosine similarity to query."""
    query_clean = preprocess_document(query)
    query_vector = vectorizer.transform([query_clean])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[-k:][::-1]
    return top_indices, similarities[top_indices]


def get_relevant_documents(query, categories, document_categories):
    """Get document indices relevant to query based on category matching."""
    target_cats = QUERY_TO_CATEGORIES.get(query, [])
    relevant_docs = set()
    for doc_idx, doc_cats in enumerate(document_categories):
        if any(cat in doc_cats for cat in target_cats):
            relevant_docs.add(doc_idx)
    return list(relevant_docs)


# Process and rank all queries
query_results = {}
for query in QUERIES:
    top_docs, scores = rank_documents(query, baseline_vectorizer, X_baseline, k=10)
    query_results[query] = {'top_documents': top_docs.tolist(), 'scores': scores.tolist()}
    print(f"\n{query}: Top-3 documents {top_docs[:3].tolist()} (scores {scores[:3]})")

print("\nQuery rankings completed and stored")


# =============================================================================
# SECTION 5: EVALUATION METRICS (TASK 4)
# =============================================================================

def calculate_metrics(top_docs, relevant_docs, k=10):
    """Compute Precision@k, Recall@k, and Average Precision."""
    if not relevant_docs:
        return {'P@10': 0.0, 'R@10': 0.0, 'AP': 0.0}

    top_k = top_docs[:k]
    relevant_in_top = len(set(top_k) & set(relevant_docs))

    # Precision@k
    precision_k = relevant_in_top / k

    # Recall@k
    recall_k = relevant_in_top / len(relevant_docs)

    # Average Precision
    precisions_at_ranks = []
    for i, doc in enumerate(top_docs[:k], 1):
        if doc in relevant_docs:
            prec_at_i = len(set(top_docs[:i]) & set(relevant_docs)) / i
            precisions_at_ranks.append(prec_at_i)

    ap = np.mean(precisions_at_ranks) if precisions_at_ranks else 0.0

    return {'P@10': precision_k, 'R@10': recall_k, 'AP': ap}


# Evaluate all queries
evaluation_metrics = {}
total_ap = 0
valid_queries = 0

for query in QUERIES:
    relevant_docs = get_relevant_documents(query, all_categories, all_categories)
    top_docs = query_results[query]['top_documents']

    if len(relevant_docs) > 0:
        metrics = calculate_metrics(top_docs, relevant_docs)
        evaluation_metrics[query] = {
            'relevant_count': len(relevant_docs),
            **metrics
        }
        total_ap += metrics['AP']
        valid_queries += 1
        print(f"{query}: {len(relevant_docs)} relevant, P@10={metrics['P@10']:.3f}, AP={metrics['AP']:.3f}")
    else:
        evaluation_metrics[query] = {'relevant_count': 0, 'P@10': 0.0, 'R@10': 0.0, 'AP': 0.0}
        print(f"{query}: No relevant documents found")

# Calculate Mean Average Precision (MAP)
map_score = total_ap / valid_queries if valid_queries > 0 else 0.0
print(f"\nMean Average Precision (MAP): {map_score:.3f}")

# Save evaluation results
metrics_df = pd.DataFrame(evaluation_metrics).T
metrics_df['MAP'] = map_score
metrics_df.to_csv('evaluation_metrics.csv', index_label='Query')
print("Evaluation metrics saved to evaluation_metrics.csv")


# =============================================================================
# SECTION 6: ROCHIO RELEVANCE FEEDBACK (TASK 5)
# =============================================================================

def apply_rocchio_feedback(query_vector, top_documents, tfidf_matrix, alpha=1.0, beta=0.75, gamma=0.15):
    """Apply Rocchio relevance feedback to refine query vector."""
    n_docs = tfidf_matrix.shape[0]

    # Select positive (top 3) and negative (bottom 3) documents
    pos_docs = tfidf_matrix[top_documents[:3]].toarray()  # Convert to dense
    neg_docs = tfidf_matrix[top_documents[-3:]].toarray()

    # Original query as dense array
    q_dense = query_vector.toarray()

    # Rocchio formula: alpha * q + beta * avg_pos - gamma * avg_neg
    pos_mean = np.mean(pos_docs, axis=0)
    neg_mean = np.mean(neg_docs, axis=0)

    q_new_dense = (alpha * q_dense + beta * pos_mean - gamma * neg_mean)

    # Normalize
    norm = np.linalg.norm(q_new_dense)
    if norm > 1e-10:
        q_new_dense = q_new_dense / norm

    # Convert back to sparse matrix
    q_new_sparse = csr_matrix(q_new_dense)
    return q_new_sparse


# Test Rocchio on sample query
sample_query = "oil market"
print(f"\n=== ROCHIO FEEDBACK: {sample_query} ===")

# Original ranking
q_vec_orig = baseline_vectorizer.transform([preprocess_document(sample_query)])
sims_orig = cosine_similarity(q_vec_orig, X_baseline).flatten()
top_docs_orig = np.argsort(sims_orig)[-10:][::-1]

# Apply feedback
q_vec_feedback = apply_rocchio_feedback(q_vec_orig, top_docs_orig, X_baseline)

# New ranking
sims_feedback = cosine_similarity(q_vec_feedback, X_baseline).flatten()
top_docs_feedback = np.argsort(sims_feedback)[-5:][::-1]

print(f"Original top-5: {top_docs_orig[:5].tolist()}")
print(f"Feedback top-5: {top_docs_feedback.tolist()}")

# Evaluate improvement
relevant_docs = get_relevant_documents(sample_query, all_categories, all_categories)
orig_metrics = calculate_metrics(top_docs_orig, relevant_docs)
feedback_metrics = calculate_metrics(top_docs_feedback, relevant_docs)

print(f"Original AP: {orig_metrics['AP']:.3f} -> Feedback AP: {feedback_metrics['AP']:.3f}")
print(f"Improvement: {feedback_metrics['AP'] - orig_metrics['AP']:+.3f}")


# =============================================================================
# SECTION 7: ABLATION STUDY (TASK 6)
# =============================================================================

def create_ablation_configs():
    """Define preprocessing configurations for ablation study."""
    return [
        ('baseline', True, False),  # Stops only
        ('no_stops', False, False),  # No preprocessing
        ('with_stemming', True, True),  # Stops + stemming
        ('stems_no_stops', False, True)  # Stemming only
    ]


def ablation_study(documents, doc_categories, queries, n_docs_per_config=200):
    """Compare MAP across preprocessing configurations."""
    configs = create_ablation_configs()
    ablation_results = {}

    # Subsample for faster ablation (full corpus in production)
    subsample_docs = documents[:n_docs_per_config]
    subsample_cats = doc_categories[:n_docs_per_config]

    print("\n=== ABLATION STUDY ===")

    for config_name, use_stops, use_stem in configs:
        print(f"\nProcessing: {config_name} (stops={use_stops}, stem={use_stem})")

        # Preprocess with config
        config_docs = [preprocess_document(doc, use_stops, use_stem)
                       for doc in subsample_docs]

        # Create vectorizer (handle stemming)
        vectorizer_config = TfidfVectorizer(
            lowercase=True,
            min_df=2,
            max_features=3000,
            stop_words='english' if use_stops else None,
            norm='l2'
        )

        # Apply stemming tokenizer if needed
        if use_stem:
            def stem_tokenizer(text):
                tokens = word_tokenize(text.lower())
                tokens = [t for t in tokens if t.isalpha() and len(t) >= 3]
                if use_stops:
                    tokens = [t for t in tokens if t not in stop_words]
                return [stemmer.stem(t) for t in tokens]

            vectorizer_config.tokenizer = stem_tokenizer

        X_config = vectorizer_config.fit_transform(config_docs)
        vocab_size = len(vectorizer_config.vocabulary_)

        # Evaluate on queries
        config_map = 0
        valid_queries_count = 0

        for query in queries:
            q_clean = preprocess_document(query, use_stops, use_stem)
            q_vec = vectorizer_config.transform([q_clean])
            sims = cosine_similarity(q_vec, X_config).flatten()
            top_docs = np.argsort(sims)[-10:][::-1]

            relevant = get_relevant_documents(query, subsample_cats, subsample_cats)
            if relevant:
                ap = calculate_metrics(top_docs, relevant)['AP']
                config_map += ap
                valid_queries_count += 1

        config_map_score = config_map / valid_queries_count if valid_queries_count > 0 else 0
        ablation_results[config_name] = {
            'MAP': config_map_score,
            'vocabulary_size': vocab_size,
            'evaluated_queries': valid_queries_count,
            'matrix_shape': X_config.shape
        }

        print(f"  MAP: {config_map_score:.3f}, Vocab: {vocab_size}, Shape: {X_config.shape}")

    return ablation_results


# Run ablation study
ablation_results = ablation_study(docs, all_categories, QUERIES)

# Save and visualize results
ablation_df = pd.DataFrame(ablation_results).T
ablation_df.to_csv('ablation_study_results.csv', index_label='Configuration')

# Create visualization
plt.figure(figsize=(10, 6))
configs = list(ablation_results.keys())
maps = [ablation_results[config]['MAP'] for config in configs]
vocab_sizes = [ablation_results[config]['vocabulary_size'] for config in configs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# MAP comparison
bars1 = ax1.bar(configs, maps, color=['skyblue', 'orange', 'lightgreen', 'lightcoral'])
ax1.set_ylabel('Mean Average Precision (MAP)')
ax1.set_title('Ablation Study: Retrieval Performance')
ax1.set_ylim(0, max(maps) * 1.1)

# Add value labels on bars
for bar, map_val in zip(bars1, maps):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{map_val:.3f}', ha='center', va='bottom', fontsize=9)

# Vocabulary size comparison
bars2 = ax2.bar(configs, vocab_sizes, color=['skyblue', 'orange', 'lightgreen', 'lightcoral'])
ax2.set_ylabel('Vocabulary Size')
ax2.set_title('Ablation Study: Vocabulary Impact')
ax2.set_ylim(0, max(vocab_sizes) * 1.1)

for bar, vocab_size in zip(bars2, vocab_sizes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 50,
             f'{vocab_size:,}', ha='center', va='bottom', fontsize=9)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ablation_study_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAblation study completed:")
print(ablation_df[['MAP', 'vocabulary_size']].round(3))

# Find best configuration
best_config = max(ablation_results, key=lambda x: ablation_results[x]['MAP'])
print(f"\nBest configuration: {best_config}")
print(f"  MAP: {ablation_results[best_config]['MAP']:.3f}")
print(f"  Vocabulary size: {ablation_results[best_config]['vocabulary_size']:,}")
print(f"  Matrix shape: {ablation_results[best_config]['matrix_shape']}")


# =============================================================================
# SECTION 8: RESULTS SUMMARY AND DOCUMENTATION
# =============================================================================

def generate_project_summary():
    """Generate summary statistics for project report."""
    summary = {
        'project_title': 'ISAMM Mini-Projet 1: Economic News Retrieval System',
        'corpus': 'Reuters-21578',
        'documents_processed': len(docs),
        'unique_categories': len(set(sum(all_categories, []))),
        'inverted_index_terms': len(inverted_index),
        'tfidf_features': X_baseline.shape[1],
        'queries_evaluated': len(QUERIES),
        'mean_average_precision': map_score,
        'best_ablation_config': best_config,
        'best_ablation_map': ablation_results[best_config]['MAP'],
        'rocchio_sample_improvement': feedback_metrics['AP'] - orig_metrics['AP'],
        'files_generated': [
            'inverted_index.json',
            'tfidf_vocabulary.txt',
            'evaluation_metrics.csv',
            'ablation_study_results.csv',
            'ablation_study_results.png'
        ]
    }

    # Save summary
    with open('project_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# Generate final summary
project_summary = generate_project_summary()
print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"📊 Documents processed: {project_summary['documents_processed']}")
print(f"🔤 Inverted index terms: {project_summary['inverted_index_terms']:,}")
print(f"⚡ TF-IDF features: {project_summary['tfidf_features']:,}")
print(f"📈 Mean Average Precision: {project_summary['mean_average_precision']:.3f}")
print(f"🎯 Best ablation: {project_summary['best_ablation_config']} (MAP {project_summary['best_ablation_map']:.3f})")
print(f"📈 Rocchio improvement: {project_summary['rocchio_sample_improvement']:+.3f}")
print(f"\n📁 Generated files:")
for file in project_summary['files_generated']:
    print(f"   • {file}")
print(f"\n📋 Summary saved: project_summary.json")
print("\n🎓 All deliverables ready for submission!")
print("=" * 60)

# Final evaluation table for report
print("\n📋 EVALUATION SUMMARY TABLE:")
print("-" * 80)
print(f"{'Query':<18} {'Relevants':<10} {'P@10':<6} {'R@10':<6} {'AP':<6} {'Ranked Top-3'}")
print("-" * 80)

for query in QUERIES:
    if evaluation_metrics[query]['relevant_count'] > 0:
        row = evaluation_metrics[query]
        top3 = query_results[query]['top_documents'][:3]
        print(f"{query:<18} {row['relevant_count']:<10} {row['P@10']:<6.2f} "
              f"{row['R@10']:<6.2f} {row['AP']:<6.2f} {top3}")
    else:
        print(f"{query:<18} {'0':<10} {'-':<6} {'-':<6} {'-':<6} {'N/A'}")

print(f"\n{'Overall MAP':<55} {map_score:.3f}")
print("-" * 80)
