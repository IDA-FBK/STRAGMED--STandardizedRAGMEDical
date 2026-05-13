"""
Retrieval Evaluation Script v8 - English MIRAGE Benchmark
==========================================================
Evaluates on PubMedQA and BioASQ with DUAL RRF (k=0 and k=60).

Key differences from v7:
  - Loads questions from JSON (eval_questions.json) instead of CSV
  - Handles MULTIPLE gold docs per question (BioASQ has avg ~6)
  - Tracks metrics for BOTH RRF k=0 and k=60 variants
  - Hit = ANY gold doc in top-3 (not just single match)

Stages tracked:
  - Vector (before/after rerank)
  - BM25 (before/after rerank)
  - HyDE (before/after rerank)
  - RRF k=0 (before/after final rerank)
  - RRF k=60 (before/after final rerank)
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from pipeline import (
    RAGpipeline,
    USE_PER_RETRIEVER_RERANKING, USE_FINAL_RERANKER,
    RERANKER_MODEL, PER_RETRIEVER_TOP_N, FINAL_RERANKER_TOP_N,
    DEBUG_VERBOSE,
    PROJECT_ROOT, DEFAULT_EMBEDDINGS_DIR,
)
from llama_index.core.prompts import RichPromptTemplate

DEFAULT_QUESTIONS = PROJECT_ROOT / "inputs" / "eval_questions.json"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def normalize_name(fname):
    if fname is None or pd.isna(fname) or fname == "nan":
        return ""
    return str(fname).lower().replace('.txt', '').strip()


def get_clean_top_n(items, n=50):
    """Extract doc names from results (handles dict format)."""
    result = []
    for item in items[:n]:
        if isinstance(item, dict):
            result.append(normalize_name(item.get('doc', '')))
        elif isinstance(item, (tuple, list)):
            result.append(normalize_name(item[0]))
        else:
            result.append(normalize_name(item))
    return result


def check_hit(gold_docs: list, retrieved_list: list, top_k: int = 3) -> bool:
    """
    Check if ANY gold doc appears in top-k retrieved docs.
    This handles BioASQ's multiple gold docs per question.
    """
    clean_retrieved = get_clean_top_n(retrieved_list, n=top_k)
    clean_gold = [normalize_name(g) for g in gold_docs]
    return any(g in clean_retrieved for g in clean_gold)


def get_best_rank(gold_docs: list, retrieved_list: list, limit: int = 50, penalty: int = 100) -> int:
    """
    Get the BEST (lowest) rank among all gold docs.
    Returns penalty if no gold doc found in top-limit.
    """
    clean_retrieved = get_clean_top_n(retrieved_list, n=limit)
    clean_gold = [normalize_name(g) for g in gold_docs]

    best_rank = penalty
    for gold in clean_gold:
        try:
            rank = clean_retrieved.index(gold) + 1
            if rank < best_rank:
                best_rank = rank
        except ValueError:
            continue

    return best_rank


# -----------------------------------------------------------------------------
# MAIN EVALUATION
# -----------------------------------------------------------------------------
def evaluate(questions_file: str, output_file: str = None, num_rows: int = None,
             source_filter: str = None, hyde_docs_file: str = None, row_range: tuple = None,
             embeddings_dir: str = None):
    """
    Run evaluation on MIRAGE benchmark.

    Args:
        questions_file: Path to eval_questions.json
        output_file: Output CSV path
        num_rows: Limit number of questions (for testing)
        source_filter: Filter by source ("pubmedqa" or "bioasq")
        hyde_docs_file: Path to pre-generated HyDE docs JSON (optional)
    """

    # Load pre-generated HyDE docs if provided
    hyde_docs = {}
    if hyde_docs_file:
        print(f"Loading pre-generated HyDE docs from {hyde_docs_file}...")
        with open(hyde_docs_file, 'r', encoding='utf-8') as f:
            hyde_data = json.load(f)
        hyde_docs = hyde_data.get('hyde_docs', {})
        print(f"Loaded {len(hyde_docs)} pre-generated HyDE docs")

    config_info = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Pipeline_Script': 'pipeline',
        'Reranker_Model': RERANKER_MODEL,
        'Per_Retriever_Reranking': USE_PER_RETRIEVER_RERANKING,
        'Per_Retriever_Top_N': PER_RETRIEVER_TOP_N,
        'Final_Reranking': USE_FINAL_RERANKER,
        'Final_Reranker_Top_N': FINAL_RERANKER_TOP_N if USE_FINAL_RERANKER else "N/A",
        'Source_Filter': source_filter or "all",
        'Hyde_Docs_File': hyde_docs_file or "generated on-the-fly",
    }

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        output_file = str(DEFAULT_RUNS_DIR / f"mirage_eval_{timestamp}.csv")
    else:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"MIRAGE RETRIEVAL EVALUATION v8 - Dual RRF (k=0 and k=60)")
    print(f"Config: {json.dumps(config_info, indent=2)}")
    print(f"Output: {output_file}")
    print("=" * 70)

    # Load questions
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data['questions']
    print(f"\nLoaded {len(questions)} questions from {questions_file}")

    # Filter by source if specified
    if source_filter:
        questions = [q for q in questions if q['source'] == source_filter]
        print(f"Filtered to {len(questions)} {source_filter} questions")

    # Limit rows if specified
    if row_range:
        start, end = row_range
        questions = questions[start:end]
        print(f"Selected rows {start} to {end} ({len(questions)} questions)")
    elif num_rows:
        questions = questions[:num_rows]
        print(f"Limited to {num_rows} questions")

    # Filter out questions without gold docs
    questions_with_gold = [q for q in questions if len(q.get('gold_docs', [])) > 0]
    print(f"Questions with gold docs: {len(questions_with_gold)}")

    # English HyDE prompt
    hyde_prompt = RichPromptTemplate(
        """You are a medical expert assistant.
Answer the question based on medical knowledge.

Use the following context if helpful: {{ context }}

### Question:
{{ query_str }}

### Answer:
"""
    )

    print("\n>>> Initializing RAG Pipeline...")
    pipeline = RAGpipeline(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        hyde_prompt=hyde_prompt,
        embeddings_dir=embeddings_dir,
    )

    config_info['Vector_Top_K'] = pipeline.vector_retriever._similarity_top_k
    config_info['BM25_Top_K'] = pipeline.bm25_retriever.similarity_top_k

    results = []
    detailed_results = []
    generated_hyde_docs = {}  # NEW: Track HyDE docs + contexts

    # Tracking stats for all stages
    stages = [
        'vec', 'vec_rerank',
        'bm25', 'bm25_rerank',
        'hyde', 'hyde_rerank',
        'rrf_k0', 'final_k0',
        'rrf_k60', 'final_k60'
    ]
    rank_stats = {s: [] for s in stages}
    hit_stats = {s: 0 for s in stages}

    # Per-source tracking
    source_stats = {
        'pubmedqa': {'total': 0, 'hits': {s: 0 for s in stages}},
        'bioasq': {'total': 0, 'hits': {s: 0 for s in stages}}
    }

    total = 0

    for q in tqdm(questions_with_gold, desc="Evaluating"):
        question = q['question']
        gold_docs = q['gold_docs']
        source = q['source']
        question_id = q['question_id']

        try:
            # Use pre-generated HyDE doc if available
            precomputed_hyde = None
            if question_id in hyde_docs:
                precomputed_hyde = hyde_docs[question_id].get('hyde_doc')

            origin_files, scores, pieces, hydoc, intermediate_results = pipeline.retrieve_documents(
                question, precomputed_hyde=precomputed_hyde
            )
            total += 1
            source_stats[source]['total'] += 1

            # Map stage names to intermediate_results keys
            stage_map = {
                'vec': 'vector_top50',
                'vec_rerank': 'vector_reranked_top50',
                'bm25': 'bm25_top50',
                'bm25_rerank': 'bm25_reranked_top50',
                'hyde': 'hyde_top50',
                'hyde_rerank': 'hyde_reranked_top50',
                'rrf_k0': 'rrf_k0_top50',
                'final_k0': 'final_k0_top50',
                'rrf_k60': 'rrf_k60_top50',
                'final_k60': 'final_k60_top50',
            }

            row_data = {
                'question_id': question_id,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'source': source,
                'num_gold_docs': len(gold_docs),
                'gold_docs': ','.join(gold_docs[:5]),  # First 5 for display
            }

            ranks = {}
            hits = {}

            for stage, key in stage_map.items():
                retrieved = intermediate_results.get(key, [])
                rank = get_best_rank(gold_docs, retrieved)
                hit = 1 if rank <= 3 else 0

                ranks[stage] = rank
                hits[stage] = hit

                rank_stats[stage].append(rank)
                hit_stats[stage] += hit
                source_stats[source]['hits'][stage] += hit

                # Get top 3 for display
                top3 = get_clean_top_n(retrieved, 3)
                row_data[f'{stage}_1'] = top3[0] if len(top3) > 0 else ""
                row_data[f'{stage}_2'] = top3[1] if len(top3) > 1 else ""
                row_data[f'{stage}_3'] = top3[2] if len(top3) > 2 else ""
                row_data[f'{stage}_hit'] = hit
                row_data[f'{stage}_rank'] = rank

            results.append(row_data)

            # Detailed results for JSON
            detailed_row = {
                'question_id': question_id,
                'question': question,
                'source': source,
                'gold_docs': gold_docs,
                'ranks': ranks,
                'hits': hits,
                **{k: intermediate_results.get(k, []) for k in stage_map.values()}
            }
            detailed_results.append(detailed_row)

            # Save HyDE doc + context (only if generated, not precomputed)
            if not precomputed_hyde:
                generated_hyde_docs[question_id] = {
                    'question': question,
                    'hyde_doc': hydoc,
                    'context_docs': intermediate_results.get('hyde_context_docs', []),
                    'source': source,
                    'generated_at': datetime.now().isoformat()
                }

        except Exception as e:
            print(f"\nError on {question_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'question_id': question_id, 'status': 'ERROR', 'error': str(e)})

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    if total > 0:
        def get_avg(lst): return sum(lst) / len(lst) if lst else 0
        def get_pct(count, t=total): return f"{count/t*100:.1f}%" if t > 0 else "0%"

        results_df = pd.DataFrame(results)

        # Add summary rows
        spacer = {k: '' for k in results_df.columns}
        header_summary = {k: '' for k in results_df.columns}
        header_summary['question_id'] = "=== OVERALL METRICS ==="

        # Average ranks
        avg_row = {k: '' for k in results_df.columns}
        avg_row['question_id'] = "AVG RANK (lower is better)"
        for stage in stages:
            avg_row[f'{stage}_rank'] = f"{get_avg(rank_stats[stage]):.2f}"

        # Hit rates
        hit_row = {k: '' for k in results_df.columns}
        hit_row['question_id'] = "HIT RATE @ 3 (higher is better)"
        for stage in stages:
            hit_row[f'{stage}_rank'] = get_pct(hit_stats[stage])

        # Per-source metrics
        source_rows = []
        for src in ['pubmedqa', 'bioasq']:
            if source_stats[src]['total'] > 0:
                src_row = {k: '' for k in results_df.columns}
                src_row['question_id'] = f"--- {src.upper()} (n={source_stats[src]['total']}) ---"
                source_rows.append(src_row)

                src_hit_row = {k: '' for k in results_df.columns}
                src_hit_row['question_id'] = f"  Hit@3"
                for stage in stages:
                    src_hit_row[f'{stage}_rank'] = get_pct(
                        source_stats[src]['hits'][stage],
                        source_stats[src]['total']
                    )
                source_rows.append(src_hit_row)

        # Config rows
        header_config = {k: '' for k in results_df.columns}
        header_config['question_id'] = "=== CONFIGURATION ==="
        config_rows = []
        for key, val in config_info.items():
            row = {k: '' for k in results_df.columns}
            row['question_id'] = str(key)
            row['question'] = str(val)
            config_rows.append(row)

        final_df = pd.concat([
            results_df,
            pd.DataFrame([spacer, header_summary, avg_row, hit_row]),
            pd.DataFrame(source_rows),
            pd.DataFrame([spacer, header_config] + config_rows)
        ], ignore_index=True)

        final_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Save detailed JSON
        json_output = output_file.replace('.csv', '_detailed.json')
        json_data = {
            'config': config_info,
            'summary': {
                'total_queries': total,
                'avg_ranks': {k: get_avg(v) for k, v in rank_stats.items()},
                'hit_rates': {k: v/total*100 for k, v in hit_stats.items()},
                'per_source': {
                    src: {
                        'total': source_stats[src]['total'],
                        'hit_rates': {
                            stage: (source_stats[src]['hits'][stage] / source_stats[src]['total'] * 100)
                            if source_stats[src]['total'] > 0 else 0
                            for stage in stages
                        }
                    }
                    for src in ['pubmedqa', 'bioasq']
                }
            },
            'results': detailed_results
        }
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"Detailed results saved to: {json_output}")

        # Save HyDE docs + contexts to separate file (if any were generated)
        if generated_hyde_docs:
            hyde_output = output_file.replace('.csv', '_hyde_docs.json')
            hyde_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_questions': len(generated_hyde_docs),
                    'model': 'meta-llama/llama-3.3-70b-instruct',
                    'embedding_model': 'mixedbread-ai/mxbai-embed-large-v1',
                    'reranker_model': RERANKER_MODEL,
                    'context_strategy': 'top5_bm25 + top5_vector (up to 8 unique docs)',
                    'note': 'Use --hyde_docs <this_file> to reuse these HyDE docs in future runs'
                },
                'hyde_docs': generated_hyde_docs
            }
            with open(hyde_output, 'w', encoding='utf-8') as f:
                json.dump(hyde_data, f, ensure_ascii=False, indent=2)
            print(f"HyDE docs + contexts saved to: {hyde_output}")
            print(f"  -> Reuse with: --hyde_docs {hyde_output}")

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY METRICS")
        print("=" * 80)
        print(f"\n{'Stage':<20} {'Avg Rank':<12} {'Hit@3':<10}")
        print("-" * 42)
        for stage in stages:
            print(f"{stage:<20} {get_avg(rank_stats[stage]):<12.2f} {get_pct(hit_stats[stage]):<10}")

        print("\n" + "-" * 42)
        print("PER-SOURCE HIT@3:")
        print("-" * 42)
        for src in ['pubmedqa', 'bioasq']:
            if source_stats[src]['total'] > 0:
                print(f"\n{src.upper()} (n={source_stats[src]['total']}):")
                for stage in ['final_k0', 'final_k60']:
                    hit_pct = source_stats[src]['hits'][stage] / source_stats[src]['total'] * 100
                    print(f"  {stage}: {hit_pct:.1f}%")

        print("=" * 80)

        return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIRAGE Benchmark Evaluation v8")
    parser.add_argument("--questions", type=str, default=str(DEFAULT_QUESTIONS),
                        help=f"Path to eval_questions.json (default: {DEFAULT_QUESTIONS})")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output CSV path (default: {DEFAULT_RUNS_DIR}/mirage_eval_<timestamp>.csv)")
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR),
                        help=f"ChromaDB index dir (default: {DEFAULT_EMBEDDINGS_DIR})")
    parser.add_argument("--num_rows", type=int, default=None,
                        help="Limit number of questions (for testing)")
    parser.add_argument("--row_range", type=int, nargs=2, metavar=("START", "END"),
                        help="Evaluate specific row range, e.g., --row_range 10 20")
    parser.add_argument("--source", type=str, choices=["pubmedqa", "bioasq"], default=None,
                        help="Filter by source dataset")
    parser.add_argument("--hyde_docs", type=str, default=None,
                        help="Path to pre-generated HyDE docs JSON")
    parser.add_argument("--display", action="store_true",
                        help="Display results in pandas after saving")
    args = parser.parse_args()

    result_df = evaluate(
        questions_file=args.questions,
        output_file=args.output,
        num_rows=args.num_rows,
        source_filter=args.source,
        hyde_docs_file=args.hyde_docs,
        row_range=tuple(args.row_range) if args.row_range else None,
        embeddings_dir=args.embeddings_dir,
    )

    if args.display and result_df is not None:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print("\n" + "=" * 80)
        print("RESULTS DATAFRAME")
        print("=" * 80)
        print(result_df)
