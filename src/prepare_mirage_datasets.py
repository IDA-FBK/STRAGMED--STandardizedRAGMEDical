"""
MIRAGE Benchmark Dataset Preparation Script
============================================
Prepares PubMedQA and BioASQ datasets for retrieval evaluation.

Outputs unified format:
{
    "question_id": "...",
    "question": "...",
    "gold_docs": ["pmid_1", "pmid_2", ...],
    "answer": "yes/no/maybe",
    "source": "pubmedqa" or "bioasq"
}

Usage:
    python prepare_mirage_datasets.py --benchmark benchmark.json --output eval_questions.json

    # With BioASQ training file:
    python prepare_mirage_datasets.py --benchmark benchmark.json --bioasq_training training11b.json --output eval_questions.json

    # Use full PubMedQA from HuggingFace (1000 questions instead of 500 from benchmark):
    python prepare_mirage_datasets.py --benchmark benchmark.json --use_full_pubmedqa --output eval_questions.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
DEFAULT_BENCHMARK = INPUTS_DIR / "benchmark.json"
DEFAULT_OUTPUT = INPUTS_DIR / "eval_questions.json"
DEFAULT_PMIDS_OUTPUT = INPUTS_DIR / "all_pmids.json"


def load_pubmedqa_from_benchmark(benchmark_data: dict) -> List[dict]:
    """
    Load PubMedQA from benchmark.json.
    The question IDs in benchmark.json ARE the PMIDs.

    Returns list of questions in unified format.
    """
    questions = []
    pubmedqa_data = benchmark_data.get("pubmedqa", {})

    answer_map = {"A": "yes", "B": "no", "C": "maybe"}

    for pmid, item in pubmedqa_data.items():
        questions.append({
            "question_id": f"pubmedqa_{pmid}",
            "question": item["question"],
            "gold_docs": [pmid],  # The ID is the PMID
            "answer": answer_map.get(item.get("answer", ""), "unknown"),
            "source": "pubmedqa"
        })

    print(f"[PubMedQA] Loaded {len(questions)} questions from benchmark.json")
    return questions


def load_pubmedqa_from_local_file(file_path: str) -> List[dict]:
    """
    Load full PubMedQA PQA-L dataset from local JSON file (1,000 questions).

    Expected format (array of objects):
    [
        {
            "pmid": 21645374,
            "question": "...",
            "context": "...",
            "long_answer": "...",
            "final_decision": "yes/no/maybe"
        },
        ...
    ]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = []
    for item in data:
        pmid = str(item["pmid"])
        questions.append({
            "question_id": f"pubmedqa_{pmid}",
            "question": item["question"],
            "gold_docs": [pmid],
            "answer": item["final_decision"],  # yes/no/maybe
            "source": "pubmedqa",
            # Additional metadata
            "context": item.get("context", ""),
            "long_answer": item.get("long_answer", "")
        })

    print(f"[PubMedQA] Loaded {len(questions)} questions from {file_path}")
    return questions


def load_bioasq_training(training_file: str) -> Dict[str, dict]:
    """
    Load BioASQ training11b.json file.
    Returns a dict mapping question IDs to full question data (including gold doc PMIDs).

    The training file has structure:
    {
        "questions": [
            {
                "id": "...",
                "body": "...",
                "documents": ["http://www.ncbi.nlm.nih.gov/pubmed/12345", ...],
                "type": "yesno",
                "exact_answer": "yes" or "no",
                ...
            },
            ...
        ]
    }
    """
    with open(training_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions_map = {}
    for q in data.get("questions", []):
        questions_map[q["id"]] = q

    print(f"[BioASQ] Loaded {len(questions_map)} questions from training file")
    return questions_map


def extract_pmid_from_url(url: str) -> Optional[str]:
    """Extract PMID from a PubMed URL."""
    # URLs like: http://www.ncbi.nlm.nih.gov/pubmed/12345
    if "pubmed/" in url:
        return url.split("pubmed/")[-1].strip("/")
    return None


def load_bioasq_from_benchmark(
    benchmark_data: dict,
    training_data: Optional[Dict[str, dict]] = None
) -> List[dict]:
    """
    Load BioASQ from benchmark.json, matching with training11b.json for gold doc PMIDs.

    Args:
        benchmark_data: The benchmark.json data
        training_data: Optional mapping from training11b.json (qid -> full question)

    Returns list of questions in unified format.
    """
    questions = []
    bioasq_data = benchmark_data.get("bioasq", {})

    answer_map = {"A": "yes", "B": "no"}

    matched = 0
    unmatched = 0
    unmatched_ids = []

    for qid, item in bioasq_data.items():
        gold_docs = []

        if training_data and qid in training_data:
            # Get PMIDs from training data
            training_q = training_data[qid]
            doc_urls = training_q.get("documents", [])
            for url in doc_urls:
                pmid = extract_pmid_from_url(url)
                if pmid:
                    gold_docs.append(pmid)
            matched += 1
        else:
            unmatched += 1
            unmatched_ids.append(qid)

        questions.append({
            "question_id": f"bioasq_{qid}",
            "question": item["question"],
            "gold_docs": gold_docs,
            "answer": answer_map.get(item.get("answer", ""), "unknown"),
            "source": "bioasq"
        })

    print(f"[BioASQ] Loaded {len(questions)} questions from benchmark.json")
    if training_data:
        print(f"[BioASQ] Matched {matched} questions with training data")
        if unmatched > 0:
            print(f"[BioASQ] WARNING: {unmatched} questions not found in training data")
            print(f"[BioASQ] First 5 unmatched IDs: {unmatched_ids[:5]}")
    else:
        print(f"[BioASQ] WARNING: No training file provided - gold_docs will be empty!")
        print(f"[BioASQ] Download training11b.json from: http://participants-area.bioasq.org/datasets/")

    return questions


def collect_all_pmids(questions: List[dict]) -> List[str]:
    """Collect all unique PMIDs from all questions."""
    pmids = set()
    for q in questions:
        for pmid in q.get("gold_docs", []):
            pmids.add(pmid)
    return sorted(list(pmids))


def print_statistics(questions: List[dict], pmids: List[str]):
    """Print dataset statistics."""
    pubmedqa_qs = [q for q in questions if q["source"] == "pubmedqa"]
    bioasq_qs = [q for q in questions if q["source"] == "bioasq"]

    bioasq_with_docs = [q for q in bioasq_qs if len(q["gold_docs"]) > 0]

    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total questions:           {len(questions)}")
    print(f"  - PubMedQA:              {len(pubmedqa_qs)}")
    print(f"  - BioASQ:                {len(bioasq_qs)}")
    print(f"    - with gold docs:      {len(bioasq_with_docs)}")
    print(f"    - without gold docs:   {len(bioasq_qs) - len(bioasq_with_docs)}")
    print(f"\nTotal unique PMIDs:        {len(pmids)}")

    # Answer distribution
    answer_counts = defaultdict(lambda: defaultdict(int))
    for q in questions:
        answer_counts[q["source"]][q["answer"]] += 1

    print("\nAnswer distribution:")
    for source in ["pubmedqa", "bioasq"]:
        if source in answer_counts:
            dist = dict(answer_counts[source])
            print(f"  {source}: {dist}")

    # Gold docs per question
    pubmedqa_docs = [len(q["gold_docs"]) for q in pubmedqa_qs]
    bioasq_docs = [len(q["gold_docs"]) for q in bioasq_qs if len(q["gold_docs"]) > 0]

    if pubmedqa_docs:
        print(f"\nGold docs per question (PubMedQA): always 1 (the source abstract)")
    if bioasq_docs:
        avg_docs = sum(bioasq_docs) / len(bioasq_docs)
        print(f"Gold docs per question (BioASQ): avg={avg_docs:.1f}, min={min(bioasq_docs)}, max={max(bioasq_docs)}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MIRAGE benchmark datasets for retrieval evaluation"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(DEFAULT_BENCHMARK),
        help=f"Path to benchmark.json (default: {DEFAULT_BENCHMARK})"
    )
    parser.add_argument(
        "--bioasq_training",
        type=str,
        nargs='+',
        default=None,
        help="Path(s) to BioASQ training JSON files (e.g., training11b.json training12b.json)"
    )
    parser.add_argument(
        "--pubmedqa_file",
        type=str,
        default=None,
        help="Path to local PubMedQA JSON file (1000 questions) instead of using benchmark (500)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output file path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--pmids_output",
        type=str,
        default=str(DEFAULT_PMIDS_OUTPUT),
        help=f"Output file for list of all PMIDs for corpus download (default: {DEFAULT_PMIDS_OUTPUT})"
    )
    parser.add_argument(
        "--pubmedqa_only",
        action="store_true",
        help="Only process PubMedQA dataset"
    )
    parser.add_argument(
        "--bioasq_only",
        action="store_true",
        help="Only process BioASQ dataset"
    )
    args = parser.parse_args()

    # Load benchmark
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    print(f"Loading benchmark from: {benchmark_path}")
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    all_questions = []

    # Load PubMedQA
    if not args.bioasq_only:
        if args.pubmedqa_file:
            pubmedqa_path = Path(args.pubmedqa_file)
            if not pubmedqa_path.exists():
                raise FileNotFoundError(f"PubMedQA file not found: {pubmedqa_path}")
            pubmedqa_questions = load_pubmedqa_from_local_file(args.pubmedqa_file)
        else:
            pubmedqa_questions = load_pubmedqa_from_benchmark(benchmark_data)
        all_questions.extend(pubmedqa_questions)

    # Load BioASQ
    if not args.pubmedqa_only:
        # Load training data if provided (supports multiple files)
        training_data = None
        if args.bioasq_training:
            training_data = {}
            for training_file in args.bioasq_training:
                training_path = Path(training_file)
                if training_path.exists():
                    file_data = load_bioasq_training(training_file)
                    training_data.update(file_data)
                else:
                    print(f"WARNING: Training file not found: {training_path}")

        bioasq_questions = load_bioasq_from_benchmark(benchmark_data, training_data)
        all_questions.extend(bioasq_questions)

    # Collect all PMIDs
    all_pmids = collect_all_pmids(all_questions)

    # Print statistics
    print_statistics(all_questions, all_pmids)

    # Save questions
    output_path = Path(args.output)
    output_data = {
        "metadata": {
            "total_questions": len(all_questions),
            "total_pmids": len(all_pmids),
            "sources": {
                "pubmedqa": len([q for q in all_questions if q["source"] == "pubmedqa"]),
                "bioasq": len([q for q in all_questions if q["source"] == "bioasq"])
            },
            "pubmedqa_source": args.pubmedqa_file if args.pubmedqa_file else "benchmark",
            "bioasq_training_files": args.bioasq_training
        },
        "questions": all_questions
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved questions to: {output_path}")

    pmids_path = Path(args.pmids_output)
    pmids_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pmids_path, 'w') as f:
        json.dump(all_pmids, f, indent=2)
    print(f"Saved {len(all_pmids)} PMIDs to: {pmids_path}")


if __name__ == "__main__":
    main()
