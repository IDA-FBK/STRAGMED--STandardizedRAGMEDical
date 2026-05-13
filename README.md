# Standardizing RAG Pipelines in Medical Domains — Reproduction Package

Code and minimal data to reproduce the MIRAGE results of:

> Sanna, L., Yildiz, E.E., Dragoni, M. *Standardizing Retrieval-Augmented Generation
> Pipelines in Medical Domains.* AIME 2026.

The pipeline runs hybrid retrieval (Vector + BM25 + HyDE) with per-retriever
reranking, dual RRF fusion (k=0 and k=60), and a final reranker, evaluated on
the MIRAGE benchmark (PubMedQA + BioASQ) over a 1 M-document PubMed corpus.

> The ChatFAQ (Italian) results in the paper rely on proprietary clinical
> materials and are not included in this release.

## Requirements

- Python 3.12+
- Poetry
- ~30 GB free disk (~5 GB corpus, ~22 GB ChromaDB index)
- A CUDA-capable GPU (the reranker is slow on CPU)
- An [OpenRouter](https://openrouter.ai/) API key for HyDE generation
- Optional: an [NCBI](https://www.ncbi.nlm.nih.gov/account/) API key (10 req/s vs 3 req/s on PubMed downloads)

## Setup

```bash
poetry install
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY (and NCBI_API_KEY if you have one)
```

All commands below are run from the project root. Scripts default to writing
into project-root subdirectories (`corpus/`, `embeddings/`, `runs/`).

## Reproduction

### 1. Download the corpus (~1 M PubMed abstracts)

```bash
poetry run python src/download_corpus.py all --total 1000000 --mesh_portion 0.5
```

Fetches the 4,554 gold abstracts (PMIDs listed in `inputs/all_pmids.json`)
plus ~1 M negatives — 500k MeSH-targeted hard negatives via NCBI search and
500k random negatives sampled from the
[MedRAG/pubmed](https://huggingface.co/datasets/MedRAG/pubmed) HuggingFace
dataset. Resumable — already-downloaded files are skipped.

Subcommands also work in isolation:

```bash
poetry run python src/download_corpus.py gold
poetry run python src/download_corpus.py mesh --num_docs 500000
poetry run python src/download_corpus.py hf   --num_docs 500000
poetry run python src/download_corpus.py negatives --total 1000000
```

### 2. Build the ChromaDB index

```bash
poetry run python src/make_embeddings.py
```

Embeds `corpus/` with `mxbai-embed-large-v1` into
`embeddings/mxbai-embed-large-v1/`. A few hours on a single GPU.

### 3. Run the pipeline

```bash
poetry run python src/evaluate.py
```

Loads questions from `inputs/eval_questions.json` (1,000 PubMedQA + 618 BioASQ)
and writes three files to `runs/`: a CSV summary, a `*_detailed.json` with
top-50 rankings for all 10 retrieval stages, and a `*_hyde_docs.json` cache
that can be reused with `--hyde_docs`.

Useful flags:

```bash
# Smoke test on 50 questions
poetry run python src/evaluate.py --num_rows 50

# One dataset only
poetry run python src/evaluate.py --source pubmedqa
poetry run python src/evaluate.py --source bioasq
```

### 4. Compute the Table 2 metrics

```bash
poetry run python src/compute_metrics.py \
    --input runs/mirage_eval_<ts>_detailed.json \
    --by_source \
    --output runs/table2.json
```

Reports Hit@3, MRR, MAP@50, NDCG@50 per stage, both overall and per source.
Stage keys map to Table 2 rows as:

| Stage key         | Table 2 row              |
|-------------------|--------------------------|
| `vector`          | vector                   |
| `vector_reranked` | vector reranked          |
| `bm25`            | BM25                     |
| `bm25_reranked`   | BM25 reranked            |
| `hyde`            | hyde                     |
| `hyde_reranked`   | hyde reranked            |
| `rrf_k60`         | RRF (k=60)               |
| `final_k60`       | RRF reranked (k=60)      |
| `rrf_k0`          | RRF (k=0)                |
| `final_k0`        | RRF reranked (k=0)       |

## Models

| Component | Model |
|---|---|
| Embedding | `mixedbread-ai/mxbai-embed-large-v1` |
| Lexical retriever | BM25 (`llama-index-retrievers-bm25`) |
| Vector store | ChromaDB |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) |
| HyDE LLM | OpenRouter, `meta-llama/llama-3.3-70b-instruct`, seed=2025 |

## Repo layout

```
release/
├── README.md
├── pyproject.toml / poetry.lock     pinned dependencies
├── .env.example                     template for OPENROUTER_API_KEY / NCBI_API_KEY
├── inputs/
│   ├── eval_questions.json          1,618 prepared queries with gold doc PMIDs
│   └── all_pmids.json               4,554 gold PMIDs (input to download_corpus gold)
└── src/
    ├── prepare_mirage_datasets.py   (optional) re-derive inputs/ from raw upstream files
    ├── download_corpus.py           gold + negatives downloader
    ├── make_embeddings.py           build the ChromaDB index
    ├── pipeline.py                  the pipeline
    ├── evaluate.py                  eval driver — produces the detailed JSON
    └── compute_metrics.py           Table 2 metrics from the detailed JSON
```

## (Optional) Re-deriving the eval inputs

`inputs/eval_questions.json` and `inputs/all_pmids.json` are pre-built. If you
want to regenerate them, you need four upstream files:

| File | Source |
|---|---|
| `benchmark.json` | [MIRAGE benchmark](https://github.com/Teddy-XiongGZ/MIRAGE) |
| `pubmedqa_labeled_1k.json` | [PubMedQA](https://github.com/pubmedqa/pubmedqa) (PQA-L) |
| `training11b.json`, `training12b_new.json` | [BioASQ](http://bioasq.org/) Task 11b / 12b training data (registration required) |

```bash
poetry run python src/prepare_mirage_datasets.py \
    --benchmark path/to/benchmark.json \
    --pubmedqa_file path/to/pubmedqa_labeled_1k.json \
    --bioasq_training path/to/training11b.json path/to/training12b_new.json
```
