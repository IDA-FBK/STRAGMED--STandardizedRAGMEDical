"""
Unified corpus downloader for the MIRAGE retrieval evaluation.

Subcommands:
    gold        Download gold PubMed abstracts from NCBI for a list of PMIDs.
    mesh        Download MeSH-based hard negatives from NCBI (E-utilities search + fetch).
    hf          Download random negatives from the MedRAG/pubmed HuggingFace dataset.
    negatives   Build a mixed negative pool (default: 50% mesh + 50% hf, the paper config).
    all         Run `gold` then `negatives` end-to-end.

Each subcommand writes one .txt file per PMID into the corpus directory
(default: <project_root>/corpus). Files are skipped if already present, so
runs are resumable.

NCBI API key (optional): export NCBI_API_KEY=... or put it in .env.
Without a key the rate limit is 3 req/s; with a key it is 10 req/s.
"""

import argparse
import json
import os
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set
from xml.etree import ElementTree as ET

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "corpus"
DEFAULT_PMIDS_FILE = PROJECT_ROOT / "inputs" / "all_pmids.json"
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from tqdm import tqdm

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# The 15 topic-targeted MeSH queries used to build the 1 M-doc corpus in the paper.
# Topics were extracted from the gold-doc MeSH distribution.
DEFAULT_MESH_QUERIES = [
    '"COVID-19"[MeSH Major Topic] AND :2024[dp]',
    '"Epigenesis, Genetic"[MeSH Major Topic] AND :2024[dp]',
    '"Mutation"[MeSH Major Topic] AND :2024[dp]',
    '"Gene Expression Regulation"[MeSH Major Topic] AND :2024[dp]',
    '"Enhancer Elements, Genetic"[MeSH Major Topic] AND :2024[dp]',
    '"SARS-CoV-2"[MeSH Major Topic] AND :2024[dp]',
    '"DNA Damage"[MeSH Major Topic] AND :2024[dp]',
    '"Transcription, Genetic"[MeSH Major Topic] AND :2024[dp]',
    '"DNA Methylation"[MeSH Major Topic] AND :2024[dp]',
    '"Quality of Life"[MeSH Major Topic] AND :2024[dp]',
    '"Signal Transduction"[MeSH Major Topic] AND :2024[dp]',
    '"Gene Expression Regulation, Developmental"[MeSH Major Topic] AND :2024[dp]',
    '"Evolution, Molecular"[MeSH Major Topic] AND :2024[dp]',
    '"Resistance Training"[MeSH Major Topic] AND :2024[dp]',
    '"Muscular Dystrophy, Duchenne"[MeSH Major Topic] AND :2024[dp]',
]


# =====================================================================
# SHARED HELPERS
# =====================================================================
def load_excluded_pmids(path: Optional[str]) -> Set[str]:
    """Load a set of PMIDs to skip from a JSON list or {'pmids': [...]} dict."""
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return {str(x) for x in data}
    if isinstance(data, dict) and "pmids" in data:
        return {str(x) for x in data["pmids"]}
    return set()


def get_api_key(cli_key: Optional[str] = None) -> Optional[str]:
    key = cli_key or os.environ.get("NCBI_API_KEY")
    if key:
        print("[INFO] NCBI API key found (10 req/s limit)")
    else:
        print("[INFO] No NCBI API key (3 req/s limit). Set NCBI_API_KEY for faster downloads.")
    return key


def existing_pmids(output_dir: Path) -> Set[str]:
    return {f.stem for f in output_dir.glob("*.txt")} if output_dir.exists() else set()


def write_abstract(output_dir: Path, pmid: str, title: str, abstract: str) -> bool:
    """Write one PubMed abstract as <pmid>.txt. Returns True if anything was written."""
    if not title and not abstract:
        return False
    (output_dir / f"{pmid}.txt").write_text(f"{title}\n\n{abstract}", encoding="utf-8")
    return True


def search_pubmed(query: str, retmax: int, api_key: Optional[str]) -> List[str]:
    """Call NCBI ESearch and return PMIDs as strings."""
    params = {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json", "sort": "relevance"}
    if api_key:
        params["api_key"] = api_key
    url = f"{ESEARCH_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("esearchresult", {}).get("idlist", [])


def parse_pubmed_xml(xml_data: str) -> Dict[str, Dict[str, str]]:
    """Parse a PubMed efetch XML response into {pmid: {title, abstract}}."""
    out: Dict[str, Dict[str, str]] = {}
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        print(f"[WARN] XML parse error: {e}")
        return out

    for article in root.findall(".//PubmedArticle"):
        pmid_elem = article.find(".//PMID")
        if pmid_elem is None or not pmid_elem.text:
            continue
        pmid = pmid_elem.text

        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if (title_elem is not None and title_elem.text) else ""

        parts: List[str] = []
        for txt in article.findall(".//Abstract/AbstractText"):
            label = txt.get("Label", "")
            content = "".join(txt.itertext()).strip()
            parts.append(f"{label}: {content}" if label else content)
        abstract = " ".join(parts).strip()

        out[pmid] = {"title": title, "abstract": abstract}
    return out


def fetch_abstracts_batch(pmids: List[str], api_key: Optional[str]) -> Dict[str, Dict[str, str]]:
    """Fetch a batch of abstracts via NCBI EFetch. Returns {pmid: {title, abstract}}."""
    params = {"db": "pubmed", "id": ",".join(pmids), "rettype": "xml", "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key
    url = f"{EFETCH_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return parse_pubmed_xml(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[WARN] Batch fetch failed: {e}")
        return {}


def download_pmids_from_ncbi(
    pmids: List[str],
    output_dir: Path,
    api_key: Optional[str],
    batch_size: int = 200,
) -> int:
    """Fetch a list of PMIDs from NCBI and write them as .txt files. Returns count written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    delay = 0.34 if api_key else 1.1
    written = 0
    batches = [pmids[i : i + batch_size] for i in range(0, len(pmids), batch_size)]
    for batch in tqdm(batches, desc="Fetching"):
        for pmid, data in fetch_abstracts_batch(batch, api_key).items():
            if write_abstract(output_dir, pmid, data["title"], data["abstract"]):
                written += 1
        time.sleep(delay)
    return written


# =====================================================================
# SUBCOMMAND: gold
# =====================================================================
def cmd_gold(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = get_api_key(args.api_key)

    pmids = json.loads(Path(args.pmids).read_text())
    print(f"[INFO] Loaded {len(pmids)} gold PMIDs from {args.pmids}")

    # Skip already-downloaded PMIDs
    have = existing_pmids(output_dir)
    todo = [p for p in pmids if str(p) not in have]
    print(f"[INFO] {len(have)} already present, will fetch {len(todo)}")

    written = download_pmids_from_ncbi([str(p) for p in todo], output_dir, api_key, args.batch_size)
    print(f"[SUCCESS] Wrote {written} new abstracts to {output_dir}")

    missing = set(str(p) for p in pmids) - existing_pmids(output_dir)
    if missing:
        missing_path = output_dir / "missing_pmids.json"
        missing_path.write_text(json.dumps(sorted(missing), indent=2))
        print(f"[WARN] {len(missing)} PMIDs missing after download. List written to {missing_path}")


# =====================================================================
# SUBCOMMAND: mesh (NCBI search-based negatives)
# =====================================================================
def cmd_mesh(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = get_api_key(args.api_key)

    exclude = load_excluded_pmids(args.exclude) | existing_pmids(output_dir)
    print(f"[INFO] Excluding {len(exclude)} PMIDs (gold + already downloaded)")

    queries = args.query or DEFAULT_MESH_QUERIES
    per_query = (args.num_docs // max(len(queries), 1)) + 1000  # buffer for exclusions

    candidates: List[str] = []
    for q in queries:
        print(f"  searching: {q[:70]}...")
        try:
            found = search_pubmed(q, retmax=per_query, api_key=api_key)
            new = [p for p in found if p not in exclude]
            candidates.extend(new)
            print(f"    +{len(new)} new candidates")
        except Exception as e:
            print(f"    [ERROR] {e}")
        time.sleep(0.5)

    # Deduplicate, shuffle, cap at num_docs
    random.seed(args.seed)
    candidates = list(set(candidates))
    random.shuffle(candidates)
    candidates = candidates[: args.num_docs]
    print(f"[INFO] Fetching {len(candidates)} mesh negatives")

    written = download_pmids_from_ncbi(candidates, output_dir, api_key, args.batch_size)
    print(f"[SUCCESS] Wrote {written} mesh negatives to {output_dir}")
    return written


# =====================================================================
# SUBCOMMAND: hf (HuggingFace MedRAG/pubmed negatives)
# =====================================================================
def cmd_hf(args: argparse.Namespace) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("ERROR: `datasets` library required for the `hf` subcommand. Run: pip install datasets")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude = load_excluded_pmids(args.exclude) | existing_pmids(output_dir)
    print(f"[INFO] Excluding {len(exclude)} PMIDs (gold + already downloaded)")

    if args.min_pmid or args.max_pmid:
        print(f"[INFO] PMID range filter: {args.min_pmid or 0}..{args.max_pmid or '∞'}")

    print("[INFO] Streaming MedRAG/pubmed from HuggingFace (~23.9 M articles)...")
    dataset = load_dataset("MedRAG/pubmed", split="train", streaming=True)

    candidates: List[Dict[str, str]] = []
    seen: Set[str] = set()
    target = args.num_docs * 2  # collect a buffer for random sampling

    for i, example in enumerate(tqdm(dataset, desc="Scanning")):
        pmid = str(example.get("PMID", ""))
        if not pmid or pmid in exclude or pmid in seen:
            continue
        try:
            pmid_int = int(pmid)
        except ValueError:
            continue
        if args.min_pmid and pmid_int < args.min_pmid:
            continue
        if args.max_pmid and pmid_int > args.max_pmid:
            continue

        title = (example.get("title") or "").strip()
        content = (example.get("content") or "").strip()
        if not title and not content:
            continue

        candidates.append({"pmid": pmid, "title": title, "content": content})
        seen.add(pmid)

        if len(candidates) >= target:
            break

    print(f"[INFO] Collected {len(candidates)} valid candidates")
    random.seed(args.seed)
    selected = (
        random.sample(candidates, args.num_docs) if len(candidates) > args.num_docs else candidates
    )
    if len(selected) < args.num_docs:
        print(f"[WARN] Only found {len(selected)} docs (requested {args.num_docs})")

    written = 0
    for doc in tqdm(selected, desc="Writing"):
        if write_abstract(output_dir, doc["pmid"], doc["title"], doc["content"]):
            written += 1
    print(f"[SUCCESS] Wrote {written} HF negatives to {output_dir}")
    return written


# =====================================================================
# SUBCOMMAND: negatives (mixed mesh + hf — the paper config)
# =====================================================================
def cmd_negatives(args: argparse.Namespace) -> None:
    if not 0.0 <= args.mesh_portion <= 1.0:
        raise SystemExit("--mesh_portion must be between 0.0 and 1.0")
    mesh_target = int(args.total * args.mesh_portion)
    hf_target = args.total - mesh_target

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MIXED NEGATIVE SAMPLING")
    print(f"  total target  : {args.total:,}")
    print(f"  mesh (NCBI)   : {mesh_target:,}  ({args.mesh_portion*100:.0f}%)")
    print(f"  hf (MedRAG)   : {hf_target:,}  ({(1-args.mesh_portion)*100:.0f}%)")
    print("=" * 70)

    if mesh_target > 0:
        ns = argparse.Namespace(
            output=str(output_dir),
            num_docs=mesh_target,
            exclude=args.exclude,
            api_key=args.api_key,
            batch_size=args.batch_size,
            query=None,
            seed=args.seed,
        )
        cmd_mesh(ns)

    # HF fills whatever is still missing toward the total. Count only non-gold files.
    gold_pmids = load_excluded_pmids(args.exclude)
    negatives_so_far = len(existing_pmids(output_dir) - gold_pmids)
    remaining = max(0, args.total - negatives_so_far)
    print(f"[INFO] {negatives_so_far:,} negatives on disk, need {remaining:,} more")

    if remaining > 0 and hf_target > 0:
        ns = argparse.Namespace(
            output=str(output_dir),
            num_docs=min(hf_target, remaining),
            exclude=args.exclude,
            seed=args.seed,
            min_pmid=args.min_pmid,
            max_pmid=args.max_pmid,
        )
        cmd_hf(ns)

    final = len(existing_pmids(output_dir)) - len(existing_pmids(output_dir) & load_excluded_pmids(args.exclude))
    print(f"[DONE] Negatives on disk: {final:,} / target {args.total:,}")


# =====================================================================
# SUBCOMMAND: all (gold + negatives)
# =====================================================================
def cmd_all(args: argparse.Namespace) -> None:
    print(">>> Phase 1/2: gold")
    cmd_gold(args)
    print("\n>>> Phase 2/2: negatives")
    cmd_negatives(args)


# =====================================================================
# CLI
# =====================================================================
def _add_common_io_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output", default=str(DEFAULT_CORPUS_DIR),
                   help=f"Corpus directory (default: {DEFAULT_CORPUS_DIR})")
    p.add_argument("--exclude", default=str(DEFAULT_PMIDS_FILE),
                   help=f"JSON list of gold PMIDs to exclude (default: {DEFAULT_PMIDS_FILE})")


def _add_ncbi_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--api_key", default=None, help="NCBI API key (or set NCBI_API_KEY)")
    p.add_argument("--batch_size", type=int, default=200, help="PMIDs per request (max 200)")


def _add_hf_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--min_pmid", type=int, default=None, help="Only include PMID >= this value")
    p.add_argument("--max_pmid", type=int, default=None, help="Only include PMID <= this value")


def main() -> None:
    parser = argparse.ArgumentParser(description="MIRAGE corpus downloader (unified)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gold", help="Download gold abstracts from NCBI by PMID list")
    p.add_argument("--pmids", default=str(DEFAULT_PMIDS_FILE),
                   help=f"JSON list of PMIDs to fetch (default: {DEFAULT_PMIDS_FILE})")
    p.add_argument("--output", default=str(DEFAULT_CORPUS_DIR),
                   help=f"Corpus directory (default: {DEFAULT_CORPUS_DIR})")
    _add_ncbi_args(p)
    p.set_defaults(func=cmd_gold)

    p = sub.add_parser("mesh", help="Download MeSH-based hard negatives from NCBI")
    _add_common_io_args(p)
    _add_ncbi_args(p)
    p.add_argument("--num_docs", type=int, default=500000)
    p.add_argument("--query", nargs="+", default=None,
                   help="Custom MeSH queries (default: 15 paper-faithful queries)")
    p.add_argument("--seed", type=int, default=2025)
    p.set_defaults(func=cmd_mesh)

    p = sub.add_parser("hf", help="Download random negatives from MedRAG/pubmed (HuggingFace)")
    _add_common_io_args(p)
    p.add_argument("--num_docs", type=int, default=500000)
    p.add_argument("--seed", type=int, default=2025)
    _add_hf_args(p)
    p.set_defaults(func=cmd_hf)

    p = sub.add_parser("negatives", help="Mixed mesh+hf negatives (paper config: 50/50)")
    _add_common_io_args(p)
    _add_ncbi_args(p)
    _add_hf_args(p)
    p.add_argument("--total", type=int, default=1_000_000,
                   help="Total negatives to download (default: 1,000,000 — the paper config)")
    p.add_argument("--mesh_portion", type=float, default=0.5,
                   help="Fraction sourced from NCBI MeSH search (default: 0.5)")
    p.add_argument("--seed", type=int, default=2025)
    p.set_defaults(func=cmd_negatives)

    p = sub.add_parser("all", help="Run `gold` then `negatives` end-to-end")
    p.add_argument("--pmids", default=str(DEFAULT_PMIDS_FILE))
    _add_common_io_args(p)
    _add_ncbi_args(p)
    _add_hf_args(p)
    p.add_argument("--total", type=int, default=1_000_000)
    p.add_argument("--mesh_portion", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2025)
    p.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
