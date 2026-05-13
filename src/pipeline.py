"""
English Hybrid RAG Retrieval Pipeline - Dual RRF (k=0 and k=60)
================================================================
For MIRAGE benchmark evaluation (PubMedQA + BioASQ).

Implements per-retriever reranking strategy with DUAL RRF:
  1. Vector Search → Rerank → Keep all reranked
  2. BM25 Search → Rerank → Keep all reranked
  3. HyDE Search → Rerank → K
  eep all reranked
  4. Combine using RRF with k=0
  5. Combine using RRF with k=60
  6. Final rerank BOTH versions
"""

from typing import List, Tuple, Dict, Optional
from os import getenv
from dotenv import load_dotenv
from pathlib import Path
import sys
import copy

import chromadb
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.llms.openrouter import OpenRouter
from llama_index.core import Settings
from llama_index.retrievers.bm25 import BM25Retriever


# =============================================================================
# CONFIGURATION
# =============================================================================
DEBUG_VERBOSE = False

# Per-retriever reranking settings
USE_PER_RETRIEVER_RERANKING = True
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
PER_RETRIEVER_TOP_N = 50

# Final reranking (after RRF) settings
USE_FINAL_RERANKER = True
FINAL_RERANKER_TOP_N = 50
RERANKER_QUERY = "original"

# Project root (one level above src/) — used as default for embeddings & .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings" / "mxbai-embed-large-v1"
CHROMA_COLLECTION_NAME = "pubmed_mirage"


def debug_print(*args, **kwargs):
    """Conditional print for debugging."""
    if DEBUG_VERBOSE:
        print(*args, **kwargs)


def get_filename(node: NodeWithScore) -> str:
    """Safely extract filename from node metadata."""
    # For PubMed corpus, we use 'pmid' or 'filename'
    return (node.metadata.get("pmid") or
            node.metadata.get("filename") or
            node.metadata.get("file_name") or
            "<unknown>")


def get_english_index(model_name: str, embeddings_dir: Optional[Path] = None):
    """Load English PubMed index from local embeddings.

    Args:
        model_name: HuggingFace embedding model name (kept for API parity).
        embeddings_dir: Directory holding chroma_db/ and chromadb_index/.
            Defaults to ``<project_root>/embeddings/mxbai-embed-large-v1``.
    """
    embeddings_dir = Path(embeddings_dir) if embeddings_dir else DEFAULT_EMBEDDINGS_DIR
    chroma_db_path = embeddings_dir / "chroma_db"
    index_storage_path = embeddings_dir / "chromadb_index"
    debug_print(f"[INDEX] Loading English index from {embeddings_dir}")

    db = chromadb.PersistentClient(path=str(chroma_db_path))
    collection = db.get_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(index_storage_path),
    )
    index = load_index_from_storage(storage_context)

    debug_print(f"[INDEX] Loaded index with {len(index.docstore.docs)} documents in docstore")
    return index


# =============================================================================
# HYDE WITH CONTEXT
# =============================================================================
class HyDocWithContext(HyDEQueryTransform):
    """HyDE that uses retrieved context to generate better hypothetical documents."""

    def __init__(self, llm, hyde_prompt, include_original=True, context=""):
        super().__init__(llm=llm, hyde_prompt=hyde_prompt, include_original=include_original)
        self.context = context

    def _run(self, query_bundle: QueryBundle) -> Tuple[str, QueryBundle]:
        query_str = query_bundle.query_str
        hypothetical_doc = self._llm.predict(
            self._hyde_prompt,
            query_str=query_str,
            context=self.context
        )
        embedding_strs = [hypothetical_doc]
        if self._include_original:
            embedding_strs.extend(query_bundle.embedding_strs)
        return hypothetical_doc, QueryBundle(
            query_str=query_str,
            custom_embedding_strs=embedding_strs
        )


# =============================================================================
# MAIN RETRIEVER WITH DUAL RRF
# =============================================================================
class ChatFAQRetriever(BaseRetriever):
    """
    Hybrid retriever combining Vector Search + BM25 + HyDE.

    NEW: Computes RRF with BOTH k=0 and k=60, reranks both.
    Supports pre-generated HyDE docs for faster evaluation.
    """

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        hyde_prompt: PromptTemplate,
        hydoc_llm: OpenRouter = None,
        reranker: Optional[SentenceTransformerRerank] = None,
        precomputed_hyde: str = None,  # Pre-generated HyDE doc for this query
    ):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.hydoc_llm = hydoc_llm
        self.hyde_prompt = hyde_prompt
        self.reranker = reranker
        self.precomputed_hyde = precomputed_hyde

    def _compute_rrf_scores(
        self,
        ranked_lists: Dict[str, List[NodeWithScore]],
        rrf_k: int = 60,
        weights: Dict[str, float] = None
    ) -> Dict[str, Tuple[float, NodeWithScore]]:
        """
        Compute RRF scores with specified k value.
        """
        if weights is None:
            weights = {"vector": 1.0, "bm25": 1.0, "hyde": 1.0}

        rrf_scores: Dict[str, float] = {}
        best_nodes: Dict[str, NodeWithScore] = {}

        for source_name, nodes in ranked_lists.items():
            weight = weights.get(source_name, 1.0)
            seen_in_source = set()

            for rank, node in enumerate(nodes, start=1):
                filename = get_filename(node).lower().replace('.txt', '').strip()
                if not filename or filename == "<unknown>":
                    continue

                if filename in seen_in_source:
                    continue
                seen_in_source.add(filename)

                rrf_contribution = weight * (1.0 / (rrf_k + rank))

                if filename not in rrf_scores:
                    rrf_scores[filename] = 0.0
                    best_nodes[filename] = node
                rrf_scores[filename] += rrf_contribution

        return {fn: (score, best_nodes[fn]) for fn, score in rrf_scores.items()}

    def _get_full_results(self, nodes: List[NodeWithScore], n: int = 50) -> List[dict]:
        """Get top N documents with scores, NO deduplication."""
        result = []
        for i, node in enumerate(nodes[:n]):
            filename = get_filename(node).lower().replace('.txt', '').strip()
            if filename and filename != "<unknown>":
                result.append({
                    "rank": i + 1,
                    "doc": filename,
                    "score": float(node.score) if node.score else 0.0,
                    "node_id": node.node.node_id
                })
        return result

    def _rerank_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle,
        top_n: int = 50
    ) -> List[NodeWithScore]:
        """Rerank a list of nodes using the shared reranker instance."""
        if self.reranker is None or not USE_PER_RETRIEVER_RERANKING:
            return nodes
        original_top_n = self.reranker.top_n
        self.reranker.top_n = top_n
        result = self.reranker._postprocess_nodes(nodes, query_bundle)
        self.reranker.top_n = original_top_n
        return result

    def _rrf_to_nodes(self, rrf_results: Dict[str, Tuple[float, NodeWithScore]]) -> List[NodeWithScore]:
        """Convert RRF results dict to sorted list of nodes."""
        sorted_results = sorted(rrf_results.items(), key=lambda x: x[1][0], reverse=True)
        nodes = []
        for filename, (rrf_score, node) in sorted_results:
            # Create a copy to avoid modifying original
            node_copy = NodeWithScore(node=node.node, score=rrf_score)
            nodes.append(node_copy)
        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> Tuple[List[NodeWithScore], str, dict]:
        """
        Per-Retriever Reranking with DUAL RRF (k=0 and k=60):
        1. Vector → Rerank
        2. BM25 → Rerank
        3. HyDE → Rerank
        4. RRF k=0 → Final rerank
        5. RRF k=60 → Final rerank
        """
        debug_print(f"\n{'='*60}")
        debug_print(f"[Dual RRF] Query: {query_bundle.query_str[:80]}...")
        debug_print('='*60)

        intermediate_results = {}

        # ---------------------------------------------------------------------
        # STEP 1: VECTOR SEARCH + RERANK
        # ---------------------------------------------------------------------
        debug_print("\n[STEP 1] Vector Search...")
        vector_nodes = self.vector_retriever._retrieve(query_bundle)
        debug_print(f"   Retrieved {len(vector_nodes)} nodes")

        intermediate_results['vector_top50'] = self._get_full_results(vector_nodes, n=50)

        if USE_PER_RETRIEVER_RERANKING and self.reranker is not None:
            debug_print("   Reranking vector results...")
            vector_nodes_reranked = self._rerank_nodes(vector_nodes, query_bundle, PER_RETRIEVER_TOP_N)
            intermediate_results['vector_reranked_top50'] = self._get_full_results(vector_nodes_reranked, n=50)
        else:
            vector_nodes_reranked = vector_nodes
            intermediate_results['vector_reranked_top50'] = intermediate_results['vector_top50']

        # ---------------------------------------------------------------------
        # STEP 2: BM25 SEARCH + RERANK
        # ---------------------------------------------------------------------
        debug_print("\n[STEP 2] BM25 Search...")
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        debug_print(f"   Retrieved {len(bm25_nodes)} nodes")

        intermediate_results['bm25_top50'] = self._get_full_results(bm25_nodes, n=50)

        if USE_PER_RETRIEVER_RERANKING and self.reranker is not None:
            debug_print("   Reranking BM25 results...")
            bm25_nodes_reranked = self._rerank_nodes(bm25_nodes, query_bundle, PER_RETRIEVER_TOP_N)
            intermediate_results['bm25_reranked_top50'] = self._get_full_results(bm25_nodes_reranked, n=50)
        else:
            bm25_nodes_reranked = bm25_nodes
            intermediate_results['bm25_reranked_top50'] = intermediate_results['bm25_top50']

        # ---------------------------------------------------------------------
        # STEP 3: BUILD CONTEXT FOR HYDE
        # ---------------------------------------------------------------------
        context_candidates = []
        seen_ids = set()

        for node in (bm25_nodes_reranked[:5] + vector_nodes_reranked[:5]):
            if node.node.node_id not in seen_ids:
                context_candidates.append(node)
                seen_ids.add(node.node.node_id)

        contexts = [n.text for n in context_candidates[:8]]
        context_snip = "\n---\n".join(contexts)
        debug_print(f"\n[STEP 3] Built HyDE context from {len(contexts)} documents")

        # Save context documents (with node_ids) for HyDE
        intermediate_results['hyde_context_docs'] = [
            {
                'doc': get_filename(node).lower().replace('.txt', '').strip(),
                'node_id': node.node.node_id,
                'text': node.text
            }
            for node in context_candidates[:8]
        ]

        # ---------------------------------------------------------------------
        # STEP 4: HYDE GENERATION + RETRIEVAL + RERANK
        # ---------------------------------------------------------------------
        debug_print("\n[STEP 4] HyDE Generation & Retrieval...")

        # Use pre-computed HyDE doc if available, otherwise generate
        if self.precomputed_hyde:
            hydoc = self.precomputed_hyde
            debug_print(f"   Using pre-computed HyDoc: {hydoc[:100]}...")
        else:
            self.hyde = HyDocWithContext(
                llm=self.hydoc_llm,
                hyde_prompt=self.hyde_prompt,
                include_original=False,
                context=context_snip
            )
            hydoc, hyde_bundle = self.hyde._run(query_bundle)
            debug_print(f"   Generated HyDoc: {hydoc[:100]}...")

        # Retrieve using HyDE doc as query
        hyde_nodes = []
        hyde_qb = QueryBundle(query_str=hydoc)
        hyde_nodes.extend(self.vector_retriever._retrieve(hyde_qb))
        hyde_nodes.sort(key=lambda x: x.score, reverse=True)
        debug_print(f"   HyDE retrieved {len(hyde_nodes)} nodes")

        intermediate_results['hyde_top50'] = self._get_full_results(hyde_nodes, n=50)

        if USE_PER_RETRIEVER_RERANKING and self.reranker is not None:
            debug_print("   Reranking HyDE results...")
            hyde_nodes_reranked = self._rerank_nodes(hyde_nodes, query_bundle, PER_RETRIEVER_TOP_N)
            intermediate_results['hyde_reranked_top50'] = self._get_full_results(hyde_nodes_reranked, n=50)
        else:
            hyde_nodes_reranked = hyde_nodes
            intermediate_results['hyde_reranked_top50'] = intermediate_results['hyde_top50']

        # ---------------------------------------------------------------------
        # STEP 5: DUAL RRF - k=0 and k=60
        # ---------------------------------------------------------------------
        debug_print(f"\n[STEP 5] Applying DUAL RRF (k=0 and k=60)...")

        ranked_lists = {
            "vector": vector_nodes_reranked,
            "bm25": bm25_nodes_reranked,
            "hyde": hyde_nodes_reranked
        }

        # RRF with k=0
        rrf_k0_results = self._compute_rrf_scores(ranked_lists, rrf_k=0)
        rrf_k0_nodes = self._rrf_to_nodes(rrf_k0_results)
        intermediate_results['rrf_k0_top50'] = self._get_full_results(rrf_k0_nodes, n=50)

        # RRF with k=60
        rrf_k60_results = self._compute_rrf_scores(ranked_lists, rrf_k=60)
        rrf_k60_nodes = self._rrf_to_nodes(rrf_k60_results)
        intermediate_results['rrf_k60_top50'] = self._get_full_results(rrf_k60_nodes, n=50)

        debug_print(f"   RRF k=0: {len(rrf_k0_nodes)} docs")
        debug_print(f"   RRF k=60: {len(rrf_k60_nodes)} docs")

        # ---------------------------------------------------------------------
        # STEP 6: FINAL RERANKER for BOTH RRF versions
        # ---------------------------------------------------------------------
        if USE_FINAL_RERANKER and self.reranker is not None:
            debug_print(f"\n[STEP 6] Applying Final Reranker to both RRF versions...")

            reranker_query_bundle = query_bundle  # Always use original query

            original_top_n = self.reranker.top_n
            self.reranker.top_n = FINAL_RERANKER_TOP_N

            # Rerank RRF k=0
            final_k0_nodes = self.reranker._postprocess_nodes(rrf_k0_nodes, reranker_query_bundle)
            intermediate_results['final_k0_top50'] = self._get_full_results(final_k0_nodes, n=50)

            # Rerank RRF k=60
            final_k60_nodes = self.reranker._postprocess_nodes(rrf_k60_nodes, reranker_query_bundle)
            intermediate_results['final_k60_top50'] = self._get_full_results(final_k60_nodes, n=50)

            self.reranker.top_n = original_top_n

            debug_print(f"   Final k=0: {len(final_k0_nodes)} docs")
            debug_print(f"   Final k=60: {len(final_k60_nodes)} docs")

            # Return k=0 final as the primary result (for backward compatibility)
            return final_k0_nodes, hydoc, intermediate_results

        # If no final reranker
        intermediate_results['final_k0_top50'] = intermediate_results['rrf_k0_top50']
        intermediate_results['final_k60_top50'] = intermediate_results['rrf_k60_top50']

        return rrf_k0_nodes, hydoc, intermediate_results


# =============================================================================
# RAG PIPELINE FOR ENGLISH
# =============================================================================
class RAGpipeline:
    """Main RAG pipeline for English PubMed evaluation."""

    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        hyde_prompt=None,
        embeddings_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.hyde_prompt = hyde_prompt
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else DEFAULT_EMBEDDINGS_DIR

        # Load .env from project root (one level above src/)
        load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

        # OpenRouter for HyDE generation
        self.openrouter_model = OpenRouter(
            api_key=getenv("OPENROUTER_API_KEY"),
            max_tokens=500,
            context_window=4096,
            model="meta-llama/llama-3.3-70b-instruct",
            seed=2025
        )

        # Build index and retrievers
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        self.index = get_english_index(self.model_name, embeddings_dir=self.embeddings_dir)
        self.vector_retriever = VectorIndexRetriever(index=self.index, similarity_top_k=50)

        # Check docstore
        doc_count = len(self.index.docstore.docs)
        print(f"[INIT] Docstore document count: {doc_count}")
        if doc_count == 0:
            print("!!! CRITICAL WARNING: Docstore is empty. BM25 cannot work!")

        # BM25 Retriever - English language
        print("[INIT] Initializing BM25 Retriever (English)...")
        self.bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.index.docstore,
            similarity_top_k=50,
            language="en"  # English for PubMed
        )

        # Reranker
        if USE_PER_RETRIEVER_RERANKING or USE_FINAL_RERANKER:
            print(f"[INIT] Loading reranker: {RERANKER_MODEL}")
            self.reranker = SentenceTransformerRerank(
                model=RERANKER_MODEL,
                top_n=PER_RETRIEVER_TOP_N,
                device="cuda"
            )
        else:
            self.reranker = None

        print(f"[INIT] Pipeline ready. Per-retriever rerank: {USE_PER_RETRIEVER_RERANKING}, Final rerank: {USE_FINAL_RERANKER}")

    def retrieve_documents(self, query: str, precomputed_hyde: str = None) -> Tuple:
        """
        Retrieve documents for a query.

        Args:
            query: The question/query string
            precomputed_hyde: Optional pre-generated HyDE doc to skip LLM call
        """
        retriever = ChatFAQRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            hyde_prompt=self.hyde_prompt,
            hydoc_llm=self.openrouter_model,
            reranker=self.reranker,
            precomputed_hyde=precomputed_hyde,
        )
        query_bundle = QueryBundle(query)

        documents, hydoc, intermediate_results = retriever._retrieve(query_bundle)

        origin_files, scores, pieces = [], [], []
        for doc in documents:
            origin_files.append(get_filename(doc))
            scores.append(doc.score)
            pieces.append(doc.text)

        return origin_files, scores, pieces, hydoc, intermediate_results
