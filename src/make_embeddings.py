"""
Create embeddings for PubMed corpus (MIRAGE evaluation).
"""

import argparse
from pathlib import Path
from llama_index.core import Document
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "corpus"
DEFAULT_EMBEDDINGS_ROOT = PROJECT_ROOT / "embeddings"
DEFAULT_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"


def get_data(corpus_dir: str):
    """
    Load documents from corpus directory (one .txt file per PMID).
    Each file contains: title\n\nabstract
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

    documents = []
    txt_files = list(corpus_path.glob("*.txt"))

    print(f"[INFO] Found {len(txt_files)} .txt files in {corpus_dir}")

    for txt_file in txt_files:
        pmid = txt_file.stem  # filename without extension
        text = txt_file.read_text(encoding='utf-8')

        doc = Document(
            text=text,
            metadata={
                "filename": txt_file.name,  # e.g., "12345678.txt"
                "pmid": pmid,
                "doc_type": "pubmed_abstract"
            }
        )
        documents.append(doc)

    return documents


def get_embeddings(corpus_dir: str, model_name: str, output_dir: str = None):
    """
    Create embeddings and index for the corpus.

    Args:
        corpus_dir: Directory containing .txt files
        model_name: HuggingFace model name (e.g., "mixedbread-ai/mxbai-embed-large-v1")
        output_dir: Output directory for index (default: embeddings/<model_name>)
    """
    # Initialize embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)

    # Text splitter - PubMed abstracts are short, so minimal chunking needed
    # Most abstracts are ~1500 chars, well under typical chunk sizes
    text_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=64  # Less overlap needed for short docs
    )

    parsed_model_name = model_name.split("/")[-1].lower()

    # Set output directory
    if output_dir is None:
        output_dir = str(DEFAULT_EMBEDDINGS_ROOT / parsed_model_name)
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load documents
    documents = get_data(corpus_dir)
    print(f"[INFO] Loaded {len(documents)} source documents")

    # Initialize ChromaDB
    db_path = f"{output_dir}/chroma_db"
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("pubmed_mirage")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index with transformations
    print(f"[INFO] Creating embeddings with {model_name}...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        store_nodes_override=True,
        transformations=[text_splitter],
        show_progress=True
    )

    # Persist the index (includes docstore for BM25)
    index_dir = f"{output_dir}/chromadb_index"
    index.storage_context.persist(persist_dir=index_dir)

    # Verify
    doc_count = len(index.docstore.docs)
    print(f"[INFO] Index created with {doc_count} chunks")

    if doc_count == 0:
        print("[WARNING] Docstore is empty! BM25 will not work.")
    else:
        print(f"[SUCCESS] Saved index to {output_dir}/")
        print(f"  - ChromaDB: {db_path}")
        print(f"  - Docstore: {index_dir}")

    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embeddings for PubMed corpus")
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(DEFAULT_CORPUS_DIR),
        help=f"Directory containing .txt files (default: {DEFAULT_CORPUS_DIR})"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace embedding model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output directory (default: {DEFAULT_EMBEDDINGS_ROOT}/<model_name>)"
    )

    args = parser.parse_args()

    try:
        get_embeddings(args.corpus, args.model_name, args.output)
    except Exception as e:
        print(f"Error: {e}")
        raise
