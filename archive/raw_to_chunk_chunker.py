import os
import json
import asyncio
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import re

from dotenv import load_dotenv
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import DoclingDocument
from docling.chunking import HybridChunker

# ------------------------
# 🌍 CONFIGURATION
# ------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dynamically load all documents from the /documents directory
DOCUMENTS_DIR = Path("documents")
DOCUMENTS = [
    str(p)
    for p in DOCUMENTS_DIR.glob("*")
    if p.is_file() and not p.name.startswith(".")
]

if not DOCUMENTS:
    logger.warning(f"No documents found in {DOCUMENTS_DIR.resolve()}")
else:
    logger.info(f"Found {len(DOCUMENTS)} documents in {DOCUMENTS_DIR.resolve()}")
    for doc in DOCUMENTS:
        logger.info(f" - {doc}")


# Folders
OUTPUT_FOLDER = "chunks_and_metadata"
MD_OUTPUT_FOLDER = "converted_markdown_docs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MD_OUTPUT_FOLDER, exist_ok=True)


# ------------------------
# 🧠 DATA CLASSES
# ------------------------

@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    use_semantic_splitting: bool = True
    preserve_structure: bool = True
    max_tokens: int = 400


@dataclass
class DocumentChunk:
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None


# ------------------------
# 🔍 Markdown Header Utilities
# ------------------------

def extract_markdown_sections(content: str) -> List[Dict[str, Any]]:
    pattern = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)
    headers = []
    matches = list(pattern.finditer(content))
    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if (i + 1) < len(matches) else len(content)
        headers.append({"level": level, "title": title, "start": start, "end": end})
    return headers


def find_section_for_position(sections: List[Dict[str, Any]], pos: int) -> Optional[str]:
    for sec in sections:
        if sec["start"] <= pos < sec["end"]:
            return sec["title"]
    # fallback: look for the nearest previous header
    for sec in reversed(sections):
        if sec["start"] <= pos:
            return sec["title"]
    return None


# ------------------------
# 🔩 Convert Raw → Markdown
# ------------------------

def convert_raw_to_markdown(file_path: str, converter: DocumentConverter) -> Optional[Path]:
    """
    Convert a document (pdf / docx / etc) to Markdown.
    Returns Path to the markdown file, or None on error.
    """
    try:
        print(f"\n📄 Converting: {Path(file_path).name}")
        result = converter.convert(file_path)
        markdown = result.document.export_to_markdown()

        output_md_filename = f"markdown_{Path(file_path).stem}.md"
        output_path = Path(MD_OUTPUT_FOLDER) / output_md_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"   ✓ Saved Markdown: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"✗ Error converting {file_path}: {e}")
        return None


# ------------------------
# 🔧 Convert Markdown → DoclingDocument
# ------------------------

def convert_markdown_to_docling(content: str, name: str) -> Optional[DoclingDocument]:
    converter = DocumentConverter()
    try:
        format_attr = InputFormat.MD  # use MD enum
        conv_result = converter.convert_string(content=content, format=format_attr, name=name)
        return conv_result.document
    except Exception as e:
        logger.warning(f"⚠️ Failed to convert markdown to DoclingDocument ({e})")
        return None


# ------------------------
# 🧩 Chunker Class
# ------------------------

class DoclingHybridChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config
        tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Initializing tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, device_map="auto")
        self.chunker = HybridChunker(tokenizer=self.tokenizer, max_tokens=config.max_tokens, merge_peers=True)

    async def chunk_document(
        self, content: str, title: str, source: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "hybrid",
            **(metadata or {}),
        }

        # Convert markdown text into DoclingDocument
        docling_doc = convert_markdown_to_docling(content, title)
        if docling_doc is None:
            logger.warning("DoclingDocument conversion returned None — no chunks returned.")
            return []

        sections = extract_markdown_sections(content)
        chunk_iter = self.chunker.chunk(dl_doc=docling_doc)
        chunks_list = list(chunk_iter)
        total_chunks = len(chunks_list)

        results: List[DocumentChunk] = []
        current_pos = 0

        for i, chunk in enumerate(chunks_list):
            text = self.chunker.contextualize(chunk=chunk)
            token_count = len(self.tokenizer.encode(text))
            word_count = len(text.split())
            section_title = None

            if hasattr(chunk, "source_node") and chunk.source_node:
                section_title = getattr(chunk.source_node, "title", None)

            if not section_title or section_title.lower().startswith("unknown"):
                section_title = find_section_for_position(sections, current_pos)

            if section_title is None:
                section_title = "Unknown Section"

            results.append(
                DocumentChunk(
                    content=text.strip(),
                    index=i,
                    start_char=current_pos,
                    end_char=current_pos + len(text),
                    metadata={
                        **base_metadata,
                        "chunk_index": i,
                        "token_count": token_count,
                        "word_count": word_count,       
                        "total_chunks": total_chunks,   
                        "section_title": section_title,
                        "has_context": True,
                    },
                    token_count=token_count,
                )
            )
            current_pos += len(text)

        return results


# ------------------------
# 💾 Export Function
# ------------------------

def export_chunks_and_metadata(chunks: List[DocumentChunk], filename_prefix: str):
    txt_path = os.path.join(OUTPUT_FOLDER, f"chunks_{filename_prefix}.txt")
    json_path = os.path.join(OUTPUT_FOLDER, f"metadata_{filename_prefix}.json")

    with open(txt_path, "w", encoding="utf-8") as f_txt:
        for chunk in chunks:
            f_txt.write(" ".join(chunk.content.split()) + "\n")

    metadata_list = [chunk.metadata for chunk in chunks]
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(metadata_list, f_json, ensure_ascii=False, indent=2)

    print(f"✅ Saved chunks to {txt_path}")
    print(f"✅ Saved metadata to {json_path}")


# ------------------------
# 🚀 MAIN PIPELINE
# ------------------------

# ------------------------
# 🚀 MAIN PIPELINE
# ------------------------

async def main(documents: List[str]):
    print("=" * 60)
    print(" RAW → MARKDOWN → CHUNKING PIPELINE ")
    print("=" * 60)

    converter = DocumentConverter()
    config = ChunkingConfig()
    chunker = DoclingHybridChunker(config)

    summary = []

    for file_path in documents:
        doc_name = Path(file_path).stem
        md_filename = f"markdown_{doc_name}.md"
        md_path = Path(MD_OUTPUT_FOLDER) / md_filename

        # ✅ Check if markdown already exists
        if md_path.exists():
            print(f"\n⏭️ Skipping conversion for {file_path} — Markdown already exists at {md_path}")
        else:
            # Convert if markdown is missing
            md_path = convert_raw_to_markdown(file_path, converter)
            if md_path is None or not md_path.exists():
                logger.warning(f"Skipping {file_path}: markdown conversion failed.")
                continue

        # Proceed with chunking
        print(f"\n🔹 Chunking Markdown: {md_path}")
        content = md_path.read_text(encoding="utf-8")
        prefix = md_path.stem

        chunks = await chunker.chunk_document(content=content, title=prefix, source=str(md_path))
        export_chunks_and_metadata(chunks, prefix)

        summary.append({
            "original_file": file_path,
            "markdown_file": str(md_path),
            "num_chunks": len(chunks)
        })

    print("\n" + "=" * 60)
    print(" PIPELINE SUMMARY ")
    print("=" * 60)
    for r in summary:
        print(f"📘 {r['original_file']} → {r['num_chunks']} chunks (md: {r['markdown_file']})")


if __name__ == "__main__":
    asyncio.run(main(DOCUMENTS))
