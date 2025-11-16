# simple, paragraph or hybrid (semantic) chunker. Chunks and saves text as .txt and metadata as .json
# original file from https://github.com/coleam00/ottomator-agents/tree/main/docling-rag-agent/ingestion 

import os
import logging
import json
import asyncio
import time
import uuid
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument

# ✅ Added imports for Markdown conversion
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# List your markdown files here
md_files = [
    "vault.md",
    # "notes.md",
    # "todo.md"
]

# Output folder
OUTPUT_FOLDER = "chunks_and_metadata"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------------------------------------------------
# 🔧 CONFIG & DATA CLASSES
# ---------------------------------------------------------

@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    use_semantic_splitting: bool = True
    preserve_structure: bool = True
    max_tokens: int = 512

    def __post_init__(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")


@dataclass
class DocumentChunk:
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.token_count is None:
            self.token_count = len(self.content) // 4


# ---------------------------------------------------------
# 🧠 DOC CONVERTER: Markdown → DoclingDocument
# ---------------------------------------------------------

def convert_markdown_to_docling(content: str, name: str) -> Optional[DoclingDocument]:
    """Try to convert markdown text into a DoclingDocument, auto-detecting format constant."""
    converter = DocumentConverter()

    # Dynamically detect supported InputFormat for Markdown
    format_attr = None
    for candidate in ["MARKDOWN", "Markdown", "MD"]:
        if hasattr(InputFormat, candidate):
            format_attr = getattr(InputFormat, candidate)
            break

    if format_attr is None:
        # Some versions require a MIME type instead
        format_attr = "text/markdown"

    logger.info(f"Converting markdown file '{name}' into DoclingDocument...")

    try:
        conv_result = converter.convert_string(content, format=format_attr, name=name)
        return conv_result.document
    except Exception as e:
        logger.warning(f"⚠️ Failed to convert markdown to DoclingDocument ({e}). Falling back to simple chunking.")
        return None


# ---------------------------------------------------------
# 🧭 HEADER-BASED SECTION EXTRACTION
# ---------------------------------------------------------

def extract_markdown_sections(content: str) -> List[Dict[str, Any]]:
    """
    Extract sections based on Markdown headers (#, ##, ###, etc.).
    Returns a list of dicts: {level, title, start, end}.
    """
    pattern = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)
    headers = []
    matches = list(pattern.finditer(content))
    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        headers.append({
            "level": level,
            "title": title,
            "start": start,
            "end": end
        })
    return headers


def find_section_for_position(sections: List[Dict[str, Any]], pos: int) -> Optional[str]:
    """Find the nearest preceding section title for a given character position."""
    for sec in sections:
        if sec["start"] <= pos < sec["end"]:
            return sec["title"]
    # fallback: nearest previous header
    for sec in reversed(sections):
        if sec["start"] <= pos:
            return sec["title"]
    return None


# ---------------------------------------------------------
# 🧩 CHUNKERS
# ---------------------------------------------------------

class DoclingHybridChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Initializing tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=config.max_tokens,
            merge_peers=True
        )
        logger.info(f"HybridChunker initialized (max_tokens={config.max_tokens})")

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        docling_doc: Optional[DoclingDocument] = None
    ) -> List[DocumentChunk]:

        if not content.strip():
            return []

        # 🧩 Add file-level metadata
        file_size = os.path.getsize(source) if os.path.exists(source) else None
        last_modified = (
            time.ctime(os.path.getmtime(source)) if os.path.exists(source) else None
        )

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "hybrid",
            "file_size_bytes": file_size,
            "last_modified": last_modified,
            **(metadata or {}),
        }

        # ✅ Try to create a DoclingDocument automatically if not provided
        if docling_doc is None:
            docling_doc = convert_markdown_to_docling(content, title)
            if docling_doc is None:
                return self._simple_fallback_chunk(content, base_metadata)

        # 🔍 Extract markdown-based section titles (fallback)
        sections = extract_markdown_sections(content)

        try:
            chunk_iter = self.chunker.chunk(dl_doc=docling_doc)
            chunks = list(chunk_iter)
            document_chunks = []
            current_pos = 0

            for i, chunk in enumerate(chunks):
                contextualized_text = self.chunker.contextualize(chunk=chunk)
                token_count = len(self.tokenizer.encode(contextualized_text))
                word_count = len(contextualized_text.split())

                # 🧠 Try to extract section title (Docling → Markdown fallback)
                section_title = None
                if hasattr(chunk, "source_node") and chunk.source_node:
                    section_title = getattr(chunk.source_node, "title", None)
                if not section_title and hasattr(chunk, "section_title"):
                    section_title = chunk.section_title
                if not section_title or section_title.strip().lower() == "unknown section":
                    section_title = find_section_for_position(sections, current_pos)
                section_title = section_title or "Unknown Section"

                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "token_count": token_count,
                    "word_count": word_count,
                    "section_title": section_title,
                    "has_context": True,
                }

                start_char = current_pos
                end_char = start_char + len(contextualized_text)

                document_chunks.append(
                    DocumentChunk(
                        content=contextualized_text.strip(),
                        index=i,
                        start_char=start_char,
                        end_char=end_char,
                        metadata=chunk_metadata,
                        token_count=token_count,
                    )
                )
                current_pos = end_char

            logger.info(f"Created {len(document_chunks)} chunks using HybridChunker")
            return document_chunks

        except Exception as e:
            logger.error(f"HybridChunker failed: {e}, falling back to simple chunking")
            return self._simple_fallback_chunk(content, base_metadata)

    # ---------------------------------------------------------

    def _simple_fallback_chunk(self, content: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        start = 0
        chunk_index = 0

        # Also extract markdown section headers for fallback
        sections = extract_markdown_sections(content)

        while start < len(content):
            end = start + chunk_size
            if end >= len(content):
                chunk_text = content[start:]
            else:
                chunk_end = end
                for i in range(end, max(start + self.config.min_chunk_size, end - 200), -1):
                    if i < len(content) and content[i] in ".!?\n":
                        chunk_end = i + 1
                        break
                chunk_text = content[start:chunk_end]
                end = chunk_end

            if chunk_text.strip():
                token_count = len(self.tokenizer.encode(chunk_text))
                word_count = len(chunk_text.split())

                section_title = find_section_for_position(sections, start) or "Unknown Section"

                chunks.append(
                    DocumentChunk(
                        content=chunk_text.strip(),
                        index=chunk_index,
                        start_char=start,
                        end_char=end,
                        metadata={
                            **base_metadata,
                            "chunk_index": chunk_index,
                            "total_chunks": -1,
                            "token_count": token_count,
                            "word_count": word_count,
                            "section_title": section_title,
                            "has_context": False,
                        },
                        token_count=token_count,
                    )
                )
                chunk_index += 1

            start = end - overlap

        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(f"Created {len(chunks)} chunks using simple fallback")
        return chunks


# ---------------------------------------------------------
# 🧾 SIMPLE CHUNKER
# ---------------------------------------------------------

class SimpleChunker:
    def __init__(self, config: ChunkingConfig):
        self.config = config

    async def chunk_document(self, content: str, title: str, source: str, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> List[DocumentChunk]:
        if not content.strip():
            return []

        base_metadata = {"title": title, "source": source, "chunk_method": "simple", **(metadata or {})}

        import re
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""
        current_pos = 0
        chunk_index = 0

        # Extract markdown sections for simple chunker as well
        sections = extract_markdown_sections(content)

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    section_title = find_section_for_position(sections, current_pos) or "Unknown Section"
                    chunks.append(self._create_chunk(current_chunk, chunk_index, current_pos, current_pos + len(current_chunk), base_metadata.copy(), section_title))
                    current_pos += len(current_chunk)
                    chunk_index += 1
                current_chunk = paragraph

        if current_chunk:
            section_title = find_section_for_position(sections, current_pos) or "Unknown Section"
            chunks.append(self._create_chunk(current_chunk, chunk_index, current_pos, current_pos + len(current_chunk), base_metadata.copy(), section_title))

        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _create_chunk(self, content: str, index: int, start_pos: int, end_pos: int, metadata: Dict[str, Any], section_title: str) -> DocumentChunk:
        metadata["section_title"] = section_title
        return DocumentChunk(content=content.strip(), index=index, start_char=start_pos, end_char=end_pos, metadata=metadata)


# ---------------------------------------------------------
# 🧰 HELPER FUNCTIONS
# ---------------------------------------------------------

def create_chunker(config: ChunkingConfig):
    if config.use_semantic_splitting:
        print("Using HybridChunker")
        return DoclingHybridChunker(config)
    else:
        print("Using SimpleChunker")
        return SimpleChunker(config)


def export_chunks_and_metadata(chunks: List[DocumentChunk], filename_prefix: str):
    txt_path = os.path.join(OUTPUT_FOLDER, f"chunks_{filename_prefix}.txt")
    json_path = os.path.join(OUTPUT_FOLDER, f"metadata_{filename_prefix}.json")

    # Save chunks
    with open(txt_path, "w", encoding="utf-8") as f_txt:
        for chunk in chunks:
            f_txt.write(" ".join(chunk.content.split()) + "\n")

    # Save metadata
    metadata_list = [chunk.metadata for chunk in chunks]
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(metadata_list, f_json, ensure_ascii=False, indent=2)

    print(f"✅ Saved chunks to {txt_path} and metadata to {json_path}")


# ---------------------------------------------------------
# 🚀 MAIN LOOP
# ---------------------------------------------------------

async def main():
    config = ChunkingConfig()
    chunker = create_chunker(config)

    for md_file in md_files:
        if not os.path.isfile(md_file):
            logger.warning(f"File not found: {md_file}")
            continue

        filename_prefix = os.path.splitext(os.path.basename(md_file))[0]
        content = open(md_file, "r", encoding="utf-8").read()
        chunks = await chunker.chunk_document(content=content, title=filename_prefix, source=md_file)
        export_chunks_and_metadata(chunks, filename_prefix)


if __name__ == "__main__":
    asyncio.run(main())
