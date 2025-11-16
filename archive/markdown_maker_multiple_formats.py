"""
Multi-Format Document Processing with Docling
==============================================

This script demonstrates Docling's ability to handle multiple
document formats with a unified API.

Supported formats:
- PDF (.pdf)
- Word (.docx, .doc)
- PowerPoint (.pptx, .ppt)
- Excel (.xlsx, .xls)
- HTML (.html, .htm)
- Images (.png, .jpg)
- And more...

Usage:
    python markdown_maker_multiple_formats.py
"""

from docling.document_converter import DocumentConverter
from pathlib import Path

# Directories
DOCUMENTS_DIR = Path("documents")
OUTPUT_DIR = Path("md_files")
OUTPUT_DIR.mkdir(exist_ok=True)

# List of documents to process
DOCUMENTS = [
    str(p)
    for p in DOCUMENTS_DIR.glob("*")
    if p.is_file() and not p.name.startswith(".")
]

def process_document(file_path: str, converter: DocumentConverter) -> dict:
    """Process a single document and return metadata."""
    try:
        print(f"\n📄 Processing: {Path(file_path).name}")

        # Convert document
        result = converter.convert(file_path)

        # Export to markdown
        markdown = result.document.export_to_markdown()

        # Prepare output file path
        output_file = OUTPUT_DIR / f"{Path(file_path).stem}.md"

        # Save Markdown file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)

        doc_info = {
            'file': Path(file_path).name,
            'format': Path(file_path).suffix,
            'status': 'Success',
            'markdown_length': len(markdown),
            'preview': markdown[:200].replace('\n', ' '),
            'output_file': str(output_file)
        }

        print(f"   ✓ Converted successfully")
        print(f"   ✓ Output: {output_file}")

        return doc_info

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return {
            'file': Path(file_path).name,
            'format': Path(file_path).suffix,
            'status': 'Failed',
            'error': str(e)
        }

def main():
    print("=" * 60)
    print("Multi-Format Document Processing with Docling")
    print("=" * 60)

    # Initialize converter once (reusable)
    converter = DocumentConverter()

    # Process all documents
    results = []
    for doc_path in DOCUMENTS:
        result = process_document(doc_path, converter)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)

    for result in results:
        status_icon = "✓" if result['status'] == 'Success' else "✗"
        print(f"{status_icon} {result['file']} ({result['format']})")
        if result['status'] == 'Success':
            print(f"   Length: {result['markdown_length']} chars")
            print(f"   Preview: {result['preview']}...")
            print(f"   Saved to: {result['output_file']}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
        print()

    success_count = sum(1 for r in results if r['status'] == 'Success')
    print(f"Converted {success_count}/{len(results)} documents successfully")

if __name__ == "__main__":
    main()
