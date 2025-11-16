### Original YouTube Tutorial
- https://www.youtube.com/watch?v=Oe-7dGDyzPM
- https://www.youtube.com/watch?v=vFGng_3hDRk


### About Files
chunker.py - hybrid (semantic) (via docling), simple (accessed if hybrid fails) and paragraph chunker. Chunks and saves text as .txt and metadata as .json
localrag.py - largely untouched run local RAG script from original tutorial
localrag2.py - modified system prompt as compared to localrag.py
markdown_maker_multiple_formats.py - makes markdown files from multiple different file types
raw_to_chunk_chunker.py - upgraded chunker.py combining markdown_maker and chunker 
TMM_tutorial_paper.py - example pdf file used
upload.py - from original tutorial but modified to take in markdown files too. Not used in main pipeline
vault_paragraph_chunked_eg.txt - example of example md file converted chunked via paragraph chunker
vault_simply_chunked_eg.txt - example of example md file converted chunked via simple (fallback) chunker
vault.md - markdown file for further processing

### Pipeline
1. Put documents into the documents folder
2. Run raw_to_chunk_chunker.py
3. Run localrag2.py and hope that you get what you were expecting