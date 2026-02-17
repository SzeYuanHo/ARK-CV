![ARK-CV Banner](media/banner.png)

# ARK-CV: Agentic RAG for Computer Vision Architecture Recommendation

**ARK-CV** is an intelligent AI agent system that helps researchers select optimal computer vision architectures for microscopy and specialised imaging applications. By combining traditional RAG (vector search) with knowledge graph capabilities, ARK-CV analyses scientific papers, image characteristics, and domain knowledge to provide personalised architecture recommendations.

Perfect for researchers working with:
- Optical Microscopy images
- Medical imaging (histopathology, cell segmentation)
- Materials science (2D materials, nanoparticles, metallography)

## Key Features

**Intelligent Architecture Recommendations** - Get personalized CV model suggestions based on your images and research needs

**Vision Language Model Analysis** - Upload images for automatic feature extraction and architecture matching

**Knowledge Graph-Powered Search** - Queries 20+ curated research papers on CV architectures for microscopy

**Interactive Interfaces** - Choose between CLI or Gradio web UI with real-time streaming responses

**Evaluation Pipeline** - Built-in training scripts to benchmark recommended models on your datasets

**Flexible LLM Support** - Works with OpenAI, Ollama, OpenRouter, Gemini, and more (BYOK)

## Built With

- **Pydantic AI** - AI Agent Framework
- **Graphiti** - Temporal Knowledge Graph
- **PostgreSQL + pgvector** - Vector Database
- **Neo4j** - Knowledge Graph Engine
- **FastAPI** - Agent API with streaming
- **Gradio** - Web-based user interface

## Overview

ARK-CV consists of three main components:

1. **Document Ingestion Pipeline**: Processes research papers on CV architectures using semantic chunking, builds vector embeddings and knowledge graph relationships
2. **AI Agent Interface**: A conversational agent that searches across vector database and knowledge graph to recommend architectures
3. **Evaluation Pipeline**: Universal training scripts to benchmark recommended models on existing datasets (BP, Microplastics, PanNuke, TBM)
4. **Gradio Web UI**: User-friendly interface with image upload for VLM-powered architecture analysis

## For Researchers: Quick Start Guide

### What ARK-CV Does For You

ARK-CV acts as an intelligent assistant for selecting computer vision architectures. Instead of manually reviewing dozens of papers and experimenting with different models, you can:

1. **Ask questions** like "What's the best architecture for cell segmentation in fluorescence microscopy?"
2. **Upload images** of your samples and get instant architecture recommendations
3. **Get explanations** of why specific models are suitable for your task
4. **Access knowledge** from 20+ curated research papers on CV for microscopy
5. **Benchmark models** using the built-in evaluation pipeline

### Prerequisites

- Python 3.11 or higher
- PostgreSQL database (provided via Neon - see `.env.example`)
- Neo4j database (provided - see `.env.example`)
- LLM Provider API key (OpenAI, Ollama, Gemini, etc.)

## Installation

### 1. Set up a virtual environment

```bash
# Create and activate virtual environment
python -m venv venv       # python3 on Linux
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and add your LLM API key:

```bash
cp .env.example .env
```

Then edit `.env` to add your API keys:

```bash
# LLM Provider Configuration - ADD YOUR API KEY HERE
LLM_PROVIDER=openai                           # or ollama, openrouter, gemini
LLM_BASE_URL=https://api.openai.com/v1      
LLM_API_KEY=sk-your-api-key-here             # ← ADD YOUR KEY
LLM_CHOICE=gpt-4o-mini                       

# Vision Language Model (for image analysis)
VLM_BASE_URL=https://api.openai.com/v1
VLM_API_KEY=sk-your-api-key-here             # ← ADD YOUR KEY (can be same as LLM)
VLM_MODEL=gpt-4o                             # Model must support vision

# Database Configuration (Pre-configured with read-only access)
# PostgreSQL and Neo4j credentials are already set in .env.example
DATABASE_URL=postgresql://readonly_user:***@db-host.region.provider.com/arkdb
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=readonly
NEO4J_PASSWORD=***

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-your-api-key-here
EMBEDDING_MODEL=text-embedding-3-small
```

**Note**: The databases (PostgreSQL and Neo4j) are pre-configured with read-only credentials in `.env.example`. Simply copy the file and add your LLM provider API key to get started.

## Quick Start for Researchers

### 1. Start the API Server

The knowledge base is already populated with CV architecture research papers. Simply start the API:

```bash
# Activate your virtual environment first
python -m agent.api
```

The server will start on `http://localhost:8058`

### 2. Choose Your Interface

#### Option A: Gradio Web UI (Recommended for Image Analysis)

Launch the web interface with image upload support:

```bash
# In a new terminal
python gradio_ui.py
```

Open your browser to `http://localhost:7860` and:
- Upload microscopy images for automatic analysis
- Get architecture recommendations based on image features
- See which search tools the agent used (vector search, graph search, hybrid)

**Sample UI:**

<p align="center">
	<img src="media/gradio_ui_sample.png" alt="Gradio UI sample showing chat and image analysis" width="85%" />
</p>

#### Option B: Command Line Interface (CLI)

For text-only queries without images:

```bash
# In a new terminal
python cli.py
```

### 3. Example Queries

**Architecture Recommendations:**
- "What's the best architecture for cell segmentation in microscopy images?"
- "Compare U-Net, Mask R-CNN, and SAM for segmentation tasks"
- "I need to detect 2D material flakes - what model should I use?"

**With Images (Gradio UI only):**
1. Upload a microscopy image
2. Ask: "What CV architecture would you recommend for this image?"
3. The VLM analyses your image and extracts features (texture, edges, complexity)
4. The agent recommends suitable architectures with explanations

**Architecture Comparisons:**
- "Compare YOLO vs Mask R-CNN for instance segmentation"
- "When should I use Vision Transformers over CNNs?"
- "What are the trade-offs between EfficientNet and ResNet?"

### 4. Evaluate Recommended Models (Optional)

Use the built-in evaluation pipeline to benchmark models on your datasets:

```bash
cd evaluation

# Train YOLO for object detection
python trainCVmodel.py --preset bp_detection

# Train U-Net for semantic segmentation  
python trainCVmodel.py --preset tbm_semantic_seg

# Train ResNet for classification
python trainCVmodel.py --preset microplastics_classification
```

See `evaluation/README.md` for detailed instructions on dataset acquisition and training.

## How ARK-CV Works

### The Power of Hybrid RAG + Knowledge Graph for CV Research

ARK-CV combines three powerful approaches:

**1. Vector Database (PostgreSQL + pgvector)**
- Semantic search across 20+ research papers on CV architectures
- Finds papers discussing similar tasks, datasets, or challenges
- Fast retrieval of relevant architecture descriptions

**2. Knowledge Graph (Neo4j + Graphiti)**
- Relationships between architectures, tasks, and datasets
- Connections like "U-Net works well for microscopy segmentation"
- Temporal tracking of architecture evolution

**3. Vision Language Model (VLM)**
- analyses uploaded images to extract features
- Identifies texture patterns, edge characteristics, image complexity
- Provides context for architecture recommendations

**4. Intelligent Agent**
- Automatically chooses the best search strategy
- Combines results from vector search, knowledge graph, and VLM analysis
- Provides context-aware recommendations with paper citations

### Example Workflow

1. **Researcher uploads microscopy image** → VLM extracts: "fluorescence microscopy, cell structures, low contrast, overlapping objects"

2. **Agent searches knowledge base** → Finds papers on cell segmentation architectures

3. **Knowledge graph query** → Discovers U-Net and Mask R-CNN are proven for similar tasks

4. **Recommendation** → "For your fluorescence microscopy with overlapping cells, I recommend Mask R-CNN for instance segmentation. It handles overlapping objects better than U-Net's semantic approach. See [Cell Detection with Star-convex Polygons] for similar applications."

## Architecture Diagrams

Below are the conceptual architecture diagrams illustrating how ARK-CV evolved:

**Final Production Architecture**

<p align="center">
	<img src="media/final_architecture.png" alt="Final ARK-CV architecture: VLM + Agent + Vector DB + Knowledge Graph + Evaluation" width="90%" />
</p>

**Initial Prototype (Archived)**

<p align="center">
	<img src="media/initial_architecture.png" alt="Initial ARK-CV prototype architecture from archive/" width="70%" />
</p>

The initial prototype (see `archive/`) focused on early local RAG experiments before the knowledge graph and evaluation pipeline were integrated. The final architecture adds VLM-based image feature extraction, hybrid search orchestration, and a universal CV model evaluation suite.

### Why This Architecture Works for CV Research

✅ **Domain-Specific Knowledge** - Pre-loaded with microscopy and specialised CV research  
✅ **Multi-Modal Understanding** - Combines text (papers) and vision (your images)  
✅ **Relationship Discovery** - Finds connections between architectures, tasks, and datasets  
✅ **Citable Recommendations** - Every suggestion backed by research papers  
✅ **Flexible & Extensible** - Easy to add more papers or customize for your domain

## Project Structure

```
ARK-CV/
├── agent/                     # AI agent and API
│   ├── agent.py              # Main Pydantic AI agent
│   ├── api.py                # FastAPI application with streaming
│   ├── tools.py              # RAG tools (vector search, graph search, image analysis)
│   ├── providers.py          # LLM/VLM provider abstraction
│   └── prompts.py            # System prompts for agent behavior
├── ingestion/                # Document processing pipeline
│   ├── ingest.py            # Main ingestion script
│   ├── chunker.py           # Semantic chunking
│   ├── embedder.py          # Embedding generation
│   └── graph_builder.py     # Knowledge graph construction
├── evaluation/               # Model evaluation pipeline
│   ├── trainCVmodel.py      # Universal training script
│   ├── config.py            # Dataset and model configurations
│   ├── datasets/            # Dataset loaders and data
│   └── models/              # Model factory
├── archive/                  # Early iteration experiments and local tests
├── media/                    # Project assets (banner, UI sample, architecture diagrams)
├── documents/                # Research papers (20+ CV papers for microscopy)
├── sql/                      # Database schema
├── cli.py                    # Command-line interface
├── gradio_ui.py              # Web interface with image upload
└── tests/                    # Comprehensive test suite
```

## Advanced Usage

### For Developers: Adding Your Own Papers

The `documents/` folder is not included in the repository (see `.gitignore`), but this is where you should place research papers for ingestion.

To add more research papers to the knowledge base:

1. Create the `documents/` folder and add markdown files
2. Run ingestion pipeline:

```bash
python -m ingestion.ingest --clean
```

This will:
- Semantically chunk your documents
- Generate embeddings for vector search
- Extract entities and relationships for knowledge graph
- Store everything in PostgreSQL and Neo4j

**Note**: Knowledge graph construction is computationally expensive (20-30 minutes for 20 papers)

### Customizing Agent Behavior

Edit `agent/prompts.py` to control when the agent uses different search tools:
- Vector search (semantic similarity across papers)
- Graph search (relationship traversal)
- Hybrid search (combined approach)

### API Documentation

Visit `http://localhost:8058/docs` for interactive API documentation.

## Troubleshooting

**API Server Won't Start**
- Check that port 8058 is available
- Verify your LLM API key in `.env`
- Check database credentials (should work with provided read-only access)

**Gradio UI Can't Connect**
- Ensure API server is running first (`python -m agent.api`)
- Check API URL in Gradio logs
- Try accessing `http://localhost:8058/health` directly

**No Architecture Recommendations**
- Verify LLM API key is correct
- Check API usage limits with your provider
- Try with a different model (e.g., switch from gpt-4 to gpt-4o-mini)

**VLM Image Analysis Fails**
- Ensure VLM_MODEL supports vision (e.g., gpt-4o, minicpm-v for Ollama)
- Check image format (JPG, PNG supported)
- Verify VLM API key if different from LLM key

**Evaluation Pipeline Errors**
- Install all requirements: `pip install -r evaluation/requirements.txt`
- Acquire datasets following instructions in `evaluation/README.md`
- Check CUDA availability for GPU training

## Citation & Credits

### Original Creator

ARK-CV is built upon the excellent [Agentic RAG with Knowledge Graph](https://github.com/coleam00/ottomator-agents/tree/main/agentic-rag-knowledge-graph) framework created by **Cole Medin** ([@coleam00](https://github.com/coleam00)).

We extend our sincere gratitude to Cole for:
- Creating the foundational agentic RAG architecture
- Implementing the hybrid vector + knowledge graph approach
- Providing comprehensive documentation and examples
- Demonstrating best practices for Pydantic AI agents

### Research Papers

The knowledge base includes research from multiple sources. See `documents/` folder for individual paper citations. Key papers include:
- Mask R-CNN (He et al.)
- U-Net and variants
- YOLO family architectures
- Vision Transformers for microscopy
- Segment Anything Model (SAM)
- And 15+ more specialised papers

If you use ARK-CV in your research, please cite both:
1. The original Agentic RAG framework by Cole Medin
2. The specific research papers relevant to your recommendations (available in agent responses)

## Contributing

Contributions welcome! Areas of interest:
- Adding more CV research papers to the knowledge base
- Supporting additional VLM providers
- Expanding the evaluation pipeline with more datasets
- Improving architecture recommendation accuracy
- Documentation and examples

## License

This project builds upon [Cole Medin's Agentic RAG framework](https://github.com/coleam00/ottomator-agents/tree/main/agentic-rag-knowledge-graph). Please respect the original license and attribution.

---

Special thanks to Cole Medin for the foundational architecture that made this possible.
