"""
Knowledge graph builder for extracting entities and relationships.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone
import asyncio
import re

from graphiti_core import Graphiti
from dotenv import load_dotenv

from .chunker import DocumentChunk

# Import graph utilities
try:
    from ..agent.graph_utils import GraphitiClient
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.graph_utils import GraphitiClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds knowledge graph from document chunks using Graphiti."""
    
    def __init__(self):
        """Initialize graph builder."""
        self.graph_client = GraphitiClient()
        self._initialized = False
    
    async def initialize(self):
        """Initialize graph client."""
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True
    
    async def close(self):
        """Close graph client."""
        if self._initialized:
            await self.graph_client.close()
            self._initialized = False
    
    async def add_document_to_graph(
        self,
        chunks: List[DocumentChunk],
        document_title: str,
        document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 3
    ) -> Dict[str, Any]:
        """
        Add document chunks to knowledge graph.
        Graphiti's LLM will extract entities/relationships automatically.
        
        Args:
            chunks: List of document chunks
            document_title: Title of the document
            document_source: Source of the document
            document_metadata: Additional metadata (authors, year, etc.)
            batch_size: Number of chunks to process in each batch
        
        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks:
            return {"episodes_created": 0, "errors": []}
        
        logger.info(f"Adding {len(chunks)} chunks to knowledge graph for document: {document_title}")
        
        episodes_created = 0
        errors = []
        
        # Process chunks one by one to let Graphiti extract entities
        for i, chunk in enumerate(chunks):
            try:
                episode_id = f"{document_source}_{chunk.index}_{datetime.now().timestamp()}"
                
                # Prepare content with CV-specific context hints
                episode_content = self._prepare_cv_episode_content(
                    chunk,
                    document_title,
                    document_metadata
                )
                
                source_description = f"Paper: {document_title} (Section {chunk.index})"
                
                # Add episode - Graphiti's LLM will extract:
                # - Architectures (U-Net, ResNet, etc.)
                # - Datasets (ImageNet, COCO, etc.)
                # - Applications (segmentation, detection, etc.)
                # - Performance metrics
                await self.graph_client.add_episode(
                    episode_id=episode_id,
                    content=episode_content,
                    source=source_description,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "document_title": document_title,
                        "document_source": document_source,
                        "chunk_index": chunk.index,
                        "authors": document_metadata.get("authors", []) if document_metadata else [],
                        "year": document_metadata.get("year") if document_metadata else None,
                        "domain": "Computer Vision"
                    }
                )
                
                episodes_created += 1
                logger.info(f"✓ Episode {episode_id} added ({episodes_created}/{len(chunks)})")
                
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)  # Rate limiting
                    
            except Exception as e:
                error_msg = f"Failed to add chunk {chunk.index}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue
        
        result = {
            "episodes_created": episodes_created,
            "total_chunks": len(chunks),
            "errors": errors
        }
        
        logger.info(f"Graph building complete: {episodes_created} episodes, {len(errors)} errors")
        return result
    
    def _prepare_cv_episode_content(
        self,
        chunk: DocumentChunk,
        document_title: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Prepare episode content with CV research paper context.
        This guides Graphiti's LLM to extract relevant entities.
        
        Args:
            chunk: Document chunk
            document_title: Paper title
            document_metadata: Authors, year, etc.
        
        Returns:
            Formatted content for Graphiti processing
        """
        max_content_length = 6000
        
        content = chunk.content
        if len(content) > max_content_length:
            truncated = content[:max_content_length]
            last_sentence = max(
                truncated.rfind('. '),
                truncated.rfind('! '),
                truncated.rfind('? ')
            )
            
            if last_sentence > max_content_length * 0.7:
                content = truncated[:last_sentence + 1] + " [TRUNCATED]"
            else:
                content = truncated + "... [TRUNCATED]"
            
            logger.warning(f"Truncated chunk {chunk.index} from {len(chunk.content)} to {len(content)} chars")
        
        # Add structured context to guide Graphiti's entity extraction
        context_parts = [f"Computer Vision Research Paper: {document_title}"]
        
        if document_metadata:
            if document_metadata.get("authors"):
                authors = document_metadata["authors"][:3]  # First 3 authors
                context_parts.append(f"Authors: {', '.join(authors)}")
            
            if document_metadata.get("year"):
                context_parts.append(f"Year: {document_metadata['year']}")
        
        context_header = " | ".join(context_parts)
        
        # Format: Header + Content
        # Graphiti's LLM will extract entities like:
        # - Architecture names (U-Net, ResNet)
        # - Dataset characteristics
        # - Tasks (segmentation, detection)
        # - Metrics (IoU, mAP)
        episode_content = f"{context_header}\n\n{content}"
        
        return episode_content
    
    async def extract_entities_from_chunks(
        self,
        chunks: List[DocumentChunk],
        extract_architectures: bool = True,
        extract_datasets: bool = True,
        extract_applications: bool = True,
        extract_metrics: bool = True
    ) -> List[DocumentChunk]:
        """
        Add lightweight entity hints to chunk metadata.
        This is SUPPLEMENTARY to Graphiti's LLM-based extraction.
        
        Use case: Quick filtering before sending to Graphiti.
        
        Args:
            chunks: Document chunks
            extract_architectures: Extract CV architecture names
            extract_datasets: Extract dataset characteristics
            extract_applications: Extract application domains
            extract_metrics: Extract performance metrics
        
        Returns:
            Chunks with entity metadata hints
        """
        logger.info(f"Adding entity hints to {len(chunks)} chunks")
        
        enriched_chunks = []
        
        for chunk in chunks:
            entity_hints = {}
            content = chunk.content
            
            # Lightweight regex hints (not exhaustive - Graphiti will do full extraction)
            if extract_architectures:
                entity_hints["architecture_mentions"] = self._find_architecture_mentions(content)
            
            if extract_datasets:
                entity_hints["dataset_mentions"] = self._find_dataset_mentions(content)
            
            if extract_applications:
                entity_hints["application_mentions"] = self._find_application_mentions(content)
            
            if extract_metrics:
                entity_hints["metric_mentions"] = self._find_metric_mentions(content)
            
            enriched_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "entity_hints": entity_hints,  # Hints, not ground truth
                    "hint_extraction_date": datetime.now().isoformat()
                },
                token_count=chunk.token_count
            )
            
            if hasattr(chunk, 'embedding'):
                enriched_chunk.embedding = chunk.embedding
            
            enriched_chunks.append(enriched_chunk)
        
        logger.info("Entity hint extraction complete")
        return enriched_chunks
    
    def _find_architecture_mentions(self, text: str) -> List[str]:
        """Find CV architecture mentions (lightweight pattern matching)."""
        # Common CV architectures
        patterns = [
                # ---------------- U-Net family ----------------
                r"\b(?:U[-\s]?Net\+\+|UNet\+\+)\b",
                r"\b(?:U[-\s]?Net|UNet)\b",
                r"\b(?:nnU[-\s]?Net|nnUNet)\b",

                # ---------------- Residual / CNN backbones ----------------
                r"\b(?:ResNet|ResNeXt|WideResNet|Res2Net)(?:-\d+)?\b",
                r"\b(?:VGG|Inception|Inception-ResNet|Xception)(?:-\d+)?\b",
                r"\b(?:EfficientNetV2?|MobileNetV2?|DenseNet\d*|NASNet|RegNet|SENet|HRNet|ConvNeXt|GhostNet|ShuffleNetV2?)\b",

                # ---------------- Transformers for vision ----------------
                r"\b(?:ViT|Vision Transformer)\b",
                r"\bDeiT\b",
                r"\bSwin\s+Transformer\b",
                r"\bPVT\b",
                r"\bCvT\b",
                r"\bCoAtNet\b",
                r"\bMaxViT\b",
                r"\b(?:Transformer|Transformers)\b",  # keep generic transformer, but not plain "Attention"

                # ---------------- Object detection / instance seg ----------------
                r"\bYOLOv?\d+(?:[a-z])?\b",       # YOLOv3, YOLOv5x, etc.
                r"\bYOLO\b",
                r"\b(?:Faster|Mask|Cascade)\s+R-CNN\b",
                r"\bSSD\b",
                r"\bRetinaNet\b",
                r"\bEfficientDet\b",
                r"\bDETR\b",
                r"\bDeformable\s+DETR\b",
                r"\bFCOS\b",
                r"\bCenterNet\b",

                # ---------------- Segmentation models ----------------
                r"\bDeepLabv?\d\+?\b",            # DeepLabv2, DeepLabv3, DeepLabv3+
                r"\bPSPNet\b",
                r"\bSegFormer\b",
                r"\bFCN\b",
                r"\bFPN\b",
                r"\bPANet\b",
                r"\bMask2Former\b",

                # ---------------- Generative / latent models ----------------
                r"\bGANs?\b",
                r"\bGenerative Adversarial Network\b",
                r"\bDCGAN\b",
                r"\bStyleGAN(?:2|3)?\b",
                r"\bBigGAN\b",
                r"\bCycleGAN\b",
                r"\bPix2Pix(?:HD)?\b",
                r"\bVAE\b",
                r"\bVQ-VAE(?:-2)?\b",
                r"\bVQGAN\b",
                r"\bDiffusion\s+Model\b",
                r"\bDDPM\b",
                r"\bUNet-based\s+Diffusion\b",

                # ---------------- Other named architectures ----------------
                r"\bCapsNet\b",
                r"\bCapsule\s+Network\b",

                # ---------------- Generic CNN mention ----------------
                r"\bCNNs?\b",
                r"\bConvolutional Neural Network\b",
        ]
        
        mentions = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            mentions.update(matches)
        
        return list(mentions)
    
    def _find_dataset_mentions(self, text: str) -> List[str]:
        """Find dataset characteristic mentions."""
        patterns = [
            # Dataset size / scale
            r"\b(?:tiny|very small|small|limited)\s+dataset\b",
            r"\b(?:medium[-\s]sized|moderate[-\s]sized)\s+dataset\b",
            r"\b(?:large|large[-\s]scale|substantial)\s+dataset\b",
            r"\bvery large[-\s]scale\s+dataset\b",
            r"\b(\d[\d,]*)\s+(?:images?|samples?)\b",
            r"\bdataset size of\s+(\d[\d,]*)\b",
            # Label availability / type
            r"\b(?:labeled|labelled)\s+data\b",
            r"\b(?:unlabeled|unlabelled)\s+data\b",
            r"\bpartially\s+(?:labeled|labelled)\b",
            r"\bweakly\s+(?:labeled|supervised)\b",
            r"\bnoisy\s+labels?\b",
            r"\bclean\s+labels?\b",
            r"\bimage[-\s]level\s+labels?\b",
            r"\bbounding[-\s]box(?:\s+annotations|\s+labels)?\b",
            r"\bsegmentation\s+masks?\b",
            r"\b(?:keypoint|landmark)\s+(?:annotations|labels)\b",
            # Class balance
            r"\bbalanced\s+dataset\b",
            r"\bclass[-\s]balanced\s+dataset\b",
            r"\b(?:imbalanced|unbalanced|skewed|long[-\s]tailed)\s+dataset\b",
            r"\b(?:severe|strong|extreme)\s+class\s+imbalance\b",
            r"\brare\s+positive\s+class\b",
            # Data source / domain
            r"\bsynthetic\s+(?:data|images?)\b",
            r"\bsimulated\s+(?:data|images?)\b",
            r"\bin[-\s]?silico\s+(?:data|experiments?)\b",
            r"\breal[-\s]?world\s+(?:data|images?)\b",
            r"\bclinical\s+(?:data|images?)\b",
            # microscopy-ish dataset descriptors
            r"\bsimulated\s+microscopy\b",
            r"\bin\s+vitro\s+(?:imaging|microscopy)\b",
            r"\bin\s+vivo\s+(?:imaging|microscopy)\b",
            # Data modality / structure
            r"\bsingle[-\s]image\s+dataset\b",
            r"\bimage\s+sequence[s]?\b",
            r"\bvideo\s+frames?\b",
            r"\btime[-\s]lapse(?:\s+microscopy)?\b",
            r"\btime[-\s]series\s+(?:data|images?)\b",
            r"\b3d\s+(?:volumes?|volumetric\s+images?|microscopy)\b",
            r"\bz[-\s]?stack[s]?\b",
            r"\bmulti[-\s]?channel\s+(?:images?|microscopy)\b",
            r"\bmultispectral\s+(?:images?|data)\b",
            r"\bimages?\s+and\s+metadata\b",
            r"\bimages?\s+and\s+text\b",
            r"\bimage[-\s]text\s+pairs?\b",
            # Resolution / scale
            r"\bhigh[-\s]resolution\s+images?\b",
            r"\bhigh[-\s]res\s+images?\b",
            r"\blow[-\s]resolution\s+images?\b",
            r"\blow[-\s]res\s+images?\b",
            r"\bheterogeneous\s+resolution\b",
            r"\bmixed\s+resolution\b",
            # Noise / artifacts
            r"\bnoisy\s+(?:images?|data)\b",
            r"\blow[-\s]noise\s+(?:images?|data)\b",
            r"\bvarying\s+noise\s+levels\b",
            r"\bheterogeneous\s+noise\b",
            # Image quality / contrast / illumination
            r"\bhigh\s+image\s+quality\b",
            r"\bmedium\s+image\s+quality\b",
            r"\blow\s+image\s+quality\b",
            r"\bpoor\s+image\s+quality\b",
            r"\bhigh\s+contrast\s+images?\b",
            r"\bmedium\s+contrast\s+images?\b",
            r"\blow\s+contrast\s+images?\b",
            r"\bpoor\s+contrast\b",
            r"\buniform\s+illumination\b",
            r"\beven\s+illumination\b",
            r"\bnon[-\s]uniform\s+illumination\b",
            r"\buneven\s+illumination\b",
            r"\billumination\s+bias\b",
            r"\bvignetting\b",
            r"\bvignetted\s+illumination\b",
            r"\bnon[-\s]uniform\s+background\b",
            r"\buneven\s+background\b",
            # Image type / modality
            r"\boptical\s+microscopy\b",
            r"\blight\s+microscopy\b",
            r"\b2d\s+(?:microscop\w*|images?)\b",
            r"\b3d\s+(?:microscop\w*|images?|volumes?)\b",
            r"\btime[-\s]lapse\s+microscopy\b",
            r"\bbrightfield\s+microscop\w*\b",
            r"\bphase[-\s]contrast\s+microscop\w*\b",
            r"\bfluorescen\w*\s+microscop\w*\b",
            r"\bconfocal\s+microscop\w*\b",
            r"\bphotographic\s+images?\b",
            r"\bradiology\s+images?\b",
            r"\bct\s+scans?\b",
            r"\bmri\s+scans?\b",
            r"\bx[-\s]?ray\s+images?\b",
            # Protocol / evaluation
            r"\btrain[-\s]?test\s+split\b",
            r"\btraining\s+and\s+test\s+split\b",
            r"\btrain\s*/\s*test\s+split\b",
            r"\bcross[-\s]?validation\b",
            r"\bk[-\s]?fold\s+cross[-\s]?validation\b",
            r"\bdata\s+augmentation\b",
            r"\bno\s+data\s+augmentation\b",
        ]
        
        mentions = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            mentions.update(matches)
        
        return list(mentions)
    
    def _find_application_mentions(self, text: str) -> List[str]:
        """Find CV task mentions."""
        patterns = [
            r"\b(?:image|semantic|instance|panoptic)\s+segmentation\b",
            r"\b(?:object|cell|anomaly)\s+detection\b",
            r"\b(?:object|cell|instance)\s+tracking\b",
            r"\b(?:image|visual)\s+(?:classification|recognition)\b",
            r"\b(?:pose|keypoint)\s+estimation\b",
            r"\b(?:depth|monocular\s+depth)\s+estimation\b",
            r"\boptical\s+flow\s+estimation\b",
            r"\b(?:object|cell)\s+counting\b",
            r"\bdensity\s+estimation\b",
            r"\bimage\s+denoising\b",
            r"\bimage\s+super[-\s]?resolution\b",
            r"\bimage\s+reconstruction\b",
            r"\bimage\s+generation\b",
            r"\bimage\s+synthesis\b",
        ]
        
        mentions = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            mentions.update(matches)
        
        return list(mentions)
    
    def _find_metric_mentions(self, text: str) -> List[str]:
        """Find performance metric mentions (including simple metric=value hints)."""
        patterns = [
            # Core metric names (segmentation / detection / classification / quality)
            r"\b(?:IoU|Dice|mIoU|Jaccard)\b",
            r"\b(?:mAP|AP)\b",
            r"\baccuracy\b",
            r"\btop-(?:1|5)\s+accuracy\b",
            r"\bF1\b",
            r"\bprecision\b",
            r"\brecall\b",
            r"\bPSNR\b",
            r"\bSSIM\b",
            r"\bMAE\b",
            r"\bMSE\b",
            r"\bRMSE\b",
            r"\bR-squared\b",

            # Metric + value patterns, e.g. "mAP = 0.75", "IoU of 82%", "F1: 0.91"
            r"\b(IoU|Dice|mIoU|Jaccard|mAP|AP|accuracy|F1|precision|recall|PSNR|SSIM|MAE|MSE|RMSE|R-squared)"
            r"\s*(?:=|of|:)\s*([0-9]*\.?[0-9]+%?)",
        ]

        mentions = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for m in matches:
                if isinstance(m, tuple):
                    metric, value = m
                    mentions.add(f"{metric.strip()} {value.strip()}")
                else:
                    mentions.add(m.strip())

        return list(mentions)
    
    async def clear_graph(self):
        """Clear all data from knowledge graph."""
        if not self._initialized:
            await self.initialize()
        
        logger.warning("Clearing knowledge graph...")
        await self.graph_client.clear_graph()
        logger.info("Knowledge graph cleared")


# Factory function
def create_graph_builder() -> GraphBuilder:
    """Create graph builder instance."""
    return GraphBuilder()


# Example usage
async def main():
    """Example: Process a CV research paper."""
    from .chunker import ChunkingConfig, create_chunker
    
    config = ChunkingConfig(chunk_size=500, use_semantic_splitting=True)
    chunker = create_chunker(config)
    graph_builder = create_graph_builder()
    
    sample_paper = """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    
    Abstract:
    We present U-Net, a network architecture for biomedical image segmentation.
    The architecture consists of a contracting path to capture context and a symmetric
    expanding path for precise localization. We trained the network on the ISBI cell
    tracking challenge dataset and achieved an IoU of 0.92.
    
    Methods:
    Our architecture uses 3x3 convolutions followed by ReLU activation and 2x2 max pooling
    for downsampling. The dataset consisted of 30 microscopy images (512x512 pixels).
    We applied data augmentation including random rotations and elastic deformations.
    
    Results:
    On the ISBI challenge, U-Net outperformed previous methods with a segmentation
    accuracy of 92% IoU. The model generalizes well to other medical imaging tasks
    including brain tumor segmentation (BraTS dataset) and retinal vessel segmentation.
    """
    
    # Chunk document
    chunks = chunker.chunk_document(
        content=sample_paper,
        title="U-Net: Convolutional Networks for Biomedical Image Segmentation",
        source="ronneberger2015unet.pdf"
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Add entity hints (optional)
    enriched_chunks = await graph_builder.extract_entities_from_chunks(chunks)
    
    for i, chunk in enumerate(enriched_chunks):
        hints = chunk.metadata.get('entity_hints', {})
        print(f"\nChunk {i} hints:")
        for key, values in hints.items():
            if values:
                print(f"  {key}: {values}")
    
    # Add to knowledge graph (Graphiti will do full LLM-based extraction)
    try:
        result = await graph_builder.add_document_to_graph(
            chunks=enriched_chunks,
            document_title="U-Net: Convolutional Networks for Biomedical Image Segmentation",
            document_source="ronneberger2015unet.pdf",
            document_metadata={
                "authors": ["Olaf Ronneberger", "Philipp Fischer", "Thomas Brox"],
                "year": 2015,
                "venue": "MICCAI"
            }
        )
        
        print(f"\nGraph building result: {result}")
        
    except Exception as e:
        print(f"Graph building failed: {e}")
    
    finally:
        await graph_builder.close()


if __name__ == "__main__":
    asyncio.run(main())