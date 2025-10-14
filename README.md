# VectorSearch-with-HNSW

VectorSearch with HNSW is a comprehensive implementation of the Hierarchical Navigable Small World (HNSW) algorithm for efficient similarity search in high-dimensional vector spaces. This project demonstrates practical applications of approximate nearest neighbor search using the Fashion MNIST dataset, CIFAR-10 dataset as a case study.

# Fashion MNIST - Results

**Result (Query Image and Top-10 Similar Results):**

<img width="328" height="350" alt="image" src="https://github.com/user-attachments/assets/328f5c82-5a51-469d-a423-3a78723f27a9" />

<img width="1189" height="565" alt="image" src="https://github.com/user-attachments/assets/7a85617d-66f5-4a1d-9ea0-711a6535e5c1" />


# CIFAR-10 - Results

**Result (Query Image and Top-5 Similar Results):**

<img width="1189" height="245" alt="image" src="https://github.com/user-attachments/assets/941163ea-36b3-4c6b-8d0a-60c426078a86" />

# Project Overview

## Core Components

### 1. **HNSW Algorithm Implementation**
- **Multi-layer graph structure** with probabilistic level assignment
- **Thread-safe operations** supporting real-time inserts and concurrent searches
- **Diversity-aware neighbor selection** using the HNSW heuristic
- **Multiple distance metrics** including Euclidean (L2) and Cosine distance
- **Lazy deletion system** with tombstone marking and rebuild capabilities

### 2. **Key Features**
- **Efficient Construction**: Configurable parameters (M, ef_construction, mL) for balancing build time and search quality
- **High-Performance Search**: ef_search parameter controls the trade-off between accuracy and speed
- **Persistence**: Save/load functionality using NumPy and JSON formats
- **Flexible API**: CLI interface and optional FastAPI web service
- **Production Ready**: Error handling, type annotations, and comprehensive documentation

## Technical Architecture

### Data Flow
1. **Data Ingestion**: Fashion MNIST images (28×28 pixels) are flattened into 784-dimensional vectors
2. **Normalization**: Pixel values scaled to [0, 1] range for optimal distance calculations
3. **Index Building**: Vectors are inserted into the HNSW graph structure with optimized layer assignment
4. **Query Processing**: Search operations traverse the hierarchical graph for efficient nearest neighbor retrieval

### Algorithm Parameters
- **M**: Maximum number of neighbors per node (controls graph connectivity)
- **ef_construction**: Size of dynamic candidate list during construction
- **ef_search**: Size of dynamic candidate list during search
- **mL**: Level multiplier controlling layer distribution




