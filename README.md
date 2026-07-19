# papers

## 2026

- "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse" 
https://arxiv.org/pdf/2603.12201 

> One-line: reuses the same indexer across every four sparse attention layers, 
reducing per-token FLOPs by 2.9× at a 1M context length. We also improve GLM-5.2’s 
MTP layer for speculative decoding, increasing the acceptance length by up to 20%