
## scGREAT: Transformer-Based Deep Language Model for Gene Regulatory Network Inference from Single-Cell Transcriptomics

We present scGREAT, a framework for Gene Regulatory inference utilizing gene Embeddings And Transformer from single-cell transcriptomics. By constructing the gene expression dictionary and gene biotext dictionary, we propose to develop pre-trained language models to learn the relationships between transcriptome factors and genes. Cross-platform evidence demonstrates that scGREAT outperforms more than five state-of-the-art methods on well-known benchmark datasets across four types of gene network platforms (i.e. Cell-type-specific ChIP-seq, STRING, non-specific ChIP-seq, and LOF/GOF). Additionally, we nominated potentially missed regulatory relationships, which have been substantiated in various related studies including spatial transcriptomics.


#### Environment requirements
python 3.7.3
numpy  1.19.5
pandas 1.3.2
scikit-learn 1.0.2 
torch  1.9.1   



#### Run the demo code
python demo.py --batch_size 32 --embed_size 768 --num_layers 2 --num_head 4 --lr 0.00001 --epochs 80 --step_size 10 --gamma 0.999 --scheduler_flag True


