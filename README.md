
## scGREAT: Transformer-Based Deep Language Model for Gene Regulatory Network Inference from Single-Cell Transcriptomics

We present scGREAT, a framework for Gene Regulatory inference utilizing gene Embeddings And Transformer from single-cell transcriptomics. By constructing the gene expression dictionary and gene biotext dictionary, we propose to develop pre-trained language models to learn the relationships between transcriptome factors and genes. Cross-platform evidence demonstrates that scGREAT outperforms more than five state-of-the-art methods on well-known benchmark datasets across four types of gene network platforms (i.e. Cell-type-specific ChIP-seq, STRING, non-specific ChIP-seq, and LOF/GOF). Additionally, we nominated potentially missed regulatory relationships, which have been substantiated in various related studies including spatial transcriptomics.


#### Environment requirements
python 3.7.3
numpy  1.19.5
pandas 1.3.2
scikit-learn 1.0.2 
torch  1.9.1   


### All dataset links
All mentioned datasets are availabel at https://doi.org/10.5281/zenodo.3378975.


### Link to BioBERT
BioBERT Pre-trained Weights:
https://github.com/naver/biobert-pretrained 

To gain biobert embedding:
https://github.com/dmis-lab/biobert-pytorch/tree/master/embedding 


To make the data and code fully open. We have added all dataset links, uploaded pre-processing code, and added a link to BioBERT along with a complete documentation of the data pre-processing workflow, including instructions for extracting biological text feature vectors using BioBERT. Specifically, it is divided into 6 steps, as follows:

Step 1: Generate bio_name.txt file.
Use the “gen_biobert_name()” function, input “net_type” and “data_type”, to output a file containing gene names for generating biological text vectors with BioBERT.

Step 2: Generate bio_name_emb768.h5 file
Use the “get_embedding_sh()” function, input “net_type” and “data_type”, to output the file “bio_name_emb768.h5” which uses BioBERT to generate biological text vectors with 768 dimension. Use the “subprocess.run” method to execute shell scripts or use the "bash" command on the terminal command line to run shell scripts.

Step 3: Generate biovect768.npy file.	
Use the “load_embedding()” function, input “net_type” and “data_type”, to output an “.npy” file converted from the “.h5” file.

Step 4: Generate training, validation, and testing sets.
Use the “data_split()” function, input “net_type” and “data_type”, to output training, validation, and testing sets selected using the HNS method. Its “casual_flag” is used to control whether to generate multi-label labels for the causal inference task (“casual_inference(net_type, data_type)”).  

Step 5: Move all files to the datasplit folder (optional).
Use the “data_move()” function, input “net_type” and “data_type”, to move all files to the same path for convenient model invocation.

Step 6: Traing and Testing scGREAT 
#### Run the demo code
python demo.py --batch_size 32 --embed_size 768 --num_layers 2 --num_head 4 --lr 0.00001 --epochs 50 --step_size 10 --gamma 0.999 --scheduler_flag True

