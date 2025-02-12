# PeerQA


## Setup
- GROBID 0.8
- Java 21 (for BM25 retrieval experiments with pyserini)
- python 3.10
- Install the required python packages with [uv](https://docs.astral.sh/uv/)
```bash
uv pip install .
```
- To run the pyserini experi

## Data & Preprocessing

### Questions
1. Create a new directory `data` 
2. Download the labeled questions from TBD into the `data` directory

### Papers
#### Prepare PDFs
1. Download NLPeer from https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/3618/nlpeer_v0.1.zip?sequence=5&isAllowed=n
2. Download OpenReview: ICLR 2022, ICLR 2023, NeurIPS
```bash
python download_openreview.py  
```
3. EGU: ESurf, ESD
```bash
python download_egu.py  
```

#### Extract Text from PDFs
1. Download Grobid 0.8.0 from https://github.com/kermitt2/grobid/releases/tag/0.8.0. Specifically, download the source code and run `./gradlew run` inside the `grobid-0.8.0` directory to start the server.
2. Extract the text from the PDFs
```bash
python extract_text_from_pdf.py --nlpeer_path /path/to/the/nlpeer/directory
```
## Retrieval
1. Create the qrels file for sentence-level and paragraph-level retrieval
```bash
python create_qrels.py
```
2. Run retrieval experiment  
2.1 Dense & Cross-Encoder 

| Query Model | Document Model | Similarity Function | Pooling |
|---|---|---|---|
| facebook/contriever | - | dot | mean_pooling |
| facebook/contriever-msmarco | - | dot | mean_pooling |
| facebook/dragon-plus-query-encoder | facebook/dragon-plus-context-encoder | dot | first_token |
| sentence-transformers/gtr-t5-xl | - | dot | mean_pooling |
| naver/splade-v3 | - | dot | splade |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | - | cross | - |

Run the retrieval
```bash
python run_evidence_retrieval_dense_cross.py --query_model=facebook/contriever-msmarco --sim_fn=dot --pooling=mean_pooling --granularity=sentences
```
Run the retrieval evaluation
```bash
python run_evidence_retrieval_eval.py --query_model=facebook/contriever-msmarco --sim_fn=dot --granularity=sentences
```
2.2 BM25  
-> Make sure Java 21 is installed. This is required for pyserini.
Run the data preprocessing, to convert the data to pyserini format.
```bash
python data_to_pyserini.py --granularity sentences
```
Run the indexing
```bash
bash index_pyserini_bm25.sh sentences
```
Run the retrieval
```bash
python run_evidence_retrieval_bm25.py --granularity=sentences
```
Run the retrieval evaluation
```bash
python run_evidence_retrieval_eval.py --query_model=bm25 --sim_fn=sparse --granularity=sentences
```
2.3 ColBERT
Download ColBERTv2 checkpoint from https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz
## Answerability

## Answer Generation
