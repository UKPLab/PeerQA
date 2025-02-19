<div align="center">
<h1>PeerQA: A Scientific Question Answering Dataset from Peer Reviews</h1>

[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
</div>

<img src="./peer-qa-overview-with-note.png" align="right" width="275" style="padding: 10px">
We present PeerQA, a real-world, scientific, document-level Question Answering (QA) dataset. PeerQA questions have been sourced from peer reviews, which contain questions that reviewers raised while thoroughly examining the scientific article. Answers have been annotated by the original authors of each paper. The dataset contains 579 QA pairs from 208 academic articles, with a majority from ML and NLP, as well as a subset of other scientific communities like Geoscience and Public Health. 
PeerQA supports three critical tasks for developing practical QA systems: Evidence retrieval, unanswerable question classification, and answer generation. 
We provide a detailed analysis of the collected dataset and conduct experiments establishing baseline systems for all three tasks. Our experiments and analyses reveal the need for decontextualization in document-level retrieval, where we find that even simple decontextualization approaches consistently improve retrieval performance across architectures. On answer generation, PeerQA serves as a challenging benchmark for long-context modeling, as the papers have an average size of 12k tokens.

# Contact
Contact person: [Tim Baumgärtner](mailto:tim.baumgaertner@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


## Getting Started

### Setup
To run the experiments, you need to install the following dependencies:
- [GROBID 0.8](https://github.com/kermitt2/grobid/releases/tag/0.8.0)
- Java 21 (for BM25 retrieval experiments with pyserini)
- [uv](https://docs.astral.sh/uv/)

To set up the environment, you can use the following commands:
```bash
# download python version with uv
uv python install 3.10
# create a virtual environment
uv venv .venv
# activate the virtual environment
source .venv/bin/activate
# install the required python packages
uv pip install .
```

## Data & Preprocessing
This section describes how to download the data from the different sources and how to preprocess it for the experiments.
### Questions
1. Create a new directory `data` and download and unzip the questions into it
```bash
mkdir data && cd data && curl -LO 'https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4467/peerqa-data-v1.0.zip?sequence=1&isAllowed=y' && unzip peerqa-data-v1.0.zip  && mv peerqa-data-v1.0/* . && rm -rf peerqa-data-v1.0 && cd ..
```

### Papers
To adhere to the licenses of the papers, we cannot provide the papers directly. Instead, we provide the steps to download the papers from the respective sources and extract the text from them.
#### Prepare PDFs
1. Download NLPeer data from https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/3618/nlpeer_v0.zip?sequence=3&isAllowed=y. Unzip it and copy the path to the `data/nlpeer` directory
2. Download PDFs from OpenReview for ICLR 2022, ICLR 2023, NeurIPS:
```bash
uv run download_openreview.py
```
3. Download the EGU PDFS for ESurf, ESD:
```bash
uv run download_egu.py
```

#### Extract Text from PDFs
1. Download [Grobid 0.8.0](https://github.com/kermitt2/grobid/releases/tag/0.8.0). Specifically, download the source code and run `./gradlew run` inside the `grobid-0.8.0` directory to start the server.
2. Extract the text from the PDFs to create `data/papers.jsonl`
```bash
uv run extract_text_from_pdf.py --nlpeer_path data/nlpeer
```
Now the data is ready for the experiments.


### Data
Once the download and preprocessing steps are completed, the following files should be present in the `data` directory:
- papers.jsonl
- qa.jsonl
- qa-augmented-answers.jsonl
- qa-unlabeled.jsonl

#### Paper Data
|Key|Type|Description|
|---|---|---|
| idx | int | The index of the paper in the dataset |
| pidx | int | The index of the paragraph in the paper |
| sidx | int | The index of the sentence in the paragraph |
| type | str | The type of the content (e.g., title, heading, caption) |
| content | str | The content of the paragraph |
| last_heading | str | The last heading before the paragraph |
| paper_id | str | The unique identifier of the paper, where the first part is the source of the paper (e.g., openreview, egu, nlpeer) and the second part is the venue (e.g. ICLR-2022-conf, ESurf, ESD), and the third part is a unique identifier for the paper |

#### QA Data
|Key|Type|Description|
|---|---|---|
| paper_id | str | The unique identifier of the paper; see above for composition |
| question_id | str | The unique identifier of the question |
| question | str | The question |
| raw_answer_evidence | List[str] | The raw evidence that has been highlighed in the PDF by the authors |
| answer_evidence_sent | List[str] | The evidence sentences that have been extracted from the raw evidence |
| answer_evidence_mapped | List[Dict[str, Union[str, List[int]]]] | The evidence sentences with the corresponding indices in the paper. If a sentence corresponds to multiple sentences in the papers.jsonl file, multiple indices will be provided here. |
| answer_free_form | str | The free-form answer provided by the authors |
| answerable | bool | Whether the question is answerable according to the authors |
| answerable_mapped | bool | Whether the question is answerable according to the authors and it has _mapped_ evidence |


## Retrieval
This section describes how to run the retrieval experiments for the PeerQA dataset. We provide the scripts for the Dense & Cross-Encoder, BM25, and ColBERT retrieval models.
### Preprocessing
1. Create the qrels file for sentence-level and paragraph-level retrieval
```bash
uv run retrieval_create_qrels.py
```
### Dense & Cross-Encoder 

The following table provides an overview of the models used for the retrieval experiments along with their respective configurations.

To reproduce the decontextualization experiments, add a `--template` argument to the scripts. In the paper we used `--template="Title: {title} Paragraph: {content}"` for paragraph chunks (i.e. `--granularity=parapgraphs`) and `--template="Title: {title} Sentence: {content}"` for sentence chunks (i.e. `--granularity=sentences`).

| Query Model | Document Model | Similarity Function | Pooling |
|---|---|---|---|
| facebook/contriever | - | dot | mean_pooling |
| facebook/contriever-msmarco | - | dot | mean_pooling |
| facebook/dragon-plus-query-encoder | facebook/dragon-plus-context-encoder | dot | first_token |
| sentence-transformers/gtr-t5-xl | - | dot | mean_pooling |
| naver/splade-v3 | - | dot | splade |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | - | cross | - |

1. Run the retrieval
```bash
uv run retrieval_dense_cross_retrieval.py --query_model=facebook/contriever-msmarco --sim_fn=dot --pooling=mean_pooling --granularity=sentences
```
2. Run the retrieval evaluation
```bash
uv run retrieval_evalulate.py --query_model=facebook/contriever-msmarco --sim_fn=dot --granularity=sentences
```
### BM25 
0. Make sure Java 21 is installed. This is required for pyserini.
1. Run the data preprocessing, to convert the data to pyserini format.
```bash
uv run retrieval_pyserini_preprocess.py --granularity=sentences
```
2. Run the indexing
```bash
bash retrieval_pyserini_index.sh sentences
```
3. Run the retrieval
```bash
uv run retrieval_pyserini_retrieval.py --granularity=sentences
```
4. Run the retrieval evaluation
```bash
uv run retrieval_evalulate.py --query_model=bm25 --sim_fn=sparse --granularity=sentences
```
### ColBERT
Download ColBERTv2 checkpoint from https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz

1. Preprocess the data, to convert it to the ColBERT format
```bash
uv run retrieval_colbert_preprocess.py --granularity=sentences
```
2. Run the indexing
```bash
uv run retrieval_colbert_index.py --granularity=sentences
```
3. Run the search
```bash
uv run retrieval_colbert_retrieval.py --granularity=sentences
```
4. Postprocess the search results
```bash
uv run retrieval_colbert_postprocess.py --granularity=sentences
```
5. Run the retrieval evaluation
```bash
uv run retrieval_evalulate.py --query_model=colbert --sim_fn=maxsim --granularity=sentences
```

## Answerability
This section describes how to run the answerability experiments for the PeerQA dataset. We provide the scripts for the answerability prediction and evaluation.

1. Run the answerability prediction
1.1 For the full-text setting, use the following arguments:
```bash
uv run generate.py --model=llama-8B-instruct --prompt_selection=answerability-full-text
```
1.2 For the RAG setting, use the following arguments:
```bash
uv run generate.py --model=llama-8B-instruct --prompt_selection=answerability-rag --context_setting=10
```
1.3 For the gold setting, use the following arguments:
```bash
uv run generate.py --model=llama-8B-instruct --prompt_selection=answerability-rag --context_setting=gold 
```
2. Run the answerability evaluation, and set the `--generation_file` argument to the path of the generated answers from the previous step (here we use the rag setup with gold paragraphs as an example).
```bash
uv run generations_evaluate_answerability.py --generation_file=out/generations-llama-8B-instruct-8k-answerability-rag-gold.jsonl
```

To run the answerability task with OpenAI models, use `generate_openai.py` instead of `generate.py`.


## Answer Generation
This section describes how to run the answer generation experiments for the PeerQA dataset. We provide the scripts for the answer generation and evaluation.

1. Download AlignScore Model and NLTK for evaluation
```bash
curl -L https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt?download=true -o AlignScore-large.ckpt
```
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

2. Run the answer generation
2.1 For the full-text setting, use the following arguments:
```bash
uv run generate.py --model=llama-8B-instruct --prompt_selection=full-text
```
2.2 For the RAG setting, use the following arguments:
```bash
uv run generate.py --model=llama-8B-instruct --prompt_selection=rag --context_setting=10
```
2.3 For the gold setting, use the following arguments:
```bash
uv run generate.py --model=llama-8B-instruct --prompt_selection=rag --context_setting=gold
```

3. Run Rouge and AlignScore evaluation and set the `--generation_file` argument to the path of the generated answers from the previous step (here we use the full-text setup as an example).
```bash
uv run generations_evaluate_rouge_alignscore.py --generation_file=out/generations-llama-8B-instruct-8k-full-text.jsonl
```
4. Run Prometheus evaluation
```bash
uv run generations_evaluate_prometheus.py --generation_file=out/generations-llama-8B-instruct-8k-full-text.jsonl
```

To run the answer generation task with OpenAI models, use `generate_openai.py` instead of `generate.py`.

## Cite

Please use the following citation:

```
tbd
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
