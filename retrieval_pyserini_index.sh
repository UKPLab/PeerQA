GRANULARITY=$1
OUTPUT_DIR="out"
for PAPER_ID in ${OUTPUT_DIR}/pyserini/documents-bm25-${GRANULARITY}/*; do
    PAPER_ID=$(basename $PAPER_ID)
    echo "Indexing $PAPER_ID"
    python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input ${OUTPUT_DIR}/pyserini/documents-bm25-${GRANULARITY}/$PAPER_ID \
        --index ${OUTPUT_DIR}/pyserini/indexes/bm25-${GRANULARITY}/$PAPER_ID \
        --generator DefaultLuceneDocumentGenerator \
        --threads 1 \
        --storePositions --storeDocvectors --storeRaw
done
