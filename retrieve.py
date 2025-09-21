from rag_pipeline.retriever.retriever import retrieve_query_relevant_docs
from rag_pipeline.preprocessing.preprocessing import set_embeddings


set_embeddings('intfloat/multilingual-e5-base')

query_relevant_docs = retrieve_query_relevant_docs(
    text='requirements para sa tourist po',
    similarity_top_k=20,
    alpha=0.8,
    index_name='Requirements',
    cross_encoder_model='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
    rerank_top_k=3
)

print(len(query_relevant_docs))