from rag_pipeline.rag_pipeline import build_rag_pipeline

query_engine, client = build_rag_pipeline(
    index_name="Requirements",
    embeddings_model_name="intfloat/multilingual-e5-base",
    model="gemma3:1b",
    similarity_top_k=20,
    alpha=0.6,
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_top_n=5, 
)

response = query_engine.query("What are the requirements for a Japan tourist visa?")
print(response)

client.close()
