from rag_pipeline.preprocessing.preprocessing import preprocess


preprocess(
    data_path='./documents',
    index_name='Requirements',
    embeddings_model_name='intfloat/multilingual-e5-base',
    chunk_overlap_rate=0.2,
    max_token=500,
    add_context=False
)