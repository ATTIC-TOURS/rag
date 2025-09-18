"""
stores the document into vector database for search and retrieve query relevant documents

steps to process
1. retrieve all documents
1. clean documents
2. split documents
3. augment document context for each splitted document (chunks) - I/O bound
4. get embeddings
5. store embeddings to vector database - I/O bound
"""

from rag_pipeline.preprocessing.preprocessing import (
    retrieve_documents,
    clean_documents,
    set_embeddings,
    split_and_store,
    split_documents,
    store_nodes,
    remove_index
)
from rag_pipeline.retriever.retriever import retrieve_query_relevant_docs

index_name = "Requirements"
remove_index(index_name=index_name)

docs = retrieve_documents("./documents")

docs = clean_documents(docs)

set_embeddings(model_name="all-MiniLM-L6-v2")

nodes = split_documents(documents=docs, chunk_overlap_rate=0.2, max_tokens=500, add_context=True)

store_nodes(nodes=nodes, index_name=index_name)

# test query
# response = retrieve_query_relevant_docs(text="japan visa", similarity_top_k=2, index_name=index_name)

# print(response)
