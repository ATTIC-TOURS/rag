from llama_index.core import Settings

Settings.llm = None  # type: ignore
from rag.retriever.retriever import retrieve_query_relevant_docs
from rag.indexing.modules import set_global_embeddings
from dotenv import load_dotenv

load_dotenv()

hf_embeddings_name = "intfloat/multilingual-e5-base"
openai_embeddings_name = "text-embedding-3-small"

set_global_embeddings(model_name=hf_embeddings_name, provider="hf")
query_relevant_docs = retrieve_query_relevant_docs(
    text="requirements para sa transit po",
    alpha=0.8,
    similarity_top_k=1,
    index_name="Normal_splitter_hf",
)
print('hf')
print(query_relevant_docs)

# set_global_embeddings(model_name=openai_embeddings_name, provider="openai")
# query_relevant_docs = retrieve_query_relevant_docs(
#     text="requirements para sa transit po",
#     alpha=0.8,
#     similarity_top_k=1,
#     index_name="Normal_splitter_openai",
# )
# print('openai')
# print(query_relevant_docs)


