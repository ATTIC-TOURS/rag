from llama_index.core import Settings

Settings.llm = None  # type: ignore
from rag.indexing.modules import (
    list_all_index,
    remove_all_index,
    retrieve_documents,
    clean_documents,
    set_global_embeddings,
    split_and_store,
)
from dotenv import load_dotenv

load_dotenv()

remove_all_index()

documents_path = "./documents"
context_augment_model_name = "gemma3:1b"
hf_embeddings_name = "intfloat/multilingual-e5-base"
openai_embeddings_name = "text-embedding-3-small"

documents = retrieve_documents(documents_path)
documents = clean_documents(documents)

# test purposes
# documents = documents[9:10]  # comment out when not testing!!!!


########################## HuggingFace ##########################

'''Create Index of the following
Normal_splitter_hf,
Normal_splitter_w_context_hf,
Custom_splitter_hf,
Custom_splitter_w_context_hf
'''

set_global_embeddings(model_name=hf_embeddings_name, provider="hf")

# Normal_splitter_hf
split_and_store(
    documents=documents,
    index_name="Normal_splitter_hf",
    splitter_type="normal",
)

# Normal_splitter_w_context_hf
split_and_store(
    documents=documents,
    index_name="Normal_splitter_w_context_hf",
    splitter_type="normal",
    # with context
    add_context=True,
    context_augment_model_name=context_augment_model_name,
)


# Custom_splitter_hf
split_and_store(
    documents=documents,
    index_name="Custom_splitter_hf",
    splitter_type="custom",
)

# Custom_splitter_w_context_hf
split_and_store(
    documents=documents,
    index_name="Custom_splitter_w_context_hf",
    splitter_type="custom",
    # with context
    add_context=True,
    context_augment_model_name=context_augment_model_name,
)


########################## OPENAI ##########################

'''Create Index of the following
Normal_splitter_openai,
Normal_splitter_w_context_openai,
Custom_splitter_openai,
Custom_splitter_w_context_openai
'''

set_global_embeddings(model_name=openai_embeddings_name, provider="openai")


# Normal_splitter_openai
split_and_store(
    documents=documents,
    index_name="Normal_splitter_openai",
    splitter_type="normal",
)

# Normal_splitter_w_context_openai
split_and_store(
    documents=documents,
    index_name="Normal_splitter_w_context_openai",
    splitter_type="normal",
    # with context
    add_context=True,
    context_augment_model_name=context_augment_model_name,
)


# Custom_splitter_openai
split_and_store(
    documents=documents,
    index_name="Custom_splitter_openai",
    splitter_type="custom",
)

# Custom_splitter_w_context_openai
split_and_store(
    documents=documents,
    index_name="Custom_splitter_w_context_openai",
    splitter_type="custom",
    # with context
    add_context=True,
    context_augment_model_name=context_augment_model_name,
)

print(list_all_index())
