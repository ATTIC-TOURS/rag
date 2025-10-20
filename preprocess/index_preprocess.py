from rag.indexing.modules import (
    list_all_index,
    set_global_embeddings,
    preprocess_index,
)

# documents_path = "./documents"
# embeddings_name = "text-embedding-3-small"
# provider = "openai"

# set_global_embeddings(model_name=embeddings_name, provider=provider)
# preprocess_index(
#     directory=documents_path,
#     index_name="JapanVisaDemo",
#     splitter_type="custom",
#     has_embed_context=True,
#     is_cloud_storage=True
# )
print(list_all_index(is_cloud_storage=True))


















# ########################## HuggingFace ##########################

# '''Create Index of the following
# Normal_splitter_hf,
# Normal_splitter_w_context_hf,
# Custom_splitter_hf,
# Custom_splitter_w_context_hf
# '''

# set_global_embeddings(model_name=hf_embeddings_name, provider="hf")

# # Normal_splitter_hf
# split_and_store(
#     documents=documents,
#     index_name="Normal_splitter_hf",
#     splitter_type="normal",
# )

# # Normal_splitter_w_context_hf
# split_and_store(
#     documents=documents,
#     index_name="Normal_splitter_w_context_hf",
#     splitter_type="normal",
#     # with context
#     add_context=True,
#     context_augment_model_name=context_augment_model_name,
# )


# # Custom_splitter_hf
# split_and_store(
#     documents=documents,
#     index_name="Custom_splitter_hf",
#     splitter_type="custom",
# )

# # Custom_splitter_w_context_hf
# split_and_store(
#     documents=documents,
#     index_name="Custom_splitter_w_context_hf",
#     splitter_type="custom",
#     # with context
#     add_context=True,
#     context_augment_model_name=context_augment_model_name,
# )


# ########################## OPENAI ##########################

# '''Create Index of the following
# Normal_splitter_openai,
# Normal_splitter_w_context_openai,
# Custom_splitter_openai,
# Custom_splitter_w_context_openai
# '''

# set_global_embeddings(model_name=openai_embeddings_name, provider="openai")


# # Normal_splitter_openai
# split_and_store(
#     documents=documents,
#     index_name="Normal_splitter_openai",
#     splitter_type="normal",
# )

# # Normal_splitter_w_context_openai
# split_and_store(
#     documents=documents,
#     index_name="Normal_splitter_w_context_openai",
#     splitter_type="normal",
#     # with context
#     add_context=True,
#     context_augment_model_name=context_augment_model_name,
# )


# # Custom_splitter_openai
# split_and_store(
#     documents=documents,
#     index_name="Custom_splitter_openai",
#     splitter_type="custom",
# )

# # Custom_splitter_w_context_openai
# split_and_store(
#     documents=documents,
#     index_name="Custom_splitter_w_context_openai",
#     splitter_type="custom",
#     # with context
#     add_context=True,
#     context_augment_model_name=context_augment_model_name,
# )

# print(list_all_index(is_cloud_storage=False))

