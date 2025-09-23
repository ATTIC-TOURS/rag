from ..text_cleaning_strategy.docs.v2 import DocsCleaningStrategyV2
from ..embeddings.embeddings import MyEmbeddings

from ..chunking_strategy.pdf_based_custom_splitter import PdfBasedCustomSplitter

from ..context_augment.context_embedder import ContextEmbedderLLM
from ..context_augment.context_augment import ContextAugmentNodeProcessor

from sentence_transformers import SentenceTransformer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.readers.file import UnstructuredReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.schema import TextNode

import weaviate

import uuid
import os
import time
from typing import Literal
from colorama import init, Fore

init(autoreset=True)
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)


# connection settings to weaviate (vector database)
connection_config = {"port": 8080, "grpc_port": 50051, "skip_init_checks": True}


def retrieve_documents(directory: str):
    reader = UnstructuredReader()
    documents = []

    for root, _, files in os.walk(directory):
        len_files = len(files)
        for idx, file in enumerate(files):
            print(Fore.GREEN + f"{idx+1}/{len_files} retrieving: {file}")
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)

                docs = reader.load_data(file_path)  # type: ignore
                for doc in docs:
                    doc.doc_id = str(uuid.uuid4())
                    doc.extra_info = {"source_path": file}
                    documents.append(doc)

    return documents


def clean_documents(documents):
    docs_cleaning = DocsCleaningStrategyV2()

    cleaned_docs = []
    len_documents = len(documents)
    for idx, document in enumerate(documents):
        cleaned_text = docs_cleaning.clean_text(document.text)
        print(
            Fore.CYAN
            + f'{idx+1}/{len_documents} cleaning: {document.extra_info["source_path"]} ..done'
        )
        cleaned_doc = Document(
            text=cleaned_text,  # main content
            doc_id=document.doc_id,  # preserve UUID
            extra_info=document.extra_info or {},  # preserve metadata
        )
        cleaned_docs.append(cleaned_doc)
    return cleaned_docs


Provider = Literal["hf", "openai"]


def set_global_embeddings(model_name: str, provider: Provider):
    if provider == "hf":
        model = SentenceTransformer(model_name)
    elif provider == "openai":
        model = OpenAIEmbeddings(model=model_name)
    else:
        raise ValueError("Provider must be 'hf' or 'openai'")

    Settings.embed_model = MyEmbeddings(model)

    print("Global embeddings set to model: ", end="")
    print(Fore.BLUE + f"{provider}:{model_name}")


SplitterType = Literal["normal", "custom"]


def split_documents(
    documents: list[Document],
    splitter_type: SplitterType,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunk_overlap_rate: float = 0.2,
    max_tokens: int = 500,
    add_context: bool = False,
    context_augment_model_name: str = "",
) -> list[TextNode]:

    if splitter_type == "normal":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if splitter_type == "custom":
        splitter = PdfBasedCustomSplitter(
            chunk_overlap_rate=chunk_overlap_rate, max_tokens=max_tokens
        )

    context_embedder = None
    if add_context:
        context_embedder = ContextEmbedderLLM(model_name=context_augment_model_name)

    nodes: list[TextNode] = []
    len_documents = len(documents)
    for idx, doc in enumerate(documents):
        print(
            Fore.GREEN
            + f'{idx+1}/{len_documents} splitting: {doc.extra_info["source_path"]}'
        )
        chunks = splitter.split_text(doc.text)

        if add_context:
            print(
                Fore.CYAN
                + f'adding context to chunks of {doc.extra_info["source_path"]}'
            )
            postprocessor = ContextAugmentNodeProcessor(
                doc.text, context_embedder=context_embedder
            )
            nodes.extend(
                [postprocessor.process_node(TextNode(text=chunk)) for chunk in chunks]
            )
        else:
            nodes.extend([TextNode(text=chunk) for chunk in chunks])

    return nodes


def store_nodes(nodes: list[TextNode], index_name: str):
    client = None
    try:
        client = weaviate.connect_to_local(**connection_config)
        vector_store = WeaviateVectorStore(
            weaviate_client=client, index_name=index_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        print(Fore.CYAN + "storing to vector database")
        VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        print(f"✅ {len(nodes)} chunks stored")
    except Exception as e:
        print("error in store_nodes")
        print(e)
    finally:
        if client:
            client.close()


def remove_index(index_name: str):
    print(Fore.GREEN + f"resetting index: {index_name}")
    client = None
    try:
        client = weaviate.connect_to_local(**connection_config)
        client.collections.delete(index_name)
    except Exception as e:
        print("error in remove_index")
        print(e)
    finally:
        if client:
            client.close()


def remove_all_index():
    client = None
    try:
        client = weaviate.connect_to_local(**connection_config)
        client.collections.delete_all()
    except Exception as e:
        print("error in remove_all_index")
        print(e)
    finally:
        if client:
            client.close()


def list_all_index():
    client = None
    try:
        client = weaviate.connect_to_local(**connection_config)
        collections = client.collections.list_all(simple=True)
        print("Current Index List")
        return list(collections.keys())
    except Exception as e:
        print("error in remove_all_index")
        print(e)
    finally:
        if client:
            client.close()


def split_and_store(
    documents: list[Document],
    index_name: str,
    splitter_type: SplitterType,
    add_context: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunk_overlap_rate: float = 0.2,
    max_token: int = 500,
    context_augment_model_name: str = "",
):
    """
    stores the document into vector database for search and retrieve query relevant documents

    steps to process
        1. split documents (w/ context augmented)
        2. store embeddings to the vector database - I/O bound
    """

    start = time.time()  # record start time

    # # step 1 - split documents (w/ context augmented)
    nodes = split_documents(
        documents=documents,
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_overlap_rate=chunk_overlap_rate,
        max_tokens=max_token,
        add_context=add_context,
        context_augment_model_name=context_augment_model_name,
    )

    # # step 2 - store embeddings to the vector database
    store_nodes(nodes=nodes, index_name=index_name)

    end = time.time()  # record end time

    elapsed = end - start
    print(f"🕰️ Elapsed time: {elapsed:.2f} seconds for {index_name}")
