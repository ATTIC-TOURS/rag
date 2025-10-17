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
from weaviate.classes.init import Auth

import uuid
import os
from dotenv import load_dotenv

load_dotenv()
from typing import Literal
from colorama import init, Fore

init(autoreset=True)
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)

from ..my_decorators.time import time_performance

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

# connection settings to weaviate (vector database)
connection_config = {
    "local": {"port": 8080, "grpc_port": 50051, "skip_init_checks": True},
    "cloud": {
        "cluster_url": weaviate_url,
        "auth_credentials": Auth.api_key(weaviate_api_key),
    },
}


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

@time_performance("set_global_embeddings")
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


def ingest_documents(
    documents: list[Document],
    splitter_type: SplitterType,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunk_overlap_rate: float = 0.2,
    max_tokens: int = 500,
    has_embed_context: bool = False,
    context_augment_model_name: str = "gpt-5-mini",
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
    if has_embed_context:
        context_embedder = ContextEmbedderLLM(model_name=context_augment_model_name)

    nodes: list[TextNode] = []
    len_documents = len(documents)
    for idx, doc in enumerate(documents):
        print(
            Fore.GREEN
            + f'{idx+1}/{len_documents} splitting: {doc.extra_info["source_path"]}'
        )
        chunks = splitter.split_text(doc.text)

        if has_embed_context:
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


def index(nodes: list[TextNode], index_name: str, is_cloud_storage=False):
    client = None
    try:
        if is_cloud_storage:
            client = weaviate.connect_to_weaviate_cloud(**connection_config["cloud"])
        else:
            client = weaviate.connect_to_local(**connection_config["local"])
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


def remove_index(index_name: str, is_cloud_storage: bool = False):
    print(Fore.GREEN + f"resetting index: {index_name}")
    client = None
    try:
        if is_cloud_storage:
            client = weaviate.connect_to_weaviate_cloud(**connection_config["cloud"])
        else:
            client = weaviate.connect_to_local(**connection_config["local"])
        client.collections.delete(index_name)
    except Exception as e:
        print("error in remove_index")
        print(e)
    finally:
        if client:
            client.close()


@time_performance(name="remove_all_index")
def remove_all_index(is_cloud_storage: bool = False):
    client = None
    try:
        if is_cloud_storage:
            client = weaviate.connect_to_weaviate_cloud(**connection_config["cloud"])
        else:
            client = weaviate.connect_to_local(**connection_config["local"])
        client.collections.delete_all()
    except Exception as e:
        print("error in remove_all_index")
        print(e)
    finally:
        if client:
            client.close()


@time_performance("list_all_index")
def list_all_index(is_cloud_storage: bool = False):
    client = None
    try:
        if is_cloud_storage:
            client = weaviate.connect_to_weaviate_cloud(**connection_config["cloud"])
        else:
            client = weaviate.connect_to_local(**connection_config["local"])
        collections = client.collections.list_all(simple=True)
        return list(collections.keys())
    except Exception as e:
        print("error in remove_all_index")
        print(e)
    finally:
        if client:
            client.close()


@time_performance("preprocess_index")
def preprocess_index(
    directory: str,
    index_name: str,
    splitter_type: SplitterType,
    has_embed_context: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunk_overlap_rate: float = 0.2,
    max_token: int = 500,
    context_augment_model_name: str = "gpt-5-mini",
    is_cloud_storage: bool = False,
):
    """
    stores the document into vector database for search and retrieve query relevant documents

    steps to process
        1. split documents (w/ context augmented)
        2. store embeddings to the vector database - I/O bound
    """
    
    documents = retrieve_documents(directory=directory)
    
    documents = clean_documents(documents)

    # data ingestion
    nodes = ingest_documents(
        documents=documents,
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_overlap_rate=chunk_overlap_rate,
        max_tokens=max_token,
        has_embed_context=has_embed_context,
        context_augment_model_name=context_augment_model_name,
    )

    # data indexing
    index(nodes=nodes, index_name=index_name, is_cloud_storage=is_cloud_storage)
