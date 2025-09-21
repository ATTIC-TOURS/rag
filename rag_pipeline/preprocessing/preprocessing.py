from ..text_cleaning_strategy.docs.v2 import DocsCleaningStrategyV2
from ..embeddings.embeddings import MyEmbeddings
from ..chunking_strategy.pdf_based_custom_splitter import PdfBasedCustomSplitter
from sentence_transformers import SentenceTransformer
from ..context_augment.context_augment import ContextAugmentNodeProcessor
# from ..contextual_retrieval.context_embedder import ContextEmbedderLLM
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.readers.file import UnstructuredReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.schema import TextNode
import weaviate
import uuid
import os
import time
from colorama import init, Fore
init(autoreset=True)

# connection settings to weaviate (vector database)
connection_config = {"port": 8080, "grpc_port": 50051, "skip_init_checks": True}

def retrieve_documents(directory: str):
    reader = UnstructuredReader()
    documents = []

    for root, _, files in os.walk(directory):
        len_files = len(files)
        for idx, file in enumerate(files):
            print(Fore.GREEN + f'{idx+1}/{len_files} retrieving: {file}')
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)

                docs = reader.load_data(file_path)
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
        print(Fore.CYAN + f'{idx+1}/{len_documents} cleaning: {document.extra_info["source_path"]} ..done')
        cleaned_doc = Document(
            text=cleaned_text,  # main content
            doc_id=document.doc_id,  # preserve UUID
            extra_info=document.extra_info or {},  # preserve metadata
        )
        cleaned_docs.append(cleaned_doc)
    return cleaned_docs


def set_embeddings(model_name: str):
    hf_model = SentenceTransformer(model_name)
    embed_model = MyEmbeddings(hf_model)
    Settings.embed_model = embed_model
    print('Global embeddings set to model: ', end='')
    print(Fore.BLUE + f'{model_name}')


def split_documents(
    documents: list[Document], 
    chunk_overlap_rate: float, 
    max_tokens: int, 
    add_context: bool = False, 
    context_augment_model_name: str | None = None
) -> TextNode:
    splitter = PdfBasedCustomSplitter(chunk_overlap_rate=chunk_overlap_rate, max_tokens=max_tokens)
    
    nodes = []
    len_documents = len(documents)
    for idx, doc in enumerate(documents):
        print(Fore.GREEN + f'{idx+1}/{len_documents} splitting: {doc.extra_info["source_path"]}')
        chunks = splitter.split_text(doc.text)
        
        if add_context:
            print(Fore.CYAN + f'adding context to chunks of {doc.extra_info["source_path"]}')
            postprocessor = ContextAugmentNodeProcessor(doc.text, model_name=context_augment_model_name)
            nodes.extend([postprocessor.process_node(TextNode(text=chunk)) for chunk in chunks])
        else:
            nodes.extend([TextNode(text=chunk) for chunk in chunks])
    return nodes

def store_nodes(nodes: list[TextNode], index_name: str):
    client = None
    try:
        client = weaviate.connect_to_local(**connection_config)
        vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        print(Fore.CYAN + 'storing to vector database')
        VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        print(f'✅ {len(nodes)} chunks stored')
    except Exception as e:
        print("error in store_nodes")
        print(e)
    finally:
        if client:
            client.close()

def remove_index(index_name: str):
    print(Fore.GREEN + f'resetting index: {index_name}')
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

def preprocess(
    data_path: str, 
    index_name: str, 
    embeddings_model_name: str, 
    chunk_overlap_rate: float, 
    max_token: int, 
    add_context: bool, 
    context_augment_model_name: str | None = None
):
    """
    stores the document into vector database for search and retrieve query relevant documents

    steps to process
        1. retrieve all documents
        2. clean documents
        3. split documents (w/ context augmented)
        4. store embeddings to the vector database - I/O bound
    """
    

    start = time.time()  # record start time

    remove_index(index_name=index_name)
    Settings.llm = None
    set_embeddings(model_name=embeddings_model_name)

    # step 1 - retrieve all documents
    documents = retrieve_documents(data_path)

    # step 2 - clean documents
    documents = clean_documents(documents)

    # # step 3 - split documents (w/ context augmented)
    nodes = split_documents(
        documents=documents, 
        chunk_overlap_rate=chunk_overlap_rate, 
        max_tokens=max_token, 
        add_context=add_context, 
        context_augment_model_name=context_augment_model_name
    )

    # # step 4 - store embeddings to the vector database
    store_nodes(nodes=nodes, index_name=index_name)
    
    end = time.time()  # record end time
    
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.2f} seconds")
