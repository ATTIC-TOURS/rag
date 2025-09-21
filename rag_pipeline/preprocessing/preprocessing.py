from ..text_cleaning_strategy.docs.v2 import DocsCleaningStrategyV2
from ..embeddings.embeddings import MyEmbeddings
from ..chunking_strategy.pdf_based_custom_splitter import PdfBasedCustomSplitter
from sentence_transformers import SentenceTransformer

# from ..contextual_retrieval.context_embedder import ContextEmbedderLLM
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.readers.file import UnstructuredReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.schema import TextNode
import weaviate
import uuid
import os


# connection settings to weaviate (vector database)
connection_config = {"port": 8080, "grpc_port": 50051, "skip_init_checks": True}

def retrieve_documents(directory: str):
    reader = UnstructuredReader()
    documents = []

    for root, _, files in os.walk(directory):
        for file in files:
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
    for document in documents:
        cleaned_text = docs_cleaning.clean_text(document.text)

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
    Settings.llm = None


class ContextAugmentNodeProcessor:
    def __init__(self, whole_text: str):
        self.whole_text = whole_text

    def process_node(self, node: TextNode) -> TextNode:
        # Store the local chunk in metadata
        node.metadata["chunk_text"] = node.text
        # Augment the node text with the whole PDF context
        node.text = self.whole_text + "\n\n" + node.text
        return node

def split_documents(documents: list[Document], chunk_overlap_rate: float, max_tokens: int, add_context: bool = False) -> TextNode:
    splitter = PdfBasedCustomSplitter(chunk_overlap_rate=chunk_overlap_rate, max_tokens=max_tokens)
    nodes = []
    for doc in documents:
        chunks = splitter.split_text(doc.text)
        
        if add_context:
            postprocessor = ContextAugmentNodeProcessor(doc.text)
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

        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    except Exception as e:
        print("error in store_nodes")
        print(e)
    finally:
        if client:
            client.close()

def remove_index(index_name: str):
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

def preprocess(data_path: str, index_name: str, embeddings_model_name: str, chunk_overlap_rate: float, max_token: int, add_context: bool):
    """
    stores the document into vector database for search and retrieve query relevant documents

    steps to process
        1. retrieve all documents
        2. clean documents
        3. split documents (w/ context augmented)
        4. set embeddings as global
        5. store embeddings to the vector database - I/O bound
    """
    remove_index(index_name=index_name)

    # step 1 - retrieve all documents
    documents = retrieve_documents(data_path)

    # step 2 - clean documents
    documents = clean_documents(documents)

    # step 3 - split documents (w/ context augmented)
    nodes = split_documents(documents=documents, chunk_overlap_rate=chunk_overlap_rate, max_tokens=max_token, add_context=add_context)

    # step 4 - set embeddings as global
    set_embeddings(model_name=embeddings_model_name)

    # step 5 - store embeddings to the vector database
    store_nodes(nodes=nodes, index_name=index_name)
