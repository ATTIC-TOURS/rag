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
import copy


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

def split_and_store(
    documents, index_name: str, chunk_overlap_rate: float, max_tokens: int
):
    client = None
    try:
        client = weaviate.connect_to_local(**connection_config)
        vector_store = WeaviateVectorStore(
            weaviate_client=client, index_name=index_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        splitter = PdfBasedCustomSplitter(
            chunk_overlap_rate=chunk_overlap_rate, max_tokens=max_tokens
        )
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, text_splitter=splitter
        )
        return index
    except Exception as e:
        print("error in split_and_store")
        print(e)
    finally:
        if client:
            client.close()


def preprocess():
    """
    stores the document into vector database for search and retrieve query relevant documents

    args
        document: str - text document

    steps to process
    1. clean text
    2. split text
    3. augment document context for each splitted document (chunks) - I/O bound
    4. get embeddings
    5. store embeddings to vector database - I/O bound
    """
