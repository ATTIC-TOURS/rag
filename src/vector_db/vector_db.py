import weaviate
import weaviate.classes.config as wc
from sentence_transformers import SentenceTransformer
from colorama import Fore, init

init(autoreset=True)


class MyWeaviateDB:

    connection_config = {"port": 8080, "grpc_port": 50051, "skip_init_checks": True}

    def __init__(
        self,
        embeddings: SentenceTransformer,
        ef_construction: int,
        bm25_b: float,
        bm25_k1: float,
        collection_name: str = "Requirements",
    ):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.ef_construction = ef_construction
        self.bm25_b = bm25_b
        self.bm25_k1 = bm25_k1

    def setup_collection(self):
        try:
            client = weaviate.connect_to_local(**self.connection_config)

            # client.collections.delete_all()
            client.collections.delete(self.collection_name)
            print(Fore.RED + f"{self.collection_name} has been removed!")

            client.collections.create(
                # NAME
                name=self.collection_name,
                # PROPERTIES
                properties=[
                    wc.Property(
                        name="file_name",
                        data_type=wc.DataType.TEXT,
                        index_searchable=True,
                    ),
                    wc.Property(
                        name="title",
                        data_type=wc.DataType.TEXT,
                        index_searchable=True,
                    ),  # used for filtering
                    wc.Property(
                        name="content",
                        data_type=wc.DataType.TEXT,
                        index_searchable=True,
                    ),  # stored readable text
                    wc.Property(
                        name="chunk_id",
                        data_type=wc.DataType.TEXT,
                        index_searchable=True,
                    ),
                ],
                # CONFIGURATION
                vector_config=[
                    wc.Configure.Vectors.self_provided(
                        name="custom_vector",
                        vector_index_config=wc.Configure.VectorIndex.hnsw(
                            ef_construction=self.ef_construction,
                            distance_metric=wc.VectorDistances.COSINE,
                        ),
                    )
                ],
                inverted_index_config=wc.Configure.inverted_index(
                    bm25_b=self.bm25_b,
                    bm25_k1=self.bm25_k1,
                    index_null_state=True,
                    index_property_length=True,
                    index_timestamps=True,
                ),
            )
            print(Fore.GREEN + f"New Collection {self.collection_name} created!")

        except Exception as e:
            print(Fore.RED + f"{e}")

        finally:
            if client:
                client.close()

    def store(self, chunk: dict[str, str]):
        """
        Insert one chunk into Weaviate collection with a self-provided vector.

        chunk: dict with keys 'file_name', 'title', 'chunk_id', 'content'
        """
        client = None
        try:
            client = weaviate.connect_to_local(**self.connection_config)
            collection = client.collections.get(self.collection_name)

            properties = {
                "file_name": chunk.get("file_name"),
                "title": chunk.get("title"),
                "content": chunk.get("content"),
                "chunk_id": chunk.get("chunk_id"),
            }

            # combine title + content for embedding (so vector represents both fields)
            combined_text = f"passage: {properties['content']}"

            vector = self.embeddings.encode(combined_text)
            collection.data.insert(
                properties=properties, vector={"custom_vector": vector}
            )

            print(
                Fore.GREEN
                + f"Inserted object chunk_id={properties['chunk_id']} title={properties['title']}"
            )
        except Exception as e:
            # consider using logging instead of print in real code
            print(Fore.RED + "Failed to insert object:", e)
        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass

    def search(self, query: str, alpha: int = 1, k: int = 5):
        """
        **Hybrid Search** = Keyword Search + Semantic Search
            - An alpha of 1 is a pure vector search.
            - An alpha of 0 is a pure keyword search.
        """
        query = f"query: {query}"
        client = None
        try:
            client = weaviate.connect_to_local(**self.connection_config)
            collection = client.collections.get(self.collection_name)

            response = collection.query.hybrid(
                query=query,
                alpha=alpha,
                target_vector="custom_vector",
                query_properties=["title", "content"],
                vector=self.embeddings.encode(query),
                limit=k,
            )
            return response.objects

        except Exception as e:
            print(Fore.RED + f"{e}")
        finally:
            if client:
                client.close()
