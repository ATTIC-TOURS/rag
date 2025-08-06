import weaviate
import weaviate.classes.config as wc
from sentence_transformers import SentenceTransformer
from colorama import Fore, Style, init
init(autoreset=True)


class MyWeaviateDB:

    connection_config = {"port": 8080, "grpc_port": 50051, "skip_init_checks": True}

    def __init__(
        self,
        embeddings,
        collection_name="Requirements",
        ef_construction=300,
        bm25_b=0.7,
        bm25_k1=1.25,
    ):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.ef_construction = ef_construction
        self.bm25_b = bm25_b
        self.bm25_k1 = bm25_k1

    def setup_collection(self):       
        try:
            client = weaviate.connect_to_local(**self.connection_config)
            
            client.collections.delete_all()
            print(Fore.RED + "All collections has been removed!")
            
            if not client.collections.exists(self.collection_name):
                client.collections.create(
                    # NAME
                    name=self.collection_name,
                    # PROPERTIES
                    properties=[
                        wc.Property(name="file_name", data_type=wc.DataType.TEXT),
                        wc.Property(
                            name="title", data_type=wc.DataType.TEXT
                        ),  # used for filtering
                        wc.Property(
                            name="content", data_type=wc.DataType.TEXT
                        ),  # stored readable text
                        wc.Property(name="chunk_id", data_type=wc.DataType.TEXT),
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
                print(Fore.GREEN + "New Collection created!")
            else:
                print(f"Collection: '{self.collection_name}' already exists!")
        except Exception as e:
            print(Fore.RED + f'{e}')
        
        finally:
            if client:
                client.close()

    def reset(self):
        client = None
        try:
            client = weaviate.connect_to_local(**self.connection_config)
            
        except Exception as e:
            print(e)
        finally:
            if client:
                client.close()

    def store(self, chunk):
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
            combined_text = f"{properties['title']}\n\n{properties['content']}"

            vector = self.embeddings.encode(combined_text)
            collection.data.insert(properties=properties, vector=vector)

            print(Fore.GREEN + 
                f"Inserted object chunk_id={properties['chunk_id']} title={properties['title']}"
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

    def search(self, query, alpha, k=3):
        """
        **Hybrid Search** = Keyword Search + Semantic Search
            - An alpha of 1 is a pure vector search.
            - An alpha of 0 is a pure keyword search.
        """
        client = None
        try:
            client = weaviate.connect_to_local(**self.connection_config)
            documents = client.collections.get(self.collection_name)
            response = documents.query.hybrid(
                query=query,
                alpha=alpha,  # Balance between keyword and vector
                target_vector="custom_vector",  # Specify which named vector to use for the vector component
                query_properties=[
                    "text"
                ],  # Which properties to use for the keyword (BM25) search
                # IMPORTANT: If your collection has NO vectorizer_config at the class level
                # and only self_provided named vectors, you might need to also provide `vector`
                # within the hybrid query to explicitly give the query vector.
                # The Weaviate docs indicate `query` is used for both, but if your setup is strict
                # with `vectorizer_config=none()`, `query` won't be auto-vectorized.
                # In such cases, if you want semantic search as part of hybrid, you must provide the vector.
                vector=self.embeddings.encode(query),
                limit=k,
            )

            return response.objects

        except Exception as e:
            print(Fore.RED + f"{e}")
        finally:
            if client:
                client.close()


def main():
    embeddings = SentenceTransformer("intfloat/multilingual-e5-base")
    myDB = MyWeaviateDB(embeddings=embeddings)


if __name__ == "__main__":
    main()
