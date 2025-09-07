import pandas as pd
import gradio as gr
from retriever import Retriever
from datetime import datetime
import uuid
import os
from sentence_transformers import SentenceTransformer
from retriever.prepare_docs_strategy import PrepareDocsStrategy
from text_cleaning_strategy.base import TextCleaningStrategy
from text_cleaning_strategy.docs.v1 import DocsCleaningStrategyV1
from text_cleaning_strategy.docs.v2 import DocsCleaningStrategyV2
from text_cleaning_strategy.query.v1 import QueryCleaningStrategyV1
from chunking_strategy.base import ChunkingStrategy
from chunking_strategy.v1 import ChunkingStrategyV1
from chunking_strategy.fixed_window_chunking import FixedWindowChunking
from chunking_strategy.pdf_based_chunking import PdfBasedChunking
from chunking_strategy.recursively_split_chunking import RecursivelySplitChunking
from chunking_strategy.pdf_based_recursively_split_chunking import (
    PdfBasedRecursivelySplitChunking,
)
from vector_db.vector_db import MyWeaviateDB
from summarizer.summarizer import SummarizerLLM
from contextual_retrieval.context_embedder import ContextEmbedderLLM
from sentence_transformers.cross_encoder import CrossEncoder
import time
from colorama import Fore, init

init(autoreset=True)


def extract_queries() -> pd.DataFrame:
    url_source: str = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vR1hUlRhTJQgNzSbTyRtDNh1mCrbfy0iUm6oiHK7oHb_iQQ5t7XCB_xyUCwoZ2fdg/pub?output=xlsx"
    )
    queries = pd.read_excel(url_source, sheet_name="queries")
    return queries


def transform_queries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    target_columns = ["query_id", "query"]
    return df[target_columns]


def load_queries() -> pd.DataFrame:
    df = extract_queries()
    df = transform_queries(df)
    return df


def generate_annotation_pools(
    queries: pd.DataFrame, retriever: Retriever, alpha: int, top_k: int
) -> pd.DataFrame:
    annotation_pools_data = {"query_id": [], "query": [], "rank": []}
    for query_id, query in queries.to_numpy():
        print(Fore.GREEN + f"retrieving docs for query: {query}")
        start_time = time.time()
        retrieved_docs = retriever.search(query=query, alpha=alpha, top_k=top_k)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"retrieved {len(retrieved_docs)} docs in {elapsed:.2f} seconds")
        for rank, doc in enumerate(retrieved_docs):
            # dynamically set the column name
            for column in list(doc.properties.keys()):
                annotation_pools_data.setdefault(column, [])
            for key, value in doc.properties.items():
                annotation_pools_data[key].append(value)
            annotation_pools_data["query_id"].append(query_id)
            annotation_pools_data["query"].append(query)
            annotation_pools_data["rank"].append(rank + 1)  # rank is 0-based

    annotation_pools = pd.DataFrame(annotation_pools_data)

    return annotation_pools


def save_with_file_metadata(df: pd.DataFrame, metadata: dict, fname: str) -> None:
    folder = "annotations"
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, f"{fname}.csv")

    # Save metadata first
    with open(path, "w") as f:
        for k, v in metadata.items():
            f.write(f"# {k}: {v}\n")
    # Append dataframe
    df.to_csv(path, index=False, mode="a")


def annotate(df: pd.DataFrame, metadata: dict) -> None:
    df["relevant"] = None

    def init_session(name):
        if not name.strip():  # fail if name is empty
            return (
                None,
                None,
                None,
                None,
                gr.update(visible=True),
                gr.update(value="❌ Please enter your name!"),
            )

        session_id = str(uuid.uuid4())[:8]
        metadata["timestamp"] = datetime.now().isoformat()
        return (
            df.copy(),
            0,
            session_id,
            name,
            gr.update(visible=False),
            gr.update(value=""),
        )  # hide start button

    def show_example(df, state, session_id, annotator):
        if state >= len(df):
            return (
                df,
                "✅ Annotation complete!",
                "All done!",
                state,
                gr.update(visible=True),
                session_id,
                annotator,
                gr.update(visible=False),
                gr.update(visible=False),
            )
        row = df.iloc[state]
        progress = f"""
    ### 📑 Document Annotation Tool  
    Progress: <span style="color:green; font-weight:bold;">{state+1}/{len(df)}</span>
    """
        doc_text = f"""
#### Is the Document relevant to the Query??
<b>Query {row.query_id}:</b> {row.query}  
<br>
📑 Document
<br>
<b>File Name:</b> {row.file_name}  
<div style="border:1px solid #555; padding:10px; max-height:250px; overflow-y:auto;">
{row.content}
</div>
"""
        return (
            df,
            progress,
            doc_text,
            state,
            gr.update(visible=False),
            session_id,
            annotator,
            gr.update(visible=True),
            gr.update(visible=True),
        )

    def annotate_and_next(label, df, state, session_id, annotator):
        if state < len(df):
            df.at[state, "relevant"] = int(label)
            state += 1
        return show_example(df, state, session_id, annotator)

    def stop_app(df, state, session_id, annotator):
        fname = f"annotations_{annotator}_{session_id}"
        save_with_file_metadata(df, metadata, fname)
        return (
            df,
            "👋 Session ended.",
            f"Saved in `{fname}`",
            state,
            gr.update(visible=False),
            session_id,
            annotator,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    with gr.Blocks() as demo:
        with gr.Row():
            name_input = gr.Textbox(
                label="Annotator Name", placeholder="Enter your name"
            )
            start_btn = gr.Button("Start")
            error_box = gr.Textbox(label="Error", interactive=False)

        df_state = gr.State()
        state = gr.State()
        session_id = gr.State()
        annotator = gr.State()

        title = gr.Markdown()
        # text = gr.Textbox(
        #     label="Is the Document relevant to the Query??", interactive=False
        # )

        text = gr.Markdown()

        with gr.Row():
            btn0 = gr.Button("Not Relevant (0)")
            btn1 = gr.Button("Relevant (1)")

        save_btn = gr.Button("Save", visible=False)

        start_btn.click(
            fn=init_session,
            inputs=name_input,
            outputs=[df_state, state, session_id, annotator, start_btn, error_box],
        ).then(
            fn=show_example,
            inputs=[df_state, state, session_id, annotator],
            outputs=[
                df_state,
                title,
                text,
                state,
                save_btn,
                session_id,
                annotator,
                btn0,
                btn1,
            ],
        )

        btn0.click(
            fn=lambda df, s, sid, a: annotate_and_next(0, df, s, sid, a),
            inputs=[df_state, state, session_id, annotator],
            outputs=[
                df_state,
                title,
                text,
                state,
                save_btn,
                session_id,
                annotator,
                btn0,
                btn1,
            ],
        )

        btn1.click(
            fn=lambda df, s, sid, a: annotate_and_next(1, df, s, sid, a),
            inputs=[df_state, state, session_id, annotator],
            outputs=[
                df_state,
                title,
                text,
                state,
                save_btn,
                session_id,
                annotator,
                btn0,
                btn1,
            ],
        )

        save_btn.click(
            fn=stop_app,
            inputs=[df_state, state, session_id, annotator],
            outputs=[
                df_state,
                title,
                text,
                state,
                save_btn,
                session_id,
                annotator,
                btn0,
                btn1,
            ],
        )

    demo.launch(share=True)


def main():
    queries = load_queries()

    metadata = {
        "no_of_questions": len(queries),
        "embeddings": "intfloat/multilingual-e5-base",
        "reranker": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "ef_construction": 300,
        "bm25_b": 0.7,
        "bm25_k1": 1.25,
        "alpha": 0.8,
        "top_k": 10,
    }

    db: MyWeaviateDB = MyWeaviateDB(
        collection_name="Test_requirements",
        ef_construction=metadata["ef_construction"],
        bm25_b=metadata["bm25_b"],
        bm25_k1=metadata["bm25_k1"],
    )
    embeddings: SentenceTransformer = SentenceTransformer(metadata["embeddings"])
    cross_encoder = CrossEncoder(metadata["reranker"])
    query_cleaning_strategy = QueryCleaningStrategyV1()
    retriever = Retriever(
        db=db,
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        text_cleaning_strategy=query_cleaning_strategy,
    )

    #################### INDEXING (PrepareDocsStrategy) START ####################
    # SETTINGS
    window_size = None
    overlap_size = None
    pdf_num_split = None
    chunk_overlap_rate = 0.2
    docs_cleaning_strategy = DocsCleaningStrategyV2()
    max_tokens = embeddings.get_max_seq_length()
    chunking_strategy = PdfBasedRecursivelySplitChunking(
        chunk_overlap_rate=chunk_overlap_rate, max_tokens=max_tokens
    )
    summarizer = None  # SummarizerLLM(model_name="gemma:2b")
    context_embedder = ContextEmbedderLLM(model_name="gemma:2b")

    prepareDocsStrategy = PrepareDocsStrategy(
        db=db,
        embeddings=embeddings,
        text_cleaning_strategy=docs_cleaning_strategy,
        chunking_strategy=chunking_strategy,
        summarizer=summarizer,
        context_embedder=context_embedder,
    )
    # retriever.prepare_docs(prepareDocsStrategy)

    metadata = metadata | {
        "docs_cleaning_strategy": prepareDocsStrategy.get_text_cleaning_strategy_name(),
        "query_cleaning_strategy": retriever.get_text_cleaning_strategy_name(),
        "chunking_strategy": prepareDocsStrategy.get_chunking_strategy_name(),
        "window_size": window_size,
        "overlap_size": overlap_size,
        "pdf_num_split": pdf_num_split,
        "chunk_overlap_rate": chunk_overlap_rate,
        "summarizer": True if summarizer else None,
        "context_embedder": True if context_embedder else None,
        "max_tokens": max_tokens,
    }

    #################### INDEXING (PrepareDocsStrategy) END ####################

    annotation_pools = generate_annotation_pools(
        queries, retriever, alpha=metadata["alpha"], top_k=metadata["top_k"]
    )
    annotate(
        annotation_pools,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
