import pandas as pd
import gradio as gr
from retriever import Retriever
from datetime import datetime
import uuid
import os
from sentence_transformers import SentenceTransformer
from vector_db.chunking_strategy import section_based_chunking


def extract_queries() -> pd.DataFrame:
    url_source: str = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vR1hUlRhTJQgNzSbTyRtDNh1mCrbfy0iUm6oiHK7oHb_iQQ5t7XCB_xyUCwoZ2fdg/pub?output=xlsx"
    )
    queries = pd.read_excel(url_source, sheet_name="queries")
    return queries


def remove_null_entries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df[df["intent"].notna()]


def get_only_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    target_columns = ["query_id", "query"]
    return df[target_columns]


def transform_queries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = remove_null_entries(df)
    df = get_only_target_columns(df)
    return df


def load_queries() -> pd.DataFrame:
    df = extract_queries()
    df = transform_queries(df)
    return df


def get_precision_per_query_at_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df.copy()

    str_query_id = "query_id"
    str_relevant = "relevant"
    str_precision_at_k = f"Precision@{k}"

    df_topk = df[df["rank"] <= k]

    precision_per_query_at_k = (
        df_topk.groupby(str_query_id)
        .apply(lambda g: g[str_relevant].sum() / k)
        .reset_index(name=str_precision_at_k)
    )
    return precision_per_query_at_k


def get_recall_per_query_at_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df.copy()

    str_query_id = "query_id"
    str_rank = "rank"
    str_relevant = "relevant"
    str_recall_at_k = f"Recall@{k}"
    str_retrieved_relevant = "retrieved_relevant"
    str_total_relevant = "total_relevant"

    total_relevant = (
        df.groupby(str_query_id)[str_relevant]
        .sum()
        .reset_index(name=str_total_relevant)
    )

    df_topk = df[df[str_rank] <= k]

    recall_per_query_at_k = (
        df_topk.groupby(str_query_id)
        .apply(lambda g: (g[str_relevant].sum()))
        .reset_index(name=str_retrieved_relevant)
        .merge(total_relevant, on=str_query_id)
    )

    recall_per_query_at_k[str_recall_at_k] = (
        recall_per_query_at_k[str_retrieved_relevant]
        / recall_per_query_at_k[str_total_relevant]
    )

    return recall_per_query_at_k


def get_f1_per_query_at_k(df: pd.DataFrame, k: int) -> pd.Series:
    df = df.copy()

    str_precision_at_k = f"Precision@{k}"
    str_recall_at_k = f"Recall@{k}"

    precision_per_query_at_k = get_precision_per_query_at_k(df, k)
    recall_per_query_at_k = get_recall_per_query_at_k(df, k)

    f1_per_query_at_k = (
        2
        * precision_per_query_at_k[str_precision_at_k]
        * recall_per_query_at_k[str_recall_at_k]
    ) / (
        precision_per_query_at_k[str_precision_at_k]
        + recall_per_query_at_k[str_recall_at_k]
    )

    return f1_per_query_at_k


def generate_annotation_pools(
    queries: pd.DataFrame, retriever: Retriever, alpha: int, k: int
) -> pd.DataFrame:
    annotation_pools_data = {"query_id": [], "query": [], "rank": []}
    for query_id, query in queries.to_numpy():

        retrieved_docs = retriever.retrieve_relevant_docs(query=query, alpha=alpha, k=k)

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
    folder = "src/annotations"
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
<b style="color:blue;">Query {row.query_id}:</b> {row.query}  
<br>
📑 Document
<br>
<b style="color:darkred;">File Name:</b> {row.file_name}  
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
        "embeddings": SentenceTransformer("intfloat/multilingual-e5-base"),
        "chunking strategy": section_based_chunking,
        "ef_construction": 300,
        "bm25_b": 0.7,
        "bm25_k1": 1.25,
        "alpha": 1,
        "k": 10,
    }

    retriever = Retriever(
        embeddings=metadata["embeddings"],
        ef_construction=metadata["ef_construction"],
        bm25_b=metadata["bm25_b"],
        bm25_k1=metadata["bm25_k1"],
    )

    retriever.pre_compute_docs(metadata["chunking strategy"])

    annotation_pools = generate_annotation_pools(
        queries, retriever, alpha=metadata["alpha"], k=metadata["k"]
    )
    annotate(
        annotation_pools[:10],
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
