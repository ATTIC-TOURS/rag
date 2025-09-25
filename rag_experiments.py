import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import pandas as pd

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    FactualCorrectness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig
from ragas import EvaluationDataset

from langchain_community.chat_models import ChatOllama

from rag.rag_pipeline import build_rag_pipeline
from rag.indexing.modules import set_global_embeddings
from llama_index.core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
from colorama import init, Fore

init(autoreset=True)
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)

RESULTS_FILE = "rag_experiment_results.csv"


def extract_queries() -> pd.DataFrame:
    url_source: str = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vR1hUlRhTJQgNzSbTyRtDNh1mCrbfy0iUm6oiHK7oHb_iQQ5t7XCB_xyUCwoZ2fdg/pub?output=xlsx"
    )
    # Load only the `queries` sheet
    queries = pd.read_excel(url_source, sheet_name="queries")
    return queries


def transform_queries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and keep only relevant columns:
    - query_id
    - query (renamed to 'user_input')
    - answer (renamed to 'reference')
    - optional: filter rows based on `check` column
    """
    df = df.copy()

    # Rename columns to match RAGAS expected format
    df = df.rename(columns={"query": "user_input", "answer": "reference"})

    # Drop rows with missing data
    df = df.dropna(subset=["user_input", "reference"])

    # Optional: filter only rows where check == 1 (if you use that column for QA validation)
    # if "check" in df.columns:
    #     df = df[df["check"] == 1]

    return df[["query_id", "user_input", "reference"]]


def load_queries() -> pd.DataFrame:
    print(Fore.CYAN + "Loading Query and Answer pairs from Google Spreadsheet..")
    df = extract_queries()
    df = transform_queries(df)
    print(Fore.CYAN + f"💿 Total Rows: {len(df)} retrieved")
    return df


EVAL_DF = load_queries()

# Convert DataFrame to list of dicts for iteration
EVAL_QUERIES = EVAL_DF.to_dict(orient="records")


def run_experiment(params):
    """
    Run one experiment with given params, return averaged metrics + latency.
    """

    query_engine, client = build_rag_pipeline(**params)

    results = []
    latencies = []

    # Generate response for each query
    len_eval_queries = len(EVAL_QUERIES)
    for idx, q in enumerate(EVAL_QUERIES):
        print(Fore.BLUE + f"{idx+1}/{len_eval_queries} 🤖 generating response..")
        start = time.time()
        response = query_engine.query(q["user_input"])
        elapsed = time.time() - start
        latencies.append(elapsed)

        results.append(
            {
                "user_input": q["user_input"],
                "response": str(response),
                "retrieved_contexts": [n.get_content() for n in response.source_nodes],
                "reference": q["reference"],
            }
        )

    client.close()

    dataset = EvaluationDataset.from_list(results)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    # llm = ChatOllama(model="gemma3:1b", temperature=0.0)
    evaluator_llm = LangchainLLMWrapper(llm)

    run_config = RunConfig(timeout=7200)

    eval_results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            FactualCorrectness(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=evaluator_llm,
        run_config=run_config,
    )

    # for retriever and generator
    avg_metrics = eval_results.to_pandas().describe().loc["mean"].to_dict()  # type: ignore

    # latency
    avg_metrics["avg_latency_sec"] = sum(latencies) / len(latencies)
    avg_metrics["latency_95th_sec"] = sorted(latencies)[int(0.95 * len(latencies)) - 1]

    return avg_metrics


def log_results(params, metrics):
    record = {**params, **metrics}

    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(RESULTS_FILE, index=False)
    print(Fore.BLUE + f"📝 Logged Experiment")


def count_experiment_variables(vars):
    count = 1
    for var in vars:
        count *= len(var)
    return count


def run_all_experiments():
    """
    Run a grid of experiments with different parameter settings.
    """
    # Search space
    index_names = [
        "Normal_splitter_hf",
        "Normal_splitter_openai",
        "Normal_splitter_w_context_hf",
        "Normal_splitter_w_context_openai",
        "Custom_splitter_hf",
        "Custom_splitter_openai",
        "Custom_splitter_w_context_hf",
        "Custom_splitter_w_context_openai",
    ]
    embeddings_settings = {
        "Normal_splitter_hf": {
            "model_name": "intfloat/multilingual-e5-base",
            "provider": "hf",
        },
        "Normal_splitter_w_context_hf": {
            "model_name": "intfloat/multilingual-e5-base",
            "provider": "hf",
        },
        "Custom_splitter_hf": {
            "model_name": "intfloat/multilingual-e5-base",
            "provider": "hf",
        },
        "Custom_splitter_w_context_hf": {
            "model_name": "intfloat/multilingual-e5-base",
            "provider": "hf",
        },
        "Normal_splitter_openai": {
            "model_name": "text-embedding-3-small",
            "provider": "openai",
        },
        "Normal_splitter_w_context_openai": {
            "model_name": "text-embedding-3-small",
            "provider": "openai",
        },
        "Custom_splitter_openai": {
            "model_name": "text-embedding-3-small",
            "provider": "openai",
        },
        "Custom_splitter_w_context_openai": {
            "model_name": "text-embedding-3-small",
            "provider": "openai",
        },
    }
    alpha_values = [0.8, 1.0]  # hybrid search or not
    similarity_top_k_values = [3, 5, 10, 15]
    rerank_options = [
        None,
        "cross-encoder/ms-marco-MiniLM-L-2-v2",
    ]  # has reranker or none

    # num_variables = count_experiment_variables(
    #     [
    #         index_names,
    #         alpha_values,
    #         similarity_top_k_values,
    #         rerank_options,
    #     ]
    # )
    missing_vars = [
        {
            "index_name": "Custom_splitter_wfxc bckll//_ n nmb jhjjjjhhhhjjhhhj   xaqęh nvf",
            "alpha": 1.0,
            "similarity_top_k": 10,
            "cross_encoder_model": None,
        },
        {
            "index_name": "Custom_splitter_w_context_hf",
            "alpha": 1.0,
            "similarity_top_k": 10,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        },
        {
            "index_name": "Custom_splitter_w_context_hf",
            "alpha": 1.0,
            "similarity_top_k": 15,
            "cross_encoder_model": None,
        },
        {
            "index_name": "Custom_splitter_w_context_hf",
            "alpha": 1.0,
            "similarity_top_k": 15,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 0.8,
            "similarity_top_k": 3,
            "cross_encoder_model": None,
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 0.8,
            "similarity_top_k": 3,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 0.8,
            "similarity_top_k": 5,
            "cross_encoder_model": None,
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 0.8,
            "similarity_top_k": 5,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 0.8,
            "similarity_top_k": 10,
            "cross_encoder_model": None,
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 0.8,
            "similarity_top_k": 10,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 0.8,
            "similarity_top_k": 15,
            "cross_encoder_model": None,
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 0.8,
            "similarity_top_k": 15,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 1.0,
            "similarity_top_k": 3,
            "cross_encoder_model": None,
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 1.0,
            "similarity_top_k": 3,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 1.0,
            "similarity_top_k": 5,
            "cross_encoder_model": None,
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 1.0,
            "similarity_top_k": 5,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 1.0,
            "similarity_top_k": 10,
            "cross_encoder_model": None,
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 1.0,
            "similarity_top_k": 10,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 1.0,
            "similarity_top_k": 15,
            "cross_encoder_model": None,
        },
        {
            "index_name": "Custom_splitter_w_context_openai",
            "alpha": 1.0,
            "similarity_top_k": 15,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        },
    ]

    len_missing_vars = len(missing_vars)
    print(Fore.GREEN + f"Total Experiment: {len_missing_vars}")

    fixed_params = {
        "llm_model_name": "gpt-4o-mini",
        "rerank_top_n": 3,
        "llm_provider": "openai",
        "prompt_template": PromptTemplate(
            "You are a helpful assistant that answers Japan visa questions.\n\n"
            "Question: {query_str}\n\n"
            "Here are the retrieved documents:\n{context_str}\n\n"
            "Answer clearly and concisely."
        ),
    }

    current_ongoing_experiment = 1
    for missing_var in missing_vars:
        set_global_embeddings(**embeddings_settings[missing_var["index_name"]])  # type: ignore
        params = missing_var | fixed_params
        try:
            print(
                Fore.GREEN
                + f"🟢 {current_ongoing_experiment}/{len_missing_vars} running experiment on: {missing_var}"
            )
            metrics = run_experiment(params)
            log_results(params, metrics)
            current_ongoing_experiment += 1
        except Exception as e:
            print(f"❌ Failed experiment {params}: {e}")

    print(
        Fore.GREEN
        + f"✅ {current_ongoing_experiment-1}/{len_missing_vars} experiments done"
    )

    # current_ongoing_experiment = 1
    # for index_name in index_names:
    #     for alpha in alpha_values:
    #         for top_k in similarity_top_k_values:
    #             for reranker in rerank_options:
    #                 set_global_embeddings(**embeddings_settings[index_name])  # type: ignore
    #                 current_vars = {
    #                     "index_name": index_name,
    #                     "alpha": alpha,
    #                     "similarity_top_k": top_k,
    #                     "cross_encoder_model": reranker,
    #                 }

    #                 params = current_vars | fixed_params

    #                 try:
    #                     print(
    #                         Fore.GREEN
    #                         + f"🟢 {current_ongoing_experiment}/{num_variables} running experiment on: {current_vars}"
    #                     )
    #                     metrics = run_experiment(params)
    #                     log_results(params, metrics)
    #                     current_ongoing_experiment += 1
    #                 except Exception as e:
    #                     print(f"❌ Failed experiment {params}: {e}")

    # print(
    #     Fore.GREEN
    #     + f"✅ {current_ongoing_experiment-1}/{num_variables} experiments done"
    # )


if __name__ == "__main__":
    run_all_experiments()
