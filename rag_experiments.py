import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import asyncio
import pandas as pd
import warnings
from dotenv import load_dotenv
from colorama import init, Fore

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

from rag.rag_pipeline import build_rag_pipeline
from rag.indexing.modules import set_global_embeddings
from llama_index.core.prompts import PromptTemplate

# === setup ===
load_dotenv()
init(autoreset=True)
warnings.filterwarnings("ignore", category=ResourceWarning)

RESULTS_FILE = "rag_experiment_results.csv"


# === data loading ===
def extract_queries() -> pd.DataFrame:
    url_source: str = (
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vR1hUlRhTJQgNzSbTyRtDNh1mCrbfy0iUm6oiHK7oHb_iQQ5t7XCB_xyUCwoZ2fdg/pub?output=xlsx"
    )
    queries = pd.read_excel(url_source, sheet_name="queries")
    return queries


def transform_queries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={"query": "user_input", "answer": "reference"})
    df = df.dropna(subset=["user_input", "reference"])
    return df[["query_id", "user_input", "reference"]]


def load_queries() -> pd.DataFrame:
    print(Fore.CYAN + "Loading Query and Answer pairs from Google Spreadsheet..")
    df = extract_queries()
    df = transform_queries(df)
    print(Fore.CYAN + f"💿 Total Rows: {len(df)} retrieved")
    return df


EVAL_DF = load_queries()
EVAL_QUERIES = EVAL_DF.to_dict(orient="records")


# === experiment ===
async def run_experiment(params):
    """
    Run one experiment with given params, return averaged metrics + latency.
    """
    query_engine, client = await build_rag_pipeline(**params)

    results = []
    latencies = []

    len_eval_queries = len(EVAL_QUERIES)
    for idx, q in enumerate(EVAL_QUERIES):
        print(Fore.BLUE + f"{idx+1}/{len_eval_queries} 🤖 generating response..")
        start = time.time()

        response = await query_engine.query(q["user_input"])
        elapsed = time.time() - start
        latencies.append(elapsed)

        # Normalize response (string vs Response object)

        if isinstance(response, dict):  # new two-stage dict
            facts = response["facts"]
            final_answer = response["final_answer"]
            retrieved_contexts = [n.get_content() for n in facts.source_nodes]
        else:  # fallback (string)
            final_answer = str(response)
            retrieved_contexts = []

        results.append(
            {
                "user_input": q["user_input"],
                "response": final_answer,
                "retrieved_contexts": retrieved_contexts,
                "reference": q["reference"],
            }
        )

    await client.close()

    dataset = EvaluationDataset.from_list(results)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    evaluator_llm = LangchainLLMWrapper(llm)

    run_config = RunConfig(timeout=7200)

    eval_results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            FactualCorrectness(atomicity="low", coverage="low"),
            FactualCorrectness(mode="precision", atomicity="low", coverage="low"),
            FactualCorrectness(mode="recall", atomicity="low", coverage="low"),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=evaluator_llm,
        run_config=run_config,
    )

    avg_metrics = eval_results.to_pandas().describe().loc["mean"].to_dict()

    # latency
    avg_metrics["avg_latency_sec"] = sum(latencies) / len(latencies)
    avg_metrics["latency_95th_sec"] = sorted(latencies)[int(0.95 * len(latencies)) - 1]

    return avg_metrics


# === logging ===
def log_results(params, metrics):
    record = {**params, **metrics}

    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        new_df = pd.DataFrame([record])
        new_df["#Experiment"] = 18
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(RESULTS_FILE, index=False)
    print(Fore.BLUE + f"📝 Logged Experiment")


# === run all ===
async def run_all_experiments():
    embeddings_settings = {
        "Custom_splitter_w_context_hf": {
            "model_name": "intfloat/multilingual-e5-base",
            "provider": "hf",
        },
    }

    experiment_vars = [
        {
            "index_name": "Custom_splitter_w_context_hf",
            "alpha": 0.8,
            "base_k": 5,
            "expansion_k": 5,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
            "rerank_top_n": 5,
            "fact_prompt": PromptTemplate(
                """You are a Japan visa assistant.

Your task is to answer the user's question using ONLY the retrieved documents. 
Follow these strict rules:
1. Use ONLY facts explicitly stated in the retrieved documents. 
   - If something is not mentioned, DO NOT invent, assume, or guess.
2. When multiple documents provide overlapping or complementary information, merge them into a single clear answer without losing details.
3. Present the answer in a structured, list-like format when appropriate (to maximize coverage of factual details).
4. Be concise and avoid repetition, but ensure completeness of the retrieved facts.
5. If the documents do not fully answer the question, clearly state what is missing.

Question: {query_str}

Retrieved documents:
{context_str}

Final Answer:"""
            ),
            "two_stage": False,
            "use_query_expansion": True,
            "query_expansion_num": 3,
        },
    ]

    len_experiment_vars = len(experiment_vars)
    print(Fore.GREEN + f"Total Experiment: {len_experiment_vars}")

    current_ongoing_experiment = 1
    for experiment_var in experiment_vars:
        set_global_embeddings(**embeddings_settings[experiment_var["index_name"]])  # type: ignore
        params = experiment_var
        try:
            print(
                Fore.GREEN
                + f"🟢 {current_ongoing_experiment}/{len_experiment_vars} running experiment on: {experiment_var}"
            )
            metrics = await run_experiment(params)
            log_results(params, metrics)
            current_ongoing_experiment += 1
        except Exception as e:
            print(f"❌ Failed experiment {params}: {e}")

    print(
        Fore.GREEN
        + f"✅ {current_ongoing_experiment-1}/{len_experiment_vars} experiments done"
    )


if __name__ == "__main__":
    asyncio.run(run_all_experiments())
