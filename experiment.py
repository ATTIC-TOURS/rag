import os
import time
import pandas as pd
from langchain_openai import ChatOpenAI
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.evaluation import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from ragas.run_config import RunConfig
from datasets import Dataset

from rag_pipeline.rag_pipeline import build_rag_pipeline
from llama_index.core.prompts import PromptTemplate
from dotenv import load_dotenv
from colorama import init, Fore
init(autoreset=True)

load_dotenv()

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
    - query (renamed to 'question')
    - answer (renamed to 'ground_truth')
    - optional: filter rows based on `check` column
    """
    df = df.copy()

    # Rename columns to match RAGAS expected format
    df = df.rename(columns={"query": "question", "answer": "ground_truth"})

    # Drop rows with missing data
    df = df.dropna(subset=["question", "ground_truth"])

    # Optional: filter only rows where check == 1 (if you use that column for QA validation)
    # if "check" in df.columns:
    #     df = df[df["check"] == 1]

    return df[["query_id", "question", "ground_truth"]]


def load_queries() -> pd.DataFrame:
    print(Fore.CYAN + 'Loading Query and Answer pairs from Google Spreadsheet..')
    df = extract_queries()
    df = transform_queries(df)
    print(Fore.CYAN + f'💿 Total Rows: {len(df)} retrieved')
    return df


EVAL_DF = load_queries()

# Convert DataFrame to list of dicts for iteration
EVAL_QUERIES = EVAL_DF.to_dict(orient="records")


def run_experiment(params, openai_api_key):
    """
    Run one experiment with given params, return averaged metrics + latency.
    """
    
    query_engine, client = build_rag_pipeline(**params)

    results = []
    latencies = []

    # Generate response for each query
    print(Fore.BLUE + f'🤖 generating response..')
    for q in EVAL_QUERIES:
        start = time.time()
        response = query_engine.query(q["question"])
        elapsed = time.time() - start
        latencies.append(elapsed)

        results.append(
            {
                "question": q["question"],
                "answer": str(response),
                "contexts": [n.get_content() for n in response.source_nodes],
                "ground_truth": q["ground_truth"],
            }
        )
    print(Fore.BLUE + f'response has been generated')

    client.close()

    dataset = Dataset.from_list(results)

    # Define LLM for evaluation
    # llm = ChatOpenAI(
    #     model="gpt-4o-mini",
    #     api_key=openai_api_key,
    #     temperature=0
    # )

    eval_llm = ChatOllama(model="gemma3:1b", temperature=0.0)
    ragas_llm = LangchainLLMWrapper(eval_llm)

    eval_embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

    run_config = RunConfig(timeout=7200)

    print(Fore.BLUE + 'evaluating..')
    eval_results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )

    # for retriever and generator
    avg_metrics = eval_results.to_pandas().describe().loc["mean"].to_dict()

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
    print(Fore.BLUE + f"📃 Logged experiment: {record}")

def count_experiment_variables(vars):
    count = 1
    for var in vars:
        count *= len(var)
    return count

def run_all_experiments(openai_api_key):
    """
    Run a grid of experiments with different parameter settings.
    """
    # Search space
    similarity_top_k_values = [3]
    alpha_values = [0.8]
    rerank_options = ["cross-encoder/ms-marco-MiniLM-L-2-v2"]
    prompt_templates = [
        PromptTemplate(
            "You are a helpful assistant that answers Japan visa questions.\n\n"
            "Question: {query_str}\n\n"
            "Here are the retrieved documents:\n{context_str}\n\n"
            "Answer clearly and concisely."
        ),
    ]
    
    num_variables = count_experiment_variables([
        similarity_top_k_values,
        alpha_values,
        rerank_options,
        prompt_templates
    ])
    
    print(Fore.GREEN + f'Total Experiment: {num_variables}')

    current_ongoing_experiment = 1
    for top_k in similarity_top_k_values:
        for alpha in alpha_values:
            for reranker in rerank_options:
                for prompt_template in prompt_templates:
                    params = {
                        "index_name": "Requirements",
                        "embeddings_model_name": "intfloat/multilingual-e5-base",
                        "llm_model_name": "gemma3:1b",  
                        "similarity_top_k": top_k,  
                        "alpha": alpha,  
                        "prompt_template": prompt_template, 
                        "cross_encoder_model": reranker,
                        "rerank_top_n": 1 if reranker else None, 
                        "llm_provider": "ollama",  
                        "openai_api_key": openai_api_key,  
                    }

                    try:
                        print(Fore.GREEN + f'🟢 {current_ongoing_experiment}/{num_variables} running experiment on params: {params}')
                        metrics = run_experiment(params, openai_api_key)
                        log_results(params, metrics)
                    except Exception as e:
                        print(f"❌ Failed experiment {params}: {e}")
                        
    print(Fore.GREEN + f"✅ {num_variables} experiments done")

if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    run_all_experiments(OPENAI_API_KEY)
