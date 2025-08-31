import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Protobuf gencode version.*")

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

def evaluate_retriever_at_k(df: pd.DataFrame, k: int) -> dict[str, float]:
    df = df.copy()

    precision_per_query_at_k = get_precision_per_query_at_k(df, k)
    mean_precision_at_k = precision_per_query_at_k[f"Precision@{k}"].mean()

    recall_per_query_at_k = get_recall_per_query_at_k(df, k)
    mean_recall_at_k = recall_per_query_at_k[f"Recall@{k}"].mean()
    
    f1_per_query_at_k = get_f1_per_query_at_k(df, k)
    mean_f1_at_k = f1_per_query_at_k.mean()

    return {
        f"Mean Precision@{k}": mean_precision_at_k,
        f"Mean Recall@{k}": mean_recall_at_k,
        f"Mean F1@{k}": mean_f1_at_k
    }


def evaluate_retriever(annotation_pools: pd.DataFrame, top_k: tuple[int]):
    evaluations_at_k = []
    for k in top_k:
        evaluation_at_k = evaluate_retriever_at_k(annotation_pools, k)
        evaluations_at_k.append(evaluation_at_k)
    
    evaluation = {}
    for evaluation_at_k in evaluations_at_k:
        for key, value in evaluation_at_k.items():
            evaluation[key] = value
    
    return evaluation


def get_metadata(path: str) -> dict[str, str]:
    with open(path) as f:
        metadata = {}
        while True:
            line = f.readline()
            if line[0] != "#": break
            splitted_line = line.split()
            key, value = splitted_line[1:3]
            metadata[key[:-1]] = value
    return metadata

def record_retriever_evaluation() -> None:
    BASE_PATH = "src/annotations"
    if len(os.listdir(BASE_PATH)) == 0:
        print("No Annotations Record")
        return
    for path in os.listdir(BASE_PATH):
        df = pd.read_csv(f'{BASE_PATH}/{path}', comment="#")
        retriever_evaluation = evaluate_retriever(df, top_k = (3, 5, 10))
        metadata = get_metadata(f'{BASE_PATH}/{path}')
        new_retriever_evalution = retriever_evaluation | metadata
        for key, value in new_retriever_evalution.items():
            new_retriever_evalution[key] = [value]
        try:
            old_record = None
            old_record = pd.read_csv("src/evaluations/retriever_evaluation.csv")
        except:
            pass
        new_record = pd.DataFrame(new_retriever_evalution)
        pd.concat([old_record, new_record]).to_csv("src/evaluations/retriever_evaluation.csv", index=False)
        os.remove(f'{BASE_PATH}/{path}')
    
    print("New Evaluation Recorded!")

def main():
    record_retriever_evaluation()

if __name__ == "__main__":
    main()