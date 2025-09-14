from rag_pipeline.rag_pipeline import RagPipeline
from sklearn.pipeline import Pipeline
from rag_pipeline.classifier.japan_visa_related_or_not.modules import (
    MyTextCleaner,
    MyEmbeddingTransformer,
)
import joblib
from colorama import init, Fore

init(autoreset=True)


rag_pipeline = RagPipeline(collection_name="test")
# chatbot.prepare_docs(from_google_drive=True)
clf_japan_visa_related: Pipeline = joblib.load(
    "rag_pipeline/classifier/japan_visa_related_or_not/japan_visa_related_classifier.pkl"
)
while True:
    query = input("query:")
    is_japan_visa_related = clf_japan_visa_related.predict([query])[0]
    if is_japan_visa_related:
        print(Fore.GREEN + "your query is related to Japan Visa")
        relevant_docs = rag_pipeline._retrieved_relevant_docs(query)
        messages = rag_pipeline._get_messages(query, relevant_docs)
        print(f'{messages[0]["content"]} {messages[1]["content"]}')
    else:
        print(Fore.RED + "only Japan Visa related is allowed")
