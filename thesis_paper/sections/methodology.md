# 3. Methodology

## 3.1 System Overview
The chatbot pipeline consists of:
- A user query input
- An intent classification model
- A filtered retrieval from a vector store
- A generator model (e.g., LLaMA or Mistral) that creates the final response

## 3.2 Data Collection
- Scraped FAQs from Japan embassies
- Reddit (r/JapanTravel, r/immigration)
- Public visa datasets and ChatGPT-generated questions

## 3.3 Preprocessing
- Text cleaned and chunked into ~300-token documents
- Embedded using sentence-transformers
- Stored in FAISS for retrieval

## 3.4 Intent Classifier
- Trained on 1,000+ labeled questions across 8 visa categories
- Models used:
  - TF-IDF + Logistic Regression (baseline)
  - DistilBERT fine-tuned (final model)