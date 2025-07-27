# 4. Implementation

## 4.1 Tech Stack
- Python (Hugging Face, FAISS, Flask)
- Kotlin (optional for mobile front-end)
- Google Colab for model training
- Together.ai for optional inference hosting

## 4.2 RAG Pipeline
1. User submits a question
2. Classifier predicts the intent
3. Documents matching that intent are retrieved
4. Generator formulates a response

## 4.3 Interface
- Basic chat UI via Streamlit or FastAPI
- Supports English input; future version may support multilingual