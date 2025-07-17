# 📚 RAG-based Chatbot

# 🗂️ Table of Contents
- [Retriever](#retriever)
- [Resources](#resources)
- [Thesis Paper](#thesis-paper)

# Retriever

## Two Search Approaches

1. **Keyword Search**
Looks for documents containing the **exact words** found in the prompt
2. **Semantic Search**
Looks for documents with **similar meaning** to the prompt.

**Hybrid Search** = Keyword + Semantic + Metadata Filtering

*Metadata Filtering
Excludes documents based on rigid criteria

*High-performing retrievers balance all three techniques based on project needs.*

# Resources
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401?utm_source=chatgpt.com)
- [REIC: RAG-Enhanced Intent Classification at Scale](https://www.arxiv.org/abs/2506.00210?utm_source=chatgpt.com)
- [HiQA: A Hierarchical Contextual Augmentation RAG for Multi-Documents QA](https://arxiv.org/abs/2402.01767?utm_source=chatgpt.com)
- [Leveraging the Domain Adaptation of Retrieval Augmented Generation Models for Question Answering and Reducing Hallucination](https://arxiv.org/abs/2410.17783?utm_source=chatgpt.com)
- [ChatQA: Surpassing GPT-4 on Conversational QA and RAG](https://arxiv.org/abs/2401.10225?utm_source=chatgpt.com)
- [Evaluation of a Retrieval-Augmented Generation-Powered Chatbot for Pre-CT Informed Consent: a Prospective Comparative Study](https://link.springer.com/article/10.1007/s10278-025-01483-w?utm_source=chatgpt.com)
- [Trustworthy Retrieval-Augmented Generation: A Framework for Reliability, Privacy, Safety, Fairness, and Accountability](https://www.reddit.com/r/MachineLearning/comments/1ip7dvv/r_trustworthy_retrievalaugmented_generation_a/?utm_source=chatgpt.com)
- [RAG state-of-the-art](https://www.reddit.com/r/machinelearningnews/comments/1e4otdo/rag_stateoftheart/?utm_source=chatgpt.com)

## ✅ How to Use These Papers
1. Read foundational Lewis et al. (2020) to ground your method section.

2. Study REIC to inform building your own intent‑classifier model integrated into RAG.

3. Dive into domain‑adaptation & HiQA to plan how to prepare and structure your Japan visa knowledge base.

4. Reference ChatQA for advanced tuning strategies and benchmarks.

5. Use surveys and discussions to design evaluation metrics: how you'll measure accuracy, faithfulness, absence of hallucination, and trustworthiness.

# Thesis Paper

To generate the formatted thesis document, run the following command from the root of the project:

📁 **Location:** `docs/thesis_paper/`

### 🛠️ Generate Word Document (`.docx`)

This project supports modular `.md` files using `@include "path/to/file.md"` syntax.

#### 🔧 Step-by-Step

##### 1. Combine all Markdown files:
> use the Python script

```python
python combine_md.py  # Combines thesis.md into full_thesis.md
```

##### 2. Generate Word Document using `pandoc`:
```bash
pandoc full_thesis.md \
  --bibliography=references.bib \
  --csl=ieee.csl \
  --citeproc \
  -o thesis.docx
```