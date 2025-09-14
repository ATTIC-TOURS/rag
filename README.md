# 📚 RAG-based Chatbot

Build Chatbot API prototype for supporting Filipino to obtain Japan Visa

# 🗂️ Table of Contents
- [Design](#design)
- [Dataset](#dataset)
- [Thesis Paper](#thesis-paper)

# Design

https://drive.google.com/file/d/1Jiu3KcCtv7GkDBc-pGRaDUExTsG--FEx/view?usp=sharing

# Dataset
https://docs.google.com/spreadsheets/d/1fQG8NFptlZ7sklOIuE0isuo_ZBvUbnBC/edit?usp=sharing&ouid=103300278601878530833&rtpof=true&sd=true


# Thesis Paper

To generate the formatted thesis document, run the following command from the location below:

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