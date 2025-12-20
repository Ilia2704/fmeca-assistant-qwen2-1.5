# lawer-qwen2-1.5
The small llm with inference, trained to catch discrepancies in contracts (deployed on Ubuntu 24.04.3 LTS)
The project implements an AI-powered legal assistant based on **Qwen/Qwen2-1.5B-Instruct**.

The goal of the system is to **analyze incoming contracts and agreements for compliance with applicable laws and regulations**.  
Legal norms, articles, and interpretations are stored in a **Qdrant vector database** and used as an external knowledge base.

The bot operates in a Retrieval-Augmented Generation (RAG) setup:
- incoming contracts are parsed and segmented,
- relevant legal provisions are retrieved from Qdrant via semantic search,
- the language model analyzes the contract against the retrieved laws,
- the output highlights potential inconsistencies, risks, or non-compliant clauses.

The project is designed to be:
- modular (CLI, API, and web UI),
- deployable on a remote instance,
- extensible for evaluation, fine-tuning, and domain adaptation.

## 1. Environment setup

Update the packages: 
```bash 
sudo apt update && sudo apt install -y git python3.12 python3.12-venv python3.12-dev
```

Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade core Python tooling:
```bash 
python -m pip install -U pip wheel setuptools
```

Install project dependencies:
```bash 
pip install -r requirements.txt
```

## 2. Project structure

```text 
lawer-qwen2-1.5/
├── app.py               # Streamlit web UI

├── requirements.txt
├── README.md
├── .venv/
```
## 3. CLI dialog mode

Run the model in interactive terminal mode:
```bash 
python cli_chat.py
```
Exit with:
```bash
exit | quit | stop
```

## 4. Streamlit web interface

Run locally on the server:
```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Access from your local machine via SSH tunnel:
```bash
ssh -L 8501:127.0.0.1:8501 yc-ml
```

Open in browser:
```bash
http://localhost:8501
```