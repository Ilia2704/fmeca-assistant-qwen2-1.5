# fmeca-assistant-qwen2-1.5
The small llm with inference, trained to propose cause and effects of the failure (deployed on Ubuntu 24.04.3 LTS)
The project implements an AI-powered legal assistant based on **Qwen/Qwen2-1.5B-Instruct**.

The goal of the system is to support Failure Modes, Effects, and Criticality Analysis (FMECA) by assisting engineers in the identification, structuring, and assessment of potential failure scenarios in complex systems.
Domain knowledge such as failure modes, effects, causes, detection methods, and severity/occurrence considerations is stored in a vector-based knowledge base and used as external contextual grounding.
The assistant operates in a Retrieval-Augmented Generation (RAG) setup:
- system descriptions or components are parsed and structured,
- relevant FMECA-related knowledge (failure patterns, typical effects, mitigation strategies) is retrieved via semantic search,
- the language model performs context-aware reasoning over the retrieved information,
- the output highlights potential failure modes, associated effects, risk drivers, and areas requiring expert attention.
The system is intended as a decision-support tool for reliability and safety analysis, helping engineers reason more systematically while keeping final judgments under human control.

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
python3 -m pip install -U pip wheel setuptools
```

Install project dependencies:
```bash 
pip install -r requirements.txt
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
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
python run_local.py
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