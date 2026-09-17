<!-- Animated Header -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=26&pause=1000&color=00F7FF&center=true&vCenter=true&width=900&lines=Data+Science;AI+%26+ML+Engineer;LLMs+%7C+RAG+Systems+%7C+Reinforcement+Learning;Research+%26+Production-Grade+AI+Systems;Turning+Theory+into+Deployed+Intelligence" />
</p>

---

## About Me

I'm an **AI/ML Engineer** specializing in **end-to-end Agentic RAG pipelines**, **multilingual conversational systems**, and **production ML infrastructure**. My work translates complex problems into clean, deployable solutions — from building retrieval systems for enterprise platforms to optimizing inference across low-resource languages.

Currently at **Astrik** (Aug 2026-Present), developing multilingual RAG and Agentic RAG systems for ERP and CRM platforms. Previously architected production RAG pipelines at Tesseract Innovations and optimized LLM, TTS, and STT models for low-resource languages at Karsaaz EBS.

---

## What I'm Currently Working On

Building multilingual RAG and Agentic RAG systems for enterprise platforms using **LangGraph** and **LangChain**, enabling intelligent retrieval, reasoning, and task automation across business workflows.

Focused on:
- Agentic reasoning and deterministic routing across multiple data sources
- Cross-lingual normalization and multilingual retrieval optimization
- Production inference scaling with Groq and local LLM backends
- Vector search tuning (ChromaDB, Qdrant) with cross-encoder reranking

---

## Research & Technical Interests

Deep Reinforcement Learning & policy optimization  
Retrieval-Augmented Generation (RAG) & vector search optimization  
Efficient LLM fine-tuning and inference scaling  
Multilingual NLP and speech systems (TTS/STT)  
Autonomous agents & decision-making systems  
Production ML architecture & deployment

---

## Featured Projects

### NADRA Multilingual RAG Assistant

[View Repository](https://github.com/AbrarAqeel/NADRA_RAG_Chatbot)

**Bilingual (English/Urdu) conversational RAG system for citizen services with a live, lip-synced avatar.**

This was my final year project and a technical exploration of production RAG complexity. The system retrieves answers strictly from a curated NADRA knowledge base and speaks responses through a Simli avatar over WebRTC.

**Why it stands out:**

- **Bilingual RAG at scale**: Separate ChromaDB collections per language (`nadra_en`, `nadra_ur`) with automatic language detection from both text and speech. Cross-lingual query normalization layer converts Hindi/Arabic/Farsi/Punjabi to Urdu before retrieval.
- **Avatar integration**: Real-time TTS (gTTS) → 16kHz PCM audio → Simli WebRTC. Graceful fallback if Simli is unreachable — keeps text + audio playback working.
- **Reranking architecture**: Top-15 ChromaDB candidates → cross-encoder (`ms-marco-MiniLM-L-6-v2`) rerank → keep top-5 for LLM. This two-stage approach catches false positives that lexical search alone misses.
- **Production deployment**: Docker image with CUDA base for Hugging Face Spaces. Runs locally on CPU in ~2s per query.
- **Conversation memory**: Last 10 turns inform both query enhancement and answer generation, supporting genuine multi-turn follow-ups.

**Tech:** FastAPI | Groq (Llama 3.3) | faster-whisper | ChromaDB + Cross-Encoder Reranking | React + Vite | Simli WebRTC | Docker

---

### AI Support Desk with Knowledge Routing Agent

[View Repository](https://github.com/AbrarAqeel/AI_SupportDesk_with_KnowledgeRoutingAgent)

**Deterministic multi-source agent that routes queries to PostgreSQL, vector knowledge base, or external APIs based on intent classification.**

This project was a deep dive into **when to use agents vs. simple RAG**. Instead of throwing everything at an LLM, it uses explicit rule-based routing to ensure data authority — the right query hits the right source.

**Why it stands out:**

- **Deterministic routing over semantic search**: A `RouterNode` classifies intent based on signal word priority (e.g., "ticket ID" → PostgreSQL, "how to" → vector DB, "weather" → external API). No hallucinations about which source to check first.
- **Multi-source coordination**: Single LangGraph that orchestrates PostgreSQL queries, ChromaDB retrieval, and fallback API calls in a single turn. Conversation context flows through all three branches.
- **Conversation memory strategy**: Sliding window of last 10 messages — cheap to maintain, enough for follow-ups ("tell me more about that ticket").
- **Testing-first architecture**: `/testing` folder has phase-by-phase unit tests. You can verify the router logic independently of the graph, and the graph independently of the UI.
- **Clean separation of concerns**: `/router` handles classification logic; `/tools` wraps PostgreSQL/ChromaDB/API calls; `/graph` wires everything together. Easy to swap out any layer.

**Tech:** LangGraph + LangChain | FastAPI | Streamlit | PostgreSQL | ChromaDB | Rule-based classification

---

### Voice AI Patient Intake

[View Repository](https://github.com/AbrarAqeel/VoiceAIAgent)

**Production-deployed voice-based registration system: caller dials a Vapi number, speaks naturally with an AI intake coordinator (Maya), and data persists to a REST API.**

This project showcases **voice AI integration done right** — working with Vapi's tool-calling protocol, proper error handling, and live deployment.

**Why it stands out:**

- **Vapi protocol correctness**: The Vapi webhook expects `{ "message": { "type": "tool-calls", ... } }` and responses must be `{ "results": [...] }` — documentation is sparse, but the code gets this right and documents the quirk for future developers.
- **Live deployment**: Running on Railway with a real US phone number (+1 434-290-7724). No ngrok, no local machine requirement. FastAPI health checks and graceful error handling.
- **Tool-based CRUD**: Vapi tools (`create_patient`, `find_patient_by_phone`, `update_patient`) map to FastAPI endpoints. Duplicate detection is smart — same phone number can update an existing record or reject if already active (prevents double-booking).
- **Soft-delete semantics**: Patients can be soft-deleted without losing historical data; phone numbers become reusable after deletion.
- **Validation on the server side**: Names, DOB (not in future), US phone format, state, ZIP, enum-based sex field, email. The API is defensive.

**Tech:** Vapi (voice + LLM + STT/TTS) | FastAPI | SQLite | Railway | REST API

---

### Purse Visual Retrieval System

[View Repository](https://github.com/AbrarAqeel/Purse_Retrieval)

**Two-stage computer vision pipeline: detect purses in photos, embed them with DINOv2, retrieve similar purses from a FAISS index.**

This is a **real-world search problem** — how do you match a queried purse to models in your dataset without training a custom classifier?

**Why it stands out:**

- **Model choice reasoning**: YOLOv8 for detection (COCO "handbag" class is reliable), but crucially **DINOv2 over CLIP** for embeddings. DINOv2 is trained for instance-level similarity ("is this the *same* object"), while CLIP optimizes for semantic similarity ("same *kind* of object") — unrelated bags with similar color would score too close in CLIP.
- **Incremental indexing**: `build_index.py` is a one-time operation. Photos → detection → in-memory crop (never saved) → embedding → FAISS. If the index exists, only new photos are processed. `streamlit run main.py` only loads and searches — keeps the app snappy.
- **Tunable precision-recall**: Two knobs in `config.py` — `SIMILARITY_THRESHOLD` (hard floor) and `MATCH_MARGIN` (relative spread filter). Results too strict? Adjust and rerun. Results too loose? Tighten. The code even provides a shell helper to see raw similarity scores before filtering.
- **Modular architecture**: `/core/detection.py`, `/core/embedding.py`, `/core/indexer.py`, `/core/search.py` — each responsibility isolated. Easy to swap DINOv2 for a fine-tuned model later.

**Tech:** YOLOv8 | DINOv2 | FAISS | Streamlit | Incremental indexing

---

## Open to Collaborate On

AI/ML research projects (NLP, RL, LLMs)  
Open-source AI tools and frameworks  
Agentic systems and autonomous decision-making  
Startup-focused AI products & MVPs  
Multilingual AI systems & low-resource language optimization

---

## Currently Learning

Advanced Deep Reinforcement Learning techniques  
Efficient LLM fine-tuning & retrieval optimization  
Production AI system design at scale

---

## Ask Me About

Agentic RAG systems and LangGraph workflow design  
Multilingual NLP and low-resource language optimization  
Production RAG architectures (retrieval, reranking, fusion)  
Cross-encoder reranking and vector database tuning  
Speech systems (TTS/STT fine-tuning and deployment)  
Reinforcement Learning (SARSA, Q-Learning, DQN)  
FastAPI-based ML microservices and scalable architecture  
On-device ML and TFLite quantization

---

## Fun Fact

I enjoy turning research papers into production systems — from multilingual RAG assistants to autonomous agents.

---

## Socials

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/abrar-aqeel-679194158)

---

## Tech Stack

### Languages & Frameworks

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)

### Machine Learning & AI

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-%23FFCC00.svg?style=for-the-badge&logo=huggingface&logoColor=black)

### RAG & Vector Search

![LangChain](https://img.shields.io/badge/LangChain-%2300A4EF.svg?style=for-the-badge&logo=chainlink&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-%23013243.svg?style=for-the-badge&logo=meta&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-%23000000.svg?style=for-the-badge)
![Qdrant](https://img.shields.io/badge/Qdrant-%23FF6B6B.svg?style=for-the-badge)

### Computer Vision & Speech

![OpenCV](https://img.shields.io/badge/OpenCV-%235C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-%23013243.svg?style=for-the-badge)

### Data Science & Tooling

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=matplotlib&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

### MLOps & DevOps

![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-%2300758F.svg?style=for-the-badge&logo=mysql&logoColor=white)

---

### Random Dev Quote

![Quote](https://quotes-github-readme.vercel.app/api?type=horizontal&theme=radical)

---
