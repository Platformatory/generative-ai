# Databricks notebook source
# MAGIC %md
# MAGIC Module 5A : Detailed Topic Index
# MAGIC Databricks SQL + Genie (1 hr) + Gen AI Foundations (5 hrs)
# MAGIC
# MAGIC Day 1 (3 hrs)
# MAGIC Hour 1 : Databricks SQL + Genie (Module 5A : native)
# MAGIC Source: [Introduction to AI/BI Dashboards]
# MAGIC
# MAGIC 1.1 The Databricks AI/BI family : Dashboards, Genie Agents, Genie Code, Genie One, and how they fit together 
# MAGIC 1.2 AI/BI Dashboard building blocks : datasets, canvas, visualizations, filters, pages 
# MAGIC 1.3 Dashboard dataset sources : UC tables (in scope), UC metric views, dashboard-local metric views
# MAGIC 1.4 Genie Code : AI-assisted authoring: dataset generation from NL prompt, auto-configured visualizations, full-dashboard planning, inline quick-fixes 
# MAGIC 1.5 Dashboard vs. Genie Agent : decision framework: known/repeated question → Dashboard; unknown/ad-hoc question → Genie 
# MAGIC 1.6 Three ingredients of a well-scoped dashboard : purpose, audience, data + do/don't best practices
# MAGIC
# MAGIC Demo: "Dataset-to-Genie Walkthrough"
# MAGIC  Source: [AI/BI for Data Analysts]
# MAGIC Demo: Designing Datasets for Dashboards
# MAGIC Demo: Page 1 Visualizations - KPIs and Yearly Trends
# MAGIC Demo: Genie Code for Dashboards - Building Page 2
# MAGIC
# MAGIC
# MAGIC Hour 2 : Generative AI Fundamentals & the Databricks AI Platform
# MAGIC Source: [Introduction to Generative AI], [Finding Success with Generative AI]
# MAGIC
# MAGIC 2.1 Generative AI, defined : nested field (AI → ML → Deep Learning → Gen AI); predictive vs. generative AI 
# MAGIC 2.2 What Gen AI creates : text, code, image/audio/video, synthetic data, structured (schema-valid) outputs 
# MAGIC 2.3 Foundation models vs. LLMs vs. Gen AI : the nesting relationship; proprietary (GPT, Claude, Gemini) vs. open-weight (Llama, Gemma, Qwen) examples 
# MAGIC 2.4 How an LLM turns a prompt into output : tokenize → embed → Transformer (context) → predict, one token at a time 
# MAGIC 2.5 Proprietary API vs. open-weight models : strengths/trade-offs on privacy, cost, control 
# MAGIC 2.6 Choosing a model : four criteria: privacy, quality, cost, latency 
# MAGIC 2.7 Why success is a data problem, not a model problem : data-native architecture vs. exporting data (egress cost, latency, fragmented governance) 
# MAGIC 2.8 AI Functions : lowest-code Gen AI entry point: ai_classify, ai_extract, ai_summarize, ai_translate, ai_mask, ai_parse_document, ai_query 
# MAGIC 2.9 Compound AI systems : model + retrieval + tools + classical ML + memory + guardrails, all governed by Unity Catalog 
# MAGIC 2.10 Databricks AI platform : five workflow stages: Access a Model → Build → Prepare & Serve Data → Deploy → Govern & Monitor
# MAGIC
# MAGIC Demo: “Comparing Models and Prompts in the AI Playground”
# MAGIC  Source:  [Generative AI Fundamentals]
# MAGIC
# MAGIC
# MAGIC Hour 3 : RAG Fundamentals & Databricks AI Search
# MAGIC Source: [Introduction to Generative AI], [Agent Bricks for Retrieval and Context Engineering], [Document Parsing and Chunking Strategies], [AI Search on Databricks]
# MAGIC
# MAGIC 3.1 Context engineering vs. prompt engineering : two ways to adapt a model (change what it sees vs. change its weights); why context engineering is the primary discipline 
# MAGIC 3.2 RAG pattern : indexing phase (documents → chunk → embed → vector DB) vs. query phase (question → embed → similarity search → grounded answer) 
# MAGIC 3.3 Why retrieval agents exist : knowledge cutoffs, hallucination, missing private context 
# MAGIC 3.4 Document processing pipeline : ai_parse_document (PDF → structured JSON) → ai_classify / ai_extract (categorize/structure) → ai_prep_search (chunk) 
# MAGIC 3.5 Chunking strategy considerations : chunk size vs. embedding model limits, overlap (10–20%), granularity trade-off, "lost in the middle" phenomenon 
# MAGIC 3.6 chunk_to_embed vs. chunk_to_retrieve : context-enriched text for findability vs. clean raw text for the LLM 
# MAGIC 3.7 Embeddings & cosine similarity : how vectors capture meaning; how AI Search ranks results (ANN) 
# MAGIC 3.8 Creating/syncing an AI Search index : key parameters (endpoint_name, source_table_name, embedding_model_endpoint_name, pipeline_type); TRIGGERED vs. CONTINUOUS sync; Change Data Feed 
# MAGIC 3.9 Managed RAG (Knowledge Assistant) vs. Custom RAG (Agent Framework + AI Search) : setup time, customizability, when to use each 
# MAGIC 3.10 Unity Catalog governance across the retrieval pipeline : UC privileges on volumes/tables/indexes, endpoint ACLs
# MAGIC
# MAGIC Demo: "PDF-to-Searchable-Index Pipeline"
# MAGIC  Source: [Building RAG Agents with Agent Bricks]
# MAGIC Demo: Transforming PDFs to Structured Data
# MAGIC Demo: Chunking PDFs and Vector Search
# MAGIC
# MAGIC
# MAGIC Day 2 (3 hrs)
# MAGIC Hour 1 : Agentic AI 101
# MAGIC Source: [Introduction to Generative AI], [Building Agentic Applications on Databricks]
# MAGIC
# MAGIC 4.1 From RAG to agents : classic RAG's fixed retrieve-then-generate step vs. an agent's reason → act → observe loop 
# MAGIC 4.2 What is an AI agent : perceives, reasons, plans, adapts, acts, learns; contrast with traditional single-call systems 
# MAGIC 4.3 Agent framework anatomy : system prompt + user prompt → tool calls → response (sequence of decisions) 
# MAGIC 4.4 Unity Catalog functions as governed agent tools : write a Python/SQL function, register in UC, agent discovers and calls it 
# MAGIC 4.5 Common tool patterns : structured data retrieval, unstructured retrieval (RAG), code interpreter, external connections, AI Playground prototyping 
# MAGIC 4.6 AI Playground : quick manual testing of prompts/models/UC tools before wiring code 
# MAGIC 4.7 Unity AI Gateway : guardrails, rate limits, usage tracking, traffic splitting/fallback routing across models
# MAGIC
# MAGIC Demo: “Building Agent Tools on Databricks” 
# MAGIC  Source: [Building Agentic Applications on Databricks]
# MAGIC
# MAGIC
# MAGIC Hour 2 : Multi-Agent Systems, Agent Bricks & Genie Integration
# MAGIC Source: [Building Agents with the OpenAI Agents SDK], [Agent Bricks and Genie]
# MAGIC
# MAGIC 5.1 Single-agent vs. multi-agent : decision factors: tool count (~8–10 threshold), domain complexity, instruction length, failure isolation, team ownership, latency 
# MAGIC 5.2 Supervisor-worker (router-delegate) pattern : supervisor routes/delegates to specialized workers (Data Agent, Research Agent, General Agent), synthesizes response 
# MAGIC 5.3 OpenAI Agents SDK orchestration patterns : LLM-driven (model decides) vs. code-driven (deterministic flow); handoffs 
# MAGIC 5.4 @function_tool decorator : turning a Python function into an agent tool for rapid prototyping (name/type hints/docstring → tool schema) 
# MAGIC 5.5 Agent Bricks : no-code platform for building/optimizing/governing production agents; supported types: Knowledge Assistant, Supervisor Agent, Document Intelligence, Custom Agents 
# MAGIC 5.6 Agent Bricks development lifecycle : specify → configure/evaluate on your data → continuously improve (built on UC, Model Serving, and Agent Evaluation) 
# MAGIC 5.7 Genie Space vs. Genie Agent : curated NL interface to governed data; used standalone or as a specialized worker in a multi-agent system 
# MAGIC 5.8 Integrating Genie with agents : as a subagent under a supervisor 
# MAGIC 5.9 Code-first (OpenAI SDK, LangChain, LangGraph, DSPy) vs. no-code (Agent Bricks) : when to reach for each 
# MAGIC
# MAGIC  Demo: “Building Single Agents with the OpenAI Agents SDK”
# MAGIC   Source: [Building Agentic Applications on Databricks]
# MAGIC  Demo: “Multi-Agent Orchestration with the OpenAI Agents SDK”
# MAGIC   Source:  [Building Agentic Applications on Databricks]
# MAGIC
# MAGIC
# MAGIC Hour 3 : Databricks AI Platform Map, Security & Exam-Aligned Review
# MAGIC Source: [Finding Success with Generative AI]
# MAGIC
# MAGIC 6.1 Full platform recap : AI Playground vs. Genie vs. Agent Framework vs. Agent Bricks vs. Model Serving: when you'd reach for each 
# MAGIC 6.2 A practical AI adoption roadmap : foundation-first (governed data) → build & deploy (governed) → operating model → roles & upskilling 
# MAGIC 6.3 OWASP Top 10 for LLM Applications (concept-level) : data leakage (LLM02), prompt injection (LLM01), excessive agency (LLM06), data/model poisoning (LLM04), vector/embedding weaknesses (LLM08), system prompt leakage (LLM07) 
# MAGIC 6.4 Defense in depth on Databricks : Govern (Unity Catalog) → Guard (Unity Gateway) → Monitor (inference tables) → Frame (Databricks AI Security Framework) 
# MAGIC 6.5 LLM-as-judge teaser : LLM-as-judge evaluation, scored dimensions (correctness, relevance, groundedness, safety);
# MAGIC 6.6 Self-check against the 6 official Gen AI Associate exam domains 
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

