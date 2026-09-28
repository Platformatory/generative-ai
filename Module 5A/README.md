# Module 5A: Building AI Agents on Databricks

## Overview

This module teaches the full lifecycle of building, deploying, securing, and evaluating AI agents on Databricks — from individual tools to multi-agent orchestration to production deployment.

## Demos

| Demo | Topic | Key Concepts |
|---|---|---|
| Demo 1 | AI Playground & Model Comparison | Prompt engineering, model selection, temperature/token parameters |
| Demo 2 | Comparing Models and Prompts | Side-by-side model evaluation, system prompts, few-shot examples |
| Demo 3 | PDF-to-Searchable-Index Pipeline | RAG fundamentals, `ai_parse_document`, Vector Search, embeddings, L2 distance, retrieval gating |
| Demo 4 | Building Agent Tools on Databricks | SQL/Python UC functions, `@function_tool` wrappers, tool discovery, AI Gateway concepts |
| Demo 5 | Building Agents | OpenAI Agents SDK, `Agent` objects, `Runner.run()`, single-agent with tools |
| Demo 6 | Multi-Agent Orchestration | Supervisor-worker pattern, agent handoffs|
| Demo 7 | AI Platform Map, Security & Exam Review | OWASP Top 10 for LLM, prompt injection defense, LLM-as-judge, RAG agents, AI Gateway features |

## What Else Can You Do with Module 5A

### 1. RAG Agents (Retrieval-Augmented Generation)
RAG doesn't require PDFs. Any structured data in a UC table with a Vector Search index can serve as the knowledge base.

**If data is already structured** (UC table, Delta table):
* Skip `ai_parse_document` (PDF parsing)
* Create a Vector Search index directly on the table
* Build a RAG Agent that uses Vector Search as a `@function_tool`

**If data is unstructured** (PDFs, images):
* Use `ai_parse_document(content, MAP('version', '2.0'))` to parse into structured VARIANT
* Extract text elements and insert into a UC table
* Create a Vector Search index on the table
* Build the RAG Agent the same way

**Deployment**: Wrap the RAG Agent in an MLflow `PythonModel`, log to MLflow, register in Unity Catalog, and deploy via `databricks.agents.deploy()

### 2. AI Gateway Features

After deploying an agent, enable AI Gateway features to monitor and secure endpoints:

| Feature | Description |
|---|---|
| **Usage tracking** | Enables data usage metrics for the endpoint. Tracks token consumption and cost. |
| **Inference tables and telemetry** | Logs all requests and responses into Delta tables managed by Unity Catalog via OpenTelemetry. Logs, spans, and metrics are captured into separate telemetry tables. |
| **Rate limits** | Enforces request rate limits to manage traffic. Set requests per minute and tokens per minute caps. |

**How to enable**: Navigate to the deployed endpoint in AI/ML > Agents tab > Edit Settings > AI Gateway.

### 3. Multi-Agent Orchestration
* **Supervisor-Worker pattern**: A supervisor agent routes questions to specialist agents (billing, shipping, technical) via handoffs
* **Composable tools**: Agents can share UC functions, Vector Search, and other tools
* **Deployment**: Wrap the entire multi-agent system in a single `PythonModel` and deploy as one endpoint

## Getting Started

1. Run demos in order (Demo 1 through Demo 7)
2. Each demo creates its own UC catalog/schema — no dependencies between demos
3. Cleanup cells at the end of each notebook remove all created resources
4. Demos 3-7 use the same model: `databricks-meta-llama-3-3-70b-instruct`
5. Demos 4-7 use the OpenAI Agents SDK with Databricks as the LLM provider