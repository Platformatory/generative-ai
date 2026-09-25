# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Demo Title
# MAGIC %md
# MAGIC # Demo 2 : Comparing Models and Prompts in the AI Playground
# MAGIC
# MAGIC **Module 5A - Hour 2: Generative AI Fundamentals & the Databricks AI Platform (Topics 2.1-2.10)**
# MAGIC
# MAGIC This notebook demonstrates Gen AI fundamentals using **Databricks SQL AI Functions** - the lowest-code entry point to generative AI on the platform.
# MAGIC
# MAGIC Each section follows this pattern:
# MAGIC 1. **Notes** (markdown) - concept explanation, the function signature, and *why* we're running this demo
# MAGIC 2. **Demo** (SQL) - live code on sample data with inline comments
# MAGIC
# MAGIC At the end you'll find a **Learning Conclusion** and a **Cleanup** cell that decommissions everything created during the demo.

# COMMAND ----------

# DBTITLE 1,Setup - Sample Data
# MAGIC %sql
# MAGIC -- ═══════════════════════════════════════════════════════════════
# MAGIC -- SETUP: Create catalog, schema, and sample data for all Gen AI demos
# MAGIC -- ═══════════════════════════════════════════════════════════════
# MAGIC -- We create a table of customer support tickets that we'll reuse
# MAGIC -- throughout the demo to showcase AI Functions on realistic data.
# MAGIC -- We create a dedicated catalog and schema for this demo, then a table
# MAGIC -- of customer support tickets that we'll reuse throughout the demo.
# MAGIC -- Using a dedicated catalog/schema makes cleanup easy and avoids
# MAGIC -- cluttering the default schema.
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS module5a_demo2;
# MAGIC CREATE SCHEMA IF NOT EXISTS module5a_demo2.genai;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE module5a_demo2.genai.support_tickets (
# MAGIC   ticket_id   INT,
# MAGIC   customer    STRING,
# MAGIC   message     STRING,
# MAGIC   created_at  TIMESTAMP
# MAGIC );
# MAGIC
# MAGIC INSERT INTO module5a_demo2.genai.support_tickets VALUES
# MAGIC (1,  'John Doe',      'I cannot log into my account, it keeps saying invalid credentials. Please help!',                       '2024-10-01 09:15:00'),
# MAGIC (2,  'Jane Smith',     'I was charged twice for my monthly subscription of $29.99. I need a refund immediately.',            '2024-10-01 10:30:00'),
# MAGIC (3,  'Bob Johnson',    'The app crashes every time I try to upload a photo. This is very frustrating!',                    '2024-10-01 11:45:00'),
# MAGIC (4,  'Alice Brown',    'I love the new dark mode feature! Great job team, keep it up.',                                     '2024-10-01 14:00:00'),
# MAGIC (5,  'Charlie Wilson', 'Can you add support for exporting data to Excel? That would be really helpful for my team.',    '2024-10-01 15:20:00'),
# MAGIC (6,  'Diana Prince',   'My order #12345 arrived damaged. The box was crushed and the product inside is broken.',        '2024-10-02 08:10:00'),
# MAGIC (7,  'Ethan Hunt',     'The new update is terrible. Everything is slower and the UI is confusing. I want the old version.','2024-10-02 09:30:00'),
# MAGIC (8,  'Fiona Green',   'I need to update my payment method from Visa ending 4521 to Mastercard ending 8830. Email: fiona@email.com', '2024-10-02 10:45:00'),
# MAGIC (9,  'George King',    '¿Cómo puedo cancelar mi suscripción? No encuentro la opción en la configuración.',                  '2024-10-02 12:00:00'),
# MAGIC (10, 'Hannah Lee',     'The API documentation is outdated. The endpoint /v2/users no longer exists and returns 404.',     '2024-10-02 13:15:00');
# MAGIC
# MAGIC SELECT * FROM module5a_demo2.genai.support_tickets ORDER BY ticket_id;

# COMMAND ----------

# DBTITLE 1,2.1-2.3 Notes - Gen AI Fundamentals
# MAGIC %md
# MAGIC ## 2.1-2.3 : Generative AI, Defined - What Gen AI Creates - Foundation Models vs LLMs
# MAGIC
# MAGIC ### Concepts
# MAGIC * **Generative AI** sits inside a nested field: AI → Machine Learning → Deep Learning → Generative AI.
# MAGIC   * *Predictive AI* forecasts outcomes (fraud detection, churn prediction).
# MAGIC   * *Generative AI* creates **new content**: text, code, image/audio/video, synthetic data, and structured (schema-valid) outputs.
# MAGIC * **Foundation models vs LLMs vs Gen AI** (nesting relationship):
# MAGIC   * Foundation models = large pre-trained models adaptable to many tasks.
# MAGIC   * LLMs = foundation models specialised for language.
# MAGIC   * Gen AI = the broadest category - includes image generators, music generators, code generators, etc.
# MAGIC * **Proprietary** (GPT, Claude, Gemini) vs **open-weight** (Llama, Gemma, Qwen): trade-offs in privacy, cost, and control.
# MAGIC
# MAGIC `ai_gen(prompt)` is the **simplest Gen AI entry point** - it accepts a plain-text prompt and returns generated text. No endpoint name needed; Databricks picks the default model automatically. We use it to **generate a support-ticket taxonomy from scratch**, showing that Gen AI creates *structured text*, not just prose.

# COMMAND ----------

# DBTITLE 1,2.1–2.3 Demo — ai_gen
# MAGIC %sql
# MAGIC -- 2.1-2.3 Demo: Text generation with ai_gen()
# MAGIC -- ai_gen(prompt) → STRING  — sends a prompt to the default foundation model.
# MAGIC -- No endpoint name needed — Databricks selects the model for you.
# MAGIC
# MAGIC SELECT ai_gen(
# MAGIC   'List exactly 5 short labels for categorising customer support tickets. '
# MAGIC   || 'Return only the labels, one per line, no numbering, no extra text.'
# MAGIC ) AS generated_taxonomy;

# COMMAND ----------

# DBTITLE 1,2.4 Notes - LLM Mechanics
# MAGIC %md
# MAGIC ## 2.4 : How an LLM Turns a Prompt Into Output
# MAGIC
# MAGIC ### Concept
# MAGIC An LLM processes a prompt in four stages:
# MAGIC 1. **Tokenize** - split text into sub-word tokens (the model's vocabulary).
# MAGIC 2. **Embed** - convert each token into a vector that captures semantic meaning.
# MAGIC 3. **Transformer (context)** - the self-attention mechanism lets every token “see” every other token and weigh its relevance.
# MAGIC 4. **Predict** - output one token at a time; each prediction is fed back as input for the next step (autoregressive decoding).
# MAGIC
# MAGIC `ai_query(endpoint, request, modelParameters => …)` lets us call a **specific** foundation model by name and control parameters such as `temperature` and `max_tokens`. By varying `temperature` (0 = deterministic, 1+ = creative) on the same prompt, we see how the probabilistic, autoregressive nature of token prediction produces different outputs.
# MAGIC
# MAGIC > **Note:** If you get a 403 error mentioning Unity Gateway, replace `databricks-<model>` with `system.ai.<model>` in the endpoint name.

# COMMAND ----------

# DBTITLE 1,2.4 Demo — ai_query with Model Parameters
# MAGIC %sql
# MAGIC -- 2.4 Demo: Model parameters with ai_query()
# MAGIC -- ai_query(endpoint, request, modelParameters => named_struct(...)) lets us control:
# MAGIC --   temperature : 0.0 = deterministic, 1.0+ = creative
# MAGIC --   max_tokens  : caps the output length
# MAGIC -- We send the SAME prompt at two temperature settings to observe the difference.
# MAGIC
# MAGIC SELECT ai_query(
# MAGIC   'databricks-meta-llama-3-3-70b-instruct',
# MAGIC   'In one sentence, explain what Apache Spark is.',
# MAGIC   modelParameters => named_struct('temperature', 0.0, 'max_tokens', 50)
# MAGIC ) AS response_temp_0;
# MAGIC
# MAGIC SELECT ai_query(
# MAGIC   'databricks-meta-llama-3-3-70b-instruct',
# MAGIC   'In one sentence, explain what Apache Spark is.',
# MAGIC   modelParameters => named_struct('temperature', 0.9, 'max_tokens', 50)
# MAGIC ) AS response_temp_0_9;

# COMMAND ----------

# DBTITLE 1,2.5-2.6 Notes - Model Selection
# MAGIC %md
# MAGIC ## 2.5-2.6 : Proprietary API vs Open-Weight Models - Choosing a Model
# MAGIC
# MAGIC ### Concepts
# MAGIC * **Proprietary APIs** (GPT, Claude, Gemini): vendor hosts the model; you call via API.
# MAGIC   * Strengths: state-of-the-art quality, no infrastructure to manage.
# MAGIC   * Trade-offs: data leaves your environment (egress cost, latency, governance gaps).
# MAGIC * **Open-weight models** (Llama, Gemma, Qwen): weights are downloadable; you host the model.
# MAGIC   * Strengths: full control, data stays in your environment, no per-token API cost.
# MAGIC   * Trade-offs: you manage infrastructure, quality may lag behind proprietary SOTA.
# MAGIC * **Four selection criteria**: **Privacy**, **Quality**, **Cost**, **Latency** - weigh these per use case.
# MAGIC
# MAGIC We send the same prompt to two different models using `ai_query()` and compare the outputs side-by-side. On Databricks, both models are served through the **same unified API** - no separate SDKs or credentials to manage. This demonstrates the *quality* dimension of model selection.
# MAGIC
# MAGIC > **Note:** If a model name returns an error, run `SHOW SERVING ENDPOINTS` to see which foundation model endpoints are available in your workspace.

# COMMAND ----------

# DBTITLE 1,2.5–2.6 Demo — Model Comparison
# MAGIC %sql
# MAGIC -- 2.5-2.6 Demo: Compare two models on the same prompt
# MAGIC -- Model 1: Meta Llama 3.3 70B (open-weight, larger)
# MAGIC -- Model 2: Llama 4 Maverick (open-weight, newer architecture)
# MAGIC -- Same prompt, same temperature — compare quality and conciseness.
# MAGIC
# MAGIC SELECT ai_query(
# MAGIC   'databricks-meta-llama-3-3-70b-instruct',
# MAGIC   'Classify this customer message into one word: billing, bug, feedback, shipping, account, or feature_request. Message: "I was charged twice for my subscription."',
# MAGIC   modelParameters => named_struct('temperature', 0.0, 'max_tokens', 10)
# MAGIC ) AS llama_70b_response;
# MAGIC
# MAGIC SELECT ai_query(
# MAGIC   'databricks-llama-4-maverick',
# MAGIC   'Classify this customer message into one word: billing, bug, feedback, shipping, account, or feature_request. Message: "I was charged twice for my subscription."',
# MAGIC   modelParameters => named_struct('temperature', 0.0, 'max_tokens', 50)
# MAGIC ) AS llama_4_maverick_response;

# COMMAND ----------

# DBTITLE 1,2.7 Notes - Data, Not Model
# MAGIC %md
# MAGIC ## 2.7 : Why Success Is a Data Problem, Not a Model Problem
# MAGIC
# MAGIC ### Concept
# MAGIC * The model is increasingly a **commodity** - the differentiator is **your data**.
# MAGIC * **Data-native architecture**: keep data in your lakehouse, apply AI functions *in-place*.
# MAGIC   * No egress costs (exporting data to external APIs).
# MAGIC   * No latency from network round-trips.
# MAGIC   * No fragmented governance - everything stays under Unity Catalog.
# MAGIC * **Anti-pattern**: exporting data to an external LLM API, processing it, then re-importing results.
# MAGIC
# MAGIC We run `ai_analyze_sentiment()` **directly on our governed table** - `module5a_demo2.genai.support_tickets`. The data never leaves Databricks. Unity Catalog controls who can read this table; the AI function is just another SQL operation on it. This is the data-native architecture in action.

# COMMAND ----------

# DBTITLE 1,2.7 Demo - AI on Governed Data
# MAGIC %sql
# MAGIC -- 2.7 Demo: AI functions operate on governed data in-place
# MAGIC -- No data export, no external API calls — just SQL on a UC-governed table.
# MAGIC -- Unity Catalog controls who can read this table; the AI function respects those privileges.
# MAGIC
# MAGIC SELECT 
# MAGIC   ticket_id,
# MAGIC   customer,
# MAGIC   ai_analyze_sentiment(message) AS sentiment
# MAGIC FROM module5a_demo2.genai.support_tickets
# MAGIC ORDER BY ticket_id;

# COMMAND ----------

# DBTITLE 1,2.8 Notes - AI Functions Overview
# MAGIC %md
# MAGIC ## 2.8 : AI Functions - The Lowest-Code Gen AI Entry Point
# MAGIC
# MAGIC ### AI Functions at a glance
# MAGIC
# MAGIC | Function | What it does | Returns |
# MAGIC |---|---|---|
# MAGIC | `ai_analyze_sentiment(text)` | Positive / negative / neutral / mixed | STRING |
# MAGIC | `ai_classify(text, labels, MAP(...))` | Classify into custom labels | VARIANT |
# MAGIC | `ai_extract(text, schema, MAP(...))` | Pull structured fields from text | VARIANT |
# MAGIC | `ai_summarize(text, max_words)` | Condense text | STRING |
# MAGIC | `ai_translate(text, lang_code)` | Translate to target language | STRING |
# MAGIC | `ai_mask(text, array(...))` | Redact PII entities | STRING |
# MAGIC | `ai_gen(prompt)` | Free-form text generation | STRING |
# MAGIC | `ai_query(endpoint, request)` | Call a specific model endpoint | STRING/STRUCT |
# MAGIC
# MAGIC We run each function on our `module5a_demo2.genai.support_tickets` table to show how AI Functions turn **unstructured customer messages** into **structured, actionable data** - all with a single line of SQL per function.
# MAGIC
# MAGIC > **Important:** `ai_classify` and `ai_extract` should always use `MAP('version', '2.1')` to pin the recommended output contract.

# COMMAND ----------

# DBTITLE 1,2.8a Demo - Sentiment + Classify
# MAGIC %sql
# MAGIC -- 2.8a Demo: ai_analyze_sentiment() + ai_classify()
# MAGIC -- ai_analyze_sentiment(text) → 'positive' | 'negative' | 'neutral' | 'mixed'
# MAGIC -- ai_classify(text, labels_json, MAP(...)) → VARIANT with label + confidence
# MAGIC -- We classify each ticket into support categories with confidence scores.
# MAGIC
# MAGIC SELECT
# MAGIC   ticket_id,
# MAGIC   message,
# MAGIC   ai_analyze_sentiment(message) AS sentiment,
# MAGIC   ai_classify(
# MAGIC     message,
# MAGIC     '{"billing":"Payment, invoice, or refund issues","bug":"App crashes, errors, broken features","feedback":"Praise, complaints, or general opinions","shipping":"Delivery or order damage","account":"Login, password, or profile issues","feature_request":"Suggestions for new features"}',
# MAGIC     MAP('version', '2.1', 'enableConfidenceScores', 'true')
# MAGIC   ) AS category_result
# MAGIC FROM module5a_demo2.genai.support_tickets
# MAGIC ORDER BY ticket_id;

# COMMAND ----------

# DBTITLE 1,2.8b Demo - ai_extract
# MAGIC %sql
# MAGIC -- 2.8b Demo: ai_extract()
# MAGIC -- ai_extract(text, schema_json, MAP(...)) → VARIANT
# MAGIC -- We extract structured fields: issue type, urgency level, and any mentioned amounts.
# MAGIC -- Schema uses the ai_extract user-schema format (NOT SQL DDL, NOT JSON Schema).
# MAGIC -- Allowed types: string, integer, number, boolean, enum, array, object.
# MAGIC
# MAGIC SELECT
# MAGIC   ticket_id,
# MAGIC   message,
# MAGIC   ai_extract(
# MAGIC     message,
# MAGIC     '{
# MAGIC       "issue_type": {"type": "string", "description": "The type of customer issue"},
# MAGIC       "urgency": {"type": "enum", "labels": ["low", "medium", "high"], "description": "How urgent is this ticket"},
# MAGIC       "amount_mentioned": {"type": "number", "description": "Any dollar amount mentioned, null if none"}
# MAGIC     }',
# MAGIC     MAP('version', '2.1', 'enableConfidenceScores', 'true')
# MAGIC   ) AS extracted_fields
# MAGIC FROM module5a_demo2.genai.support_tickets
# MAGIC ORDER BY ticket_id;

# COMMAND ----------

# DBTITLE 1,2.8c Demo - Summarize + Translate
# MAGIC %sql
# MAGIC -- 2.8c Demo: ai_summarize() + ai_translate()
# MAGIC -- ai_summarize(text, max_words) → STRING summary
# MAGIC -- ai_translate(text, lang_code) → STRING translated text
# MAGIC -- We summarize each ticket to 10 words and translate the Spanish ticket (#9) to English.
# MAGIC
# MAGIC SELECT
# MAGIC   ticket_id,
# MAGIC   message,
# MAGIC   ai_summarize(message, 10) AS summary,
# MAGIC   CASE 
# MAGIC     WHEN ticket_id = 9 THEN ai_translate(message, 'en')
# MAGIC     ELSE NULL
# MAGIC   END AS translated_to_english
# MAGIC FROM module5a_demo2.genai.support_tickets
# MAGIC ORDER BY ticket_id;

# COMMAND ----------

# DBTITLE 1,2.8d Demo - ai_mask
# MAGIC %sql
# MAGIC -- 2.8d Demo: ai_mask()
# MAGIC -- ai_mask(text, array('entity_types')) → STRING with [MASKED] replacements
# MAGIC -- We redact emails, phone numbers, and credit card info from the tickets.
# MAGIC -- This is critical for compliance: masked data can be shared with wider teams.
# MAGIC
# MAGIC SELECT
# MAGIC   ticket_id,
# MAGIC   customer,
# MAGIC   message AS original_message,
# MAGIC   ai_mask(message, array('email', 'phone', 'credit_card')) AS masked_message
# MAGIC FROM module5a_demo2.genai.support_tickets
# MAGIC ORDER BY ticket_id;

# COMMAND ----------

# DBTITLE 1,2.9 Notes - Compound AI Systems
# MAGIC %md
# MAGIC ## 2.9 : Compound AI Systems
# MAGIC
# MAGIC ### Concept
# MAGIC A compound AI system is not just a model - it combines:
# MAGIC * **Model** (LLM for understanding / generation)
# MAGIC * **Retrieval** (RAG for grounded knowledge)
# MAGIC * **Tools** (UC functions, code interpreters)
# MAGIC * **Classical ML** (predictive models)
# MAGIC * **Memory** (conversation context)
# MAGIC * **Guardrails** (safety filters, rate limits)
# MAGIC
# MAGIC All components are governed by **Unity Catalog** - consistent access control, lineage, and audit.
# MAGIC
# MAGIC We chain **three AI functions** into a single pipeline:
# MAGIC 1. `ai_mask()` - redact PII from the message
# MAGIC 2. `ai_classify()` - classify the cleaned message
# MAGIC 3. `ai_summarize()` - produce a one-line summary for the support dashboard
# MAGIC
# MAGIC This demonstrates **composability**: the output of one AI function feeds the input of the next, all in a single SQL query - no intermediate tables, no data movement.

# COMMAND ----------

# DBTITLE 1,2.9 Demo - Compound AI Pipeline
# MAGIC %sql
# MAGIC -- 2.9 Demo: Compound AI pipeline — chain three AI functions in one query
# MAGIC -- Step 1: ai_mask()      → redact PII (emails, phones, credit cards)
# MAGIC -- Step 2: ai_classify()  → categorise the cleaned text into support categories
# MAGIC -- Step 3: ai_summarize()  → produce a dashboard-ready one-line summary
# MAGIC -- All in a single CTE + SELECT — no intermediate tables, no data movement.
# MAGIC
# MAGIC WITH cleaned AS (
# MAGIC   SELECT
# MAGIC     ticket_id,
# MAGIC     customer,
# MAGIC     ai_mask(message, array('email', 'phone', 'credit_card')) AS masked_message
# MAGIC   FROM module5a_demo2.genai.support_tickets
# MAGIC )
# MAGIC SELECT
# MAGIC   ticket_id,
# MAGIC   customer,
# MAGIC   masked_message,
# MAGIC   ai_classify(
# MAGIC     masked_message,
# MAGIC     '{"billing":"Payment, invoice, or refund issues","bug":"App crashes, errors, broken features","feedback":"Praise, complaints, or general opinions","shipping":"Delivery or order damage","account":"Login, password, or profile issues","feature_request":"Suggestions for new features"}',
# MAGIC     MAP('version', '2.1')
# MAGIC   ) AS category,
# MAGIC   ai_summarize(masked_message, 8) AS dashboard_summary
# MAGIC FROM cleaned
# MAGIC ORDER BY ticket_id;

# COMMAND ----------

# DBTITLE 1,2.10 Notes - Platform Recap
# MAGIC %md
# MAGIC ## 2.10 : Databricks AI Platform - Five Workflow Stages
# MAGIC
# MAGIC ### The five stages
# MAGIC 1. **Access a Model** - AI Playground, Foundation Model APIs, external models via AI Gateway
# MAGIC 2. **Build** - AI Functions, Agent Framework, notebooks
# MAGIC 3. **Prepare & Serve Data** - AI Search, Vector Search, Feature Store, Unity Catalog volumes
# MAGIC 4. **Deploy** - Model Serving endpoints, Agent Bricks, Genie Spaces
# MAGIC 5. **Govern & Monitor** - Unity Catalog, AI Gateway (guardrails, rate limits), Inference Tables
# MAGIC
# MAGIC ### Where this demo fits
# MAGIC Every function we ran today lives in **stage 2 (Build)** and operates on data in **stage 3 (Prepare & Serve Data)**. The tables are governed by **stage 5 (Govern & Monitor)**. We didn't need to set up any external infrastructure - everything ran inside a single SQL notebook.

# COMMAND ----------

# DBTITLE 1,2.10 Demo - Deploy Stage
# MAGIC %sql
# MAGIC -- 2.10 Demo: The Deploy Stage - From AI Functions to Production
# MAGIC -- This demo shows how the functions we ran connect to the Deploy stage.
# MAGIC -- ai_query() calls a DEPLOYED model serving endpoint.
# MAGIC -- In this demo, we used Foundation Model APIs (pre-deployed by Databricks).
# MAGIC -- In production, you would deploy your own models to serving endpoints.
# MAGIC
# MAGIC -- 1. Show the governed data we created (Stage 3: Prepare & Serve Data)
# MAGIC SHOW TABLES IN module5a_demo2.genai;
# MAGIC
# MAGIC -- 2. The ai_query() calls we ran throughout this demo go through
# MAGIC --    a deployed serving endpoint (Stage 4: Deploy).
# MAGIC --    Here we call two different deployed models on the SAME prompt
# MAGIC --    to show that 'deploy' means 'available as an API endpoint'.
# MAGIC SELECT
# MAGIC   ai_query(
# MAGIC     'databricks-meta-llama-3-3-70b-instruct',
# MAGIC     'In one sentence, what does the Deploy stage mean on Databricks?',
# MAGIC     modelParameters => named_struct('temperature', 0.0, 'max_tokens', 80)
# MAGIC   ) AS llama_3_response,
# MAGIC   ai_query(
# MAGIC     'databricks-llama-4-maverick',
# MAGIC     'In one sentence, what does the Deploy stage mean on Databricks?',
# MAGIC     modelParameters => named_struct('temperature', 0.0, 'max_tokens', 80)
# MAGIC   ) AS llama_4_response;
# MAGIC
# MAGIC -- 3. Three deployment options on Databricks:
# MAGIC --    a. Model Serving: deploy any model as a REST API endpoint (what ai_query calls)
# MAGIC --    b. Agent Bricks: managed agent platform (no-code deployment)
# MAGIC --    c. Genie Spaces: NL interface to governed data (zero-deployment)
# MAGIC --
# MAGIC --    All three are governed by Unity Catalog (Stage 5: Govern & Monitor).
# MAGIC --    The catalog/schema/tables we created are UC-governed.
# MAGIC --    When you deploy a model, UC controls who can call it.
# MAGIC
# MAGIC -- 4. Show UC governance: who can access this catalog? (Stage 5: Govern)
# MAGIC SHOW GRANTS ON CATALOG module5a_demo2;

# COMMAND ----------

# DBTITLE 1,Learning Conclusion
# MAGIC %md
# MAGIC ## Learning Conclusion
# MAGIC
# MAGIC ### What we demonstrated
# MAGIC
# MAGIC | Topic | Function(s) Used | Key Takeaway |
# MAGIC |---|---|---|
# MAGIC | 2.1-2.3 | `ai_gen()` | Gen AI creates structured text from a prompt - the simplest entry point |
# MAGIC | 2.4 | `ai_query()` + `modelParameters` | Temperature controls determinism; LLMs predict one token at a time |
# MAGIC | 2.5-2.6 | `ai_query()` with two models | Model choice is a trade-off across privacy, quality, cost, latency |
# MAGIC | 2.7 | AI functions on a UC table | Data-native architecture: AI runs *on* your data, no export needed |
# MAGIC | 2.8 | `ai_analyze_sentiment`, `ai_classify`, `ai_extract`, `ai_summarize`, `ai_translate`, `ai_mask` | Six functions, one line of SQL each - unstructured text → structured data |
# MAGIC | 2.9 | Chained `ai_mask` → `ai_classify` → `ai_summarize` | Compound AI systems compose functions into pipelines |
# MAGIC | 2.10 | - | Five platform stages: Access → Build → Prepare & Serve → Deploy → Govern |
# MAGIC
# MAGIC ### Key principles
# MAGIC * **AI Functions are SQL** - no Python, no SDKs, no infrastructure. If you can write a SELECT, you can use Gen AI.
# MAGIC * **Data stays in the lakehouse** - no egress, no external API calls, full Unity Catalog governance.
# MAGIC * **Composability** - chain functions in a single query to build compound AI pipelines.
# MAGIC * **Model choice matters** - use `ai_gen` for convenience, `ai_query` for control.

# COMMAND ----------

# DBTITLE 1,Cleanup - Decommission Demo Artifacts
# MAGIC %sql
# MAGIC -- ═══════════════════════════════════════════════════════════════
# MAGIC -- CLEANUP: Decommission everything created in this demo
# MAGIC -- ═══════════════════════════════════════════════════════════════
# MAGIC -- Drop the schema (cascades to tables) and catalog so no demo
# MAGIC -- artifacts remain in the workspace.
# MAGIC
# MAGIC DROP SCHEMA IF EXISTS module5a_demo2.genai CASCADE;
# MAGIC DROP CATALOG IF EXISTS module5a_demo2 CASCADE;
# MAGIC
# MAGIC -- Verify cleanup - catalog should no longer exist
# MAGIC SHOW CATALOGS LIKE 'module5a_demo2';