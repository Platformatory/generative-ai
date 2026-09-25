# Databricks notebook source
# DBTITLE 1,Demo Title
# MAGIC %md
# MAGIC # Demo 3 : PDF-to-Searchable-Index Pipeline
# MAGIC
# MAGIC **Module 5A - Hour 3: RAG Fundamentals & Databricks AI Search (Topics 3.1-3.10)**
# MAGIC
# MAGIC This notebook demonstrates the full RAG pipeline: document processing, chunking, embeddings, AI Search index creation, and similarity queries - all on the Databricks platform.
# MAGIC
# MAGIC Each section has **Notes** (concept + rationale) followed by a **Demo** cell with live code.
# MAGIC At the end: **Learning Conclusion** and **Cleanup** to decommission everything created.

# COMMAND ----------

# DBTITLE 1,Setup - Catalog + Knowledge Base
# MAGIC %sql
# MAGIC -- SETUP: Create catalog, schema, and knowledge base table
# MAGIC -- documents table that has CDF and a primary key (both required
# MAGIC -- for Delta Sync vector search indexes).
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS module5a_demo3;
# MAGIC CREATE SCHEMA IF NOT EXISTS module5a_demo3.rag;
# MAGIC CREATE VOLUME IF NOT EXISTS module5a_demo3.rag.pdf_documents;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE module5a_demo3.rag.knowledge_base (
# MAGIC   doc_id     STRING NOT NULL,
# MAGIC   title      STRING,
# MAGIC   content    STRING,
# MAGIC   category   STRING,
# MAGIC   updated_at TIMESTAMP DEFAULT current_timestamp(),
# MAGIC   CONSTRAINT pk_knowledge_base PRIMARY KEY (doc_id)
# MAGIC ) TBLPROPERTIES (
# MAGIC   delta.enableChangeDataFeed = true,
# MAGIC   'delta.feature.allowColumnDefaults' = 'supported'
# MAGIC );
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Setup - Copy PDF to Volume
# Setup: Copy the sample PDF to a UC Volume
# ai_parse_document requires binary files in a UC Volume (not workspace files).
# We create the volume, then copy the PDF from the workspace static folder.

spark.sql("CREATE VOLUME IF NOT EXISTS module5a_demo3.rag.pdf_documents")

pdf_source = "file:/Workspace/Users/shriveens@platformatory.com/Data+AI Academy/Module 5A/static/databricks-ebook-a-compact-guide-to-agent-systems.pdf"
pdf_dest = "/Volumes/module5a_demo3/rag/pdf_documents/"

dbutils.fs.cp(pdf_source, pdf_dest, recurse=False)
print("PDF copied to volume")
for f in dbutils.fs.ls(pdf_dest):
    print(f"  {f.name} ({f.size:,} bytes)")

# COMMAND ----------

# DBTITLE 1,Setup - Parse PDF into Knowledge Base
# MAGIC %sql
# MAGIC -- Setup: Parse the PDF with ai_parse_document and populate the knowledge base
# MAGIC -- ai_parse_document(content, MAP('version', '2.0')) returns a VARIANT with:
# MAGIC --   parsed:document:elements - array of text, tables, figures, titles, etc.
# MAGIC --   parsed:document:pages - page metadata
# MAGIC -- We extract text elements and insert them as documents.
# MAGIC
# MAGIC TRUNCATE TABLE module5a_demo3.rag.knowledge_base;
# MAGIC
# MAGIC INSERT INTO module5a_demo3.rag.knowledge_base
# MAGIC WITH parsed AS (
# MAGIC   SELECT 
# MAGIC     ai_parse_document(content, MAP('version', '2.0')) AS parsed
# MAGIC   FROM READ_FILES(
# MAGIC     '/Volumes/module5a_demo3/rag/pdf_documents/',
# MAGIC     format => 'binaryFile'
# MAGIC   )
# MAGIC ),
# MAGIC elements AS (
# MAGIC   SELECT posexplode(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS (pos, element)
# MAGIC   FROM parsed
# MAGIC   WHERE is_variant_null(parsed:error_status)
# MAGIC )
# MAGIC SELECT
# MAGIC   concat('doc-', lpad(cast(row_number() OVER (ORDER BY pos) AS STRING), 3, '0')) AS doc_id,
# MAGIC   CASE 
# MAGIC     WHEN element:type::STRING = 'title' THEN element:content::STRING
# MAGIC     WHEN element:type::STRING = 'section_header' THEN element:content::STRING
# MAGIC     ELSE concat('Section ', cast(pos AS STRING))
# MAGIC   END AS title,
# MAGIC   element:content::STRING AS content,
# MAGIC   element:type::STRING AS category,
# MAGIC   current_timestamp() AS updated_at
# MAGIC FROM elements
# MAGIC WHERE element:type::STRING IN ('text', 'title', 'section_header')
# MAGIC   AND length(coalesce(element:content::STRING, '')) > 30;
# MAGIC
# MAGIC SELECT count(*) AS total_docs, collect_set(category) AS doc_types
# MAGIC FROM module5a_demo3.rag.knowledge_base;

# COMMAND ----------

# DBTITLE 1,Setup - Vector Search Endpoint
# Setup: Create a Vector Search endpoint
# This is a managed compute resource that hosts vector indexes.
# We use STANDARD for faster provisioning.
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType
import time

w = WorkspaceClient()
endpoint_name = "demo3_vs_endpoint"

# Create endpoint if it doesn't exist
try:
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name=endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists: {ep.endpoint_status.state}")
except Exception:
    w.vector_search_endpoints.create_endpoint(
        name=endpoint_name,
        endpoint_type=EndpointType.STANDARD,
    )
    print(f"Creating endpoint '{endpoint_name}'... (takes ~2-3 min)")

# Wait for endpoint to be ready
for i in range(30):
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name=endpoint_name)
    state = str(ep.endpoint_status.state)
    print(f"  State: {state}")
    if "ONLINE" in state:
        print(f"Endpoint '{endpoint_name}' is ready!")
        break
    time.sleep(10)

# COMMAND ----------

# DBTITLE 1,3.1 Notes - Context vs. Prompt
# MAGIC %md
# MAGIC ## 3.1 : Context Engineering vs. Prompt Engineering
# MAGIC
# MAGIC ### Concepts
# MAGIC * Two ways to adapt a model's behavior:
# MAGIC   * **Prompt engineering**: craft better instructions (change what you ask).
# MAGIC   * **Context engineering**: provide better background knowledge (change what the model sees).
# MAGIC * Context engineering is the primary discipline in practice - most "prompt" issues are actually missing-context issues.
# MAGIC * RAG (Retrieval-Augmented Generation) is the most common context engineering pattern: retrieve relevant documents, inject them as context, then let the model answer.
# MAGIC
# MAGIC `ai_query()` lets us demonstrate both approaches. Without context, the model has no knowledge of our specific platform. With context retrieved from our knowledge base, it gives grounded, specific answers.

# COMMAND ----------

# DBTITLE 1,3.1 Demo - Context vs. Prompt
# MAGIC %sql
# MAGIC -- 3.1 Demo: Context engineering vs. prompt engineering
# MAGIC -- Without context: the model gives a generic answer
# MAGIC -- With context: the model gives a grounded, specific answer
# MAGIC
# MAGIC -- Without context (prompt engineering only)
# MAGIC SELECT ai_query(
# MAGIC   'databricks-meta-llama-3-3-70b-instruct',
# MAGIC   'What is an AI agent? Answer in 2 sentences.',
# MAGIC   modelParameters => named_struct('temperature', 0.0, 'max_tokens', 100)
# MAGIC ) AS answer_without_context;
# MAGIC
# MAGIC -- With context (context engineering - we inject retrieved knowledge)
# MAGIC SELECT ai_query(
# MAGIC   'databricks-meta-llama-3-3-70b-instruct',
# MAGIC   'Based on this context: "' || (
# MAGIC     SELECT content FROM module5a_demo3.rag.knowledge_base WHERE doc_id = 'doc-006'
# MAGIC   ) || '" What is an AI agent? Answer in 2 sentences.',
# MAGIC   modelParameters => named_struct('temperature', 0.0, 'max_tokens', 100)
# MAGIC ) AS answer_with_context;

# COMMAND ----------

# DBTITLE 1,3.2 Notes - RAG Pattern
# MAGIC %md
# MAGIC ## 3.2 : The RAG Pattern - Indexing Phase vs. Query Phase
# MAGIC
# MAGIC ### Concepts
# MAGIC RAG has two distinct phases:
# MAGIC
# MAGIC * **Indexing phase** (done once or periodically):
# MAGIC   1. Load documents (PDFs, text, structured data)
# MAGIC   2. Chunk documents into smaller pieces
# MAGIC   3. Generate embeddings for each chunk
# MAGIC   4. Store embeddings in a vector search index
# MAGIC
# MAGIC * **Query phase** (every user question):
# MAGIC   1. Embed the user's question
# MAGIC   2. Search the vector index for similar chunks (similarity search / ANN)
# MAGIC   3. Inject retrieved chunks as context into the LLM prompt
# MAGIC   4. LLM generates a grounded answer
# MAGIC
# MAGIC We simulate the query phase: retrieve relevant context, then generate. In a full RAG pipeline, the retrieval step uses `vector_search()` against an AI Search index.

# COMMAND ----------

# DBTITLE 1,3.2 Demo - RAG Query Phase
# MAGIC %sql
# MAGIC -- 3.2 Demo: The RAG query phase (simulated)
# MAGIC -- Step 1: Retrieve relevant context (here we use a direct doc lookup;
# MAGIC --   in production, vector_search() finds semantically similar docs)
# MAGIC -- Step 2: Generate a grounded answer using the retrieved context
# MAGIC
# MAGIC WITH retrieved_context AS (
# MAGIC   SELECT content FROM module5a_demo3.rag.knowledge_base
# MAGIC   WHERE doc_id = 'doc-007'
# MAGIC   LIMIT 1
# MAGIC )
# MAGIC SELECT ai_query(
# MAGIC   'databricks-meta-llama-3-3-70b-instruct',
# MAGIC   'Answer this question using only the provided context.\n\nContext: ' || content || '\n\nQuestion: How does an AI agent use an LLM for reasoning?',
# MAGIC   modelParameters => named_struct('temperature', 0.0, 'max_tokens', 100)
# MAGIC ) AS rag_answer
# MAGIC FROM retrieved_context;

# COMMAND ----------

# DBTITLE 1,3.3 Notes - Why Retrieval Agents
# MAGIC %md
# MAGIC ## 3.3 : Why Retrieval Agents Exist
# MAGIC
# MAGIC ### Concepts
# MAGIC Three problems that RAG solves:
# MAGIC 1. **Knowledge cutoff**: Models are trained on data up to a point. They don't know about events, products, or policies after that date.
# MAGIC 2. **Hallucination**: Without grounded context, models fabricate plausible but incorrect answers.
# MAGIC 3. **Missing private context**: Models have no access to your organization's internal documents, data, or knowledge base.
# MAGIC
# MAGIC RAG addresses all three by retrieving relevant, authoritative context from your own data before generating an answer.

# COMMAND ----------

# DBTITLE 1,3.3 Demo - Hallucination vs. Grounded
# MAGIC %sql
# MAGIC -- 3.3 Demo: Hallucination vs. grounded answer
# MAGIC -- Without our knowledge base, the model may hallucinate about
# MAGIC -- our internal product details. With context, it stays grounded.
# MAGIC
# MAGIC -- Without context: the model guesses or fabricates
# MAGIC SELECT ai_query(
# MAGIC   'databricks-meta-llama-3-3-70b-instruct',
# MAGIC   'How does Databricks Mosaic AI help build agent systems? Answer briefly.',
# MAGIC   modelParameters => named_struct('temperature', 0.0, 'max_tokens', 50)
# MAGIC ) AS hallucinated_answer;
# MAGIC
# MAGIC -- With context from our knowledge base: the model is grounded
# MAGIC SELECT ai_query(
# MAGIC   'databricks-meta-llama-3-3-70b-instruct',
# MAGIC   'Context: "' || content || '"\n\nQuestion: How does Databricks Mosaic AI help build agent systems? Answer briefly.',
# MAGIC   modelParameters => named_struct('temperature', 0.0, 'max_tokens', 50)
# MAGIC ) AS grounded_answer
# MAGIC FROM module5a_demo3.rag.knowledge_base
# MAGIC WHERE doc_id = 'doc-004';

# COMMAND ----------

# DBTITLE 1,3.4 Notes - Document Pipeline
# MAGIC %md
# MAGIC ## 3.4 : Document Processing Pipeline
# MAGIC
# MAGIC ### Concepts
# MAGIC The full document processing pipeline:
# MAGIC 1. `ai_parse_document(content, MAP('version', '2.0'))` - parse PDFs/images into structured VARIANT (elements: text, tables, figures, titles)
# MAGIC 2. `ai_classify(parsed_content, labels, MAP(...))` - categorize documents by type
# MAGIC 3. `ai_extract(parsed_content, schema, MAP(...))` - extract structured fields
# MAGIC 4. `ai_prep_search(parsed_content)` - chunk for vector search (semantic chunking)
# MAGIC
# MAGIC We use a real PDF (Databricks eBook: A Compact Guide to Agent Systems) uploaded to a UC Volume. `ai_parse_document` parses it into structured VARIANT. We show the raw parsed output, then classify and extract key fields.
# MAGIC
# MAGIC > `ai_parse_document` requires binary files (PDFs, images, Office docs) stored in a UC Volume. See the SQL reference for the full parsing pipeline.

# COMMAND ----------

# DBTITLE 1,3.4a Demo - Raw ai_parse_document
# MAGIC %sql
# MAGIC -- 3.4a Demo: Raw ai_parse_document output on the actual PDF
# MAGIC -- This shows what ai_parse_document returns before any processing.
# MAGIC -- The VARIANT contains document.pages (page metadata) and
# MAGIC -- document.elements (text, tables, figures, titles, etc.).
# MAGIC
# MAGIC SELECT 
# MAGIC   path,
# MAGIC   size(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS num_elements,
# MAGIC   size(try_cast(parsed:document:pages AS ARRAY<VARIANT>)) AS num_pages,
# MAGIC   parsed:metadata AS metadata
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     path,
# MAGIC     ai_parse_document(content, MAP('version', '2.0')) AS parsed
# MAGIC   FROM READ_FILES(
# MAGIC     '/Volumes/module5a_demo3/rag/pdf_documents/',
# MAGIC     format => 'binaryFile'
# MAGIC   )
# MAGIC );
# MAGIC
# MAGIC -- Show the first 10 elements with their types and a content preview
# MAGIC SELECT 
# MAGIC   element:type::STRING AS element_type,
# MAGIC   substring(element:content::STRING, 1, 120) AS content_preview
# MAGIC FROM (
# MAGIC   SELECT posexplode(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS (pos, element)
# MAGIC   FROM (
# MAGIC     SELECT ai_parse_document(content, MAP('version', '2.0')) AS parsed
# MAGIC     FROM READ_FILES(
# MAGIC       '/Volumes/module5a_demo3/rag/pdf_documents/',
# MAGIC       format => 'binaryFile'
# MAGIC     )
# MAGIC   )
# MAGIC   WHERE is_variant_null(parsed:error_status)
# MAGIC )
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,3.4 Demo - Classify + Extract
# MAGIC %sql
# MAGIC -- 3.4 Demo: Document processing pipeline - classify and extract
# MAGIC -- Our knowledge_base table was populated from the real PDF using ai_parse_document.
# MAGIC -- Now we classify each document and extract key fields using AI Functions.
# MAGIC
# MAGIC SELECT
# MAGIC   doc_id,
# MAGIC   title,
# MAGIC   ai_classify(
# MAGIC     content,
# MAGIC     '{"intro":"Introduction or overview of AI agents","architecture":"Agent architecture, reasoning, or design","use_cases":"Business use cases and applications","deployment":"Building, deploying, and evaluating agents","governance":"Governance, security, and compliance"}',
# MAGIC     MAP('version', '2.1', 'enableConfidenceScores', 'true')
# MAGIC   ) AS doc_category,
# MAGIC   ai_extract(
# MAGIC     content,
# MAGIC     '{
# MAGIC       "primary_topic": {"type": "string", "description": "Main subject of the document"},
# MAGIC       "has_code": {"type": "boolean", "description": "Does the document mention code or programming"},
# MAGIC       "key_terms": {"type": "array", "items": {"type": "string"}}
# MAGIC     }',
# MAGIC     MAP('version', '2.1')
# MAGIC   ) AS extracted_fields
# MAGIC FROM module5a_demo3.rag.knowledge_base
# MAGIC ORDER BY doc_id;

# COMMAND ----------

# DBTITLE 1,3.5 Notes - Chunking Strategy
# MAGIC %md
# MAGIC ## 3.5 : Chunking Strategy Considerations
# MAGIC
# MAGIC ### Concepts
# MAGIC * **Chunk size**: Balance between too small (loses context) and too large (dilutes relevance, hits embedding model token limits).
# MAGIC * **Overlap**: 10-20% overlap between chunks prevents losing context at boundaries.
# MAGIC * **Granularity trade-off**: Smaller chunks = more precise retrieval but less context per chunk. Larger chunks = more context but less precise.
# MAGIC * **"Lost in the middle"**: Models pay less attention to content in the middle of long contexts. Keep chunks focused.
# MAGIC
# MAGIC We demonstrate manual chunking by splitting documents into sentences using SQL string functions. In production, use `ai_prep_search` for semantic chunking that respects paragraph and section boundaries.

# COMMAND ----------

# DBTITLE 1,3.5 Demo - Manual Chunking
# MAGIC %sql
# MAGIC -- 3.5 Demo: Manual chunking of documents
# MAGIC -- We split documents by sentences and show each as a potential chunk.
# MAGIC -- In production, ai_prep_search handles this automatically with
# MAGIC -- semantic awareness (respecting sentence/paragraph boundaries).
# MAGIC
# MAGIC WITH sentences AS (
# MAGIC   SELECT
# MAGIC     doc_id,
# MAGIC     title,
# MAGIC     sentence,
# MAGIC     row_number() OVER (PARTITION BY doc_id ORDER BY 1) AS sentence_num
# MAGIC   FROM module5a_demo3.rag.knowledge_base
# MAGIC   LATERAL VIEW explode(split(content, '\\. ')) AS sentence
# MAGIC   WHERE sentence != ''
# MAGIC )
# MAGIC SELECT
# MAGIC   doc_id,
# MAGIC   title,
# MAGIC   sentence_num AS chunk_id,
# MAGIC   length(sentence) AS chunk_length,
# MAGIC   sentence AS chunk_text
# MAGIC FROM sentences
# MAGIC ORDER BY doc_id, sentence_num
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,3.6 Notes - Embed vs. Retrieve Chunks
# MAGIC %md
# MAGIC ## 3.6 : chunk_to_embed vs. chunk_to_retrieve
# MAGIC
# MAGIC ### Concepts
# MAGIC * **chunk_to_embed**: The text used to generate the embedding vector. Optimized for findability - may include context enrichment (title, summary, keywords) to improve semantic matching.
# MAGIC * **chunk_to_retrieve**: The actual text returned to the LLM as context. Clean, raw content without enrichment.
# MAGIC * These can be the same text, but separating them improves retrieval quality:
# MAGIC   * The embedding input benefits from extra context (title, category).
# MAGIC   * The retrieval output should be clean to avoid confusing the LLM.
# MAGIC
# MAGIC We create a chunked table with both columns to show the distinction.

# COMMAND ----------

# DBTITLE 1,3.6 Demo - Embed vs. Retrieve
# MAGIC %sql
# MAGIC -- 3.6 Demo: chunk_to_embed vs. chunk_to_retrieve
# MAGIC -- We create a chunked version of our knowledge base with two text columns:
# MAGIC -- - chunk_to_embed: enriched with title + category for better findability
# MAGIC -- - chunk_to_retrieve: clean raw content for the LLM
# MAGIC
# MAGIC CREATE OR REPLACE TABLE module5a_demo3.rag.document_chunks AS
# MAGIC WITH sentences AS (
# MAGIC   SELECT
# MAGIC     doc_id,
# MAGIC     title,
# MAGIC     category,
# MAGIC     sentence,
# MAGIC     row_number() OVER (PARTITION BY doc_id ORDER BY 1) AS sentence_num
# MAGIC   FROM module5a_demo3.rag.knowledge_base
# MAGIC   LATERAL VIEW explode(split(content, '\\. ')) AS sentence
# MAGIC   WHERE sentence != ''
# MAGIC )
# MAGIC SELECT
# MAGIC   concat(doc_id, '-', lpad(cast(sentence_num AS STRING), 3, '0')) AS chunk_id,
# MAGIC   doc_id,
# MAGIC   title,
# MAGIC   category,
# MAGIC   -- chunk_to_embed: enriched with title + category for findability
# MAGIC   concat(title, ' | ', category, ' | ', sentence) AS chunk_to_embed,
# MAGIC   -- chunk_to_retrieve: clean raw text for the LLM
# MAGIC   sentence AS chunk_to_retrieve
# MAGIC FROM sentences;
# MAGIC
# MAGIC SELECT chunk_id, chunk_to_embed, chunk_to_retrieve
# MAGIC FROM module5a_demo3.rag.document_chunks
# MAGIC ORDER BY chunk_id
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,3.7 Notes - Embeddings & Similarity
# MAGIC %md
# MAGIC ## 3.7 : Embeddings & Cosine Similarity
# MAGIC
# MAGIC ### Concepts
# MAGIC * **Embeddings**: Numeric vectors that capture the semantic meaning of text. Similar meanings = similar vectors.
# MAGIC * **Cosine similarity**: Measures the angle between two vectors. 1 = identical meaning, 0 = unrelated.
# MAGIC * **ANN (Approximate Nearest Neighbor)**: The algorithm AI Search uses to find similar vectors efficiently without comparing against all vectors.
# MAGIC * Databricks provides managed embedding models: `databricks-gte-large-en` (1024 dims, 8192 token context) and `databricks-bge-large-en` (1024 dims, 512 token context).
# MAGIC
# MAGIC We use the Databricks SDK to call an embedding model and compute cosine similarity between two texts.

# COMMAND ----------

# DBTITLE 1,3.7 Demo - Embeddings & Similarity
# 3.7 Demo: Generate embeddings and compute cosine similarity
# We call the embedding model endpoint to generate vectors for two texts
# and compute cosine similarity between them.

import mlflow.deployments
import numpy as np

client = mlflow.deployments.get_deploy_client("databricks")

# Generate embeddings for two texts
response1 = client.predict(
    endpoint="databricks-gte-large-en",
    inputs={"input": ["AI agents use LLMs as their brain for reasoning and decision-making."]}
)
response2 = client.predict(
    endpoint="databricks-gte-large-en",
    inputs={"input": ["Delta Lake provides ACID transactions for reliable data storage."]}
)

emb1 = response1["data"][0]["embedding"]
emb2 = response2["data"][0]["embedding"]

print(f"Embedding dimensions: {len(emb1)}")
print(f"First 5 values (text 1): {[round(v, 4) for v in emb1[:5]]}")
print(f"First 5 values (text 2): {[round(v, 4) for v in emb2[:5]]}")

# Cosine similarity
cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"\nCosine similarity: {cos_sim:.4f}")
print(f"(1.0 = identical meaning, 0.0 = unrelated)")

# COMMAND ----------

# DBTITLE 1,3.8 Notes - AI Search Index
# MAGIC %md
# MAGIC ## 3.8 : Creating & Syncing an AI Search Index
# MAGIC
# MAGIC ### Concepts
# MAGIC * **Endpoint**: Compute resource that hosts indexes (Standard or Storage-Optimized).
# MAGIC * **Delta Sync Index**: Automatically syncs with a source Delta table. Databricks computes embeddings for you.
# MAGIC * **Key parameters**:
# MAGIC   * `endpoint_name`: Vector search endpoint to host the index
# MAGIC   * `source_table_name`: Delta table with CDF and primary key
# MAGIC   * `embedding_model_endpoint_name`: Model for computing embeddings (e.g., `databricks-gte-large-en`)
# MAGIC   * `pipeline_type`: `TRIGGERED` (manual sync) or `CONTINUOUS` (auto-sync)
# MAGIC * **Change Data Feed (CDF)**: Required for Delta Sync - enables incremental updates.
# MAGIC * **Primary Key**: Required constraint on the source table.
# MAGIC
# MAGIC We created the endpoint in setup. Now we create the index, sync it, and query it.

# COMMAND ----------

# DBTITLE 1,3.8 Demo - Create + Query Index
# 3.8 Demo: Create AI Search index, sync, and query
# We create a Delta Sync index from our knowledge_base table.
# Databricks automatically computes embeddings using databricks-gte-large-en.

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest, EmbeddingSourceColumn,
    VectorIndexType, PipelineType
)
import time

w = WorkspaceClient()

index_name = "module5a_demo3.rag.knowledge_base_index"
endpoint_name = "demo3_vs_endpoint"

# Create the index (delete and recreate if not ready)
try:
    idx = w.vector_search_indexes.get_index(index_name=index_name)
    if idx.status.ready:
        print(f"Index already exists and is ready!")
    else:
        print("Index not ready. Deleting and recreating...")
        w.vector_search_indexes.delete_index(index_name=index_name)
        time.sleep(15)
        raise Exception("Force recreate")
except Exception:
    print(f"Creating index '{index_name}'...")
    w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=endpoint_name,
        primary_key="doc_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table="module5a_demo3.rag.knowledge_base",
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="content",
                    embedding_model_endpoint_name="databricks-gte-large-en"
                )
            ],
            pipeline_type=PipelineType.TRIGGERED,
            columns_to_sync=["doc_id", "title", "content", "category"],
        ),
    )
    print("Index creation submitted.")

# Wait for index to be ready
for i in range(60):
    idx = w.vector_search_indexes.get_index(index_name=index_name)
    print(f"  Index ready: {idx.status.ready}")
    if idx.status.ready:
        print("Index is ready!")
        break
    time.sleep(10)

# Sync the index (TRIGGERED mode requires manual sync)
print("\nSyncing index...")
try:
    w.vector_search_indexes.sync_index(index_name=index_name)
    time.sleep(5)
except Exception as e:
    print(f"  Sync already in progress, waiting...")
    time.sleep(30)

# Query the index - semantic search
print("\n=== Semantic Search: 'What is an AI agent?' ===")
results = w.vector_search_indexes.query_index(
    index_name=index_name,
    columns=["doc_id", "title", "content", "category"],
    query_text="What is an AI agent?",
    num_results=3,
)

for row in (results.result.data_array if results.result and results.result.data_array else []):
    score = row[-1]
    print(f"  Score: {score:.4f} | {row[1]} | {row[2][:80]}...")

print("\n=== Semantic Search: 'How do you build an agent system?' ===")
results = w.vector_search_indexes.query_index(
    index_name=index_name,
    columns=["doc_id", "title", "content", "category"],
    query_text="How do you build an agent system?",
    num_results=3,
)

for row in (results.result.data_array if results.result and results.result.data_array else []):
    score = row[-1]
    print(f"  Score: {score:.4f} | {row[1]} | {row[2][:80]}...")

# COMMAND ----------

# DBTITLE 1,3.9 Notes - Managed vs. Custom RAG
# MAGIC %md
# MAGIC ## 3.9 : Managed RAG (Knowledge Assistant) vs. Custom RAG (Agent Framework + AI Search)
# MAGIC
# MAGIC ### Concepts
# MAGIC
# MAGIC | Dimension | Managed RAG (Knowledge Assistant) | Custom RAG (Agent Framework + AI Search) |
# MAGIC |---|---|---|
# MAGIC | Setup time | Minutes (no-code UI) | Hours to days (code) |
# MAGIC | Customizability | Limited (pre-built) | Full control |
# MAGIC | Chunking | Automatic | You control strategy |
# MAGIC | Retrieval | Automatic | You tune search parameters |
# MAGIC | Model choice | Limited to configured models | Any model via endpoints |
# MAGIC | When to use | Quick POC, simple Q&A | Production, complex requirements |
# MAGIC
# MAGIC * **Knowledge Assistant** (Agent Bricks): point at a UC volume of documents, get a chatbot. Best for fast time-to-value.
# MAGIC * **Custom RAG**: build your own pipeline with Agent Framework + AI Search. Best when you need custom chunking, filtering, or multi-step retrieval.
# MAGIC
# MAGIC This demo is a **Custom RAG** pipeline: we controlled the table schema, chunking, embedding model, and index parameters.

# COMMAND ----------

# DBTITLE 1,3.10 Notes - UC Governance
# MAGIC %md
# MAGIC ## 3.10 : Unity Catalog Governance Across the Retrieval Pipeline
# MAGIC
# MAGIC ### Concepts
# MAGIC Every component of the RAG pipeline is governed by Unity Catalog:
# MAGIC * **Volumes**: Source documents stored in UC Volumes (governed storage)
# MAGIC * **Tables**: Parsed/chunked documents are Delta tables (table-level ACLs)
# MAGIC * **Indexes**: AI Search indexes are UC entities (create/query permissions)
# MAGIC * **Endpoints**: Vector Search endpoints have separate ACLs (CAN_USE)
# MAGIC * **Models**: Embedding and LLM endpoints have their own access controls
# MAGIC * **Lineage**: UC tracks data flow from source documents through chunks to index to answer
# MAGIC
# MAGIC All permissions are managed in one place - no separate governance system for AI components.

# COMMAND ----------

# DBTITLE 1,Learning Conclusion
# MAGIC %md
# MAGIC ## Learning Conclusion
# MAGIC
# MAGIC ### What we demonstrated
# MAGIC
# MAGIC | Topic | What was demoed | Key Takeaway |
# MAGIC |---|---|---|
# MAGIC | 3.1 | `ai_query` with and without context | Context engineering > prompt engineering for grounded answers |
# MAGIC | 3.2 | Manual RAG query (retrieve + generate) | RAG = indexing phase + query phase |
# MAGIC | 3.3 | Hallucinated vs. grounded answer | RAG prevents hallucination by grounding in your data |
# MAGIC | 3.4 | `ai_classify` + `ai_extract` on documents | Document processing pipeline: parse, classify, extract |
# MAGIC | 3.5 | Manual sentence-level chunking | Chunk size and overlap affect retrieval quality |
# MAGIC | 3.6 | `chunk_to_embed` vs `chunk_to_retrieve` columns | Enriched text for findability, clean text for the LLM |
# MAGIC | 3.7 | Embedding model + cosine similarity | Embeddings capture meaning; cosine similarity ranks results |
# MAGIC | 3.8 | Vector Search endpoint + Delta Sync index + query | Full AI Search pipeline: create, sync, query |
# MAGIC | 3.9 | Managed vs. Custom RAG comparison | Knowledge Assistant for speed, Custom RAG for control |
# MAGIC | 3.10 | UC governance overview | Every RAG component governed by Unity Catalog |
# MAGIC
# MAGIC ### Key principles
# MAGIC * **RAG = retrieve + generate**: Get relevant context from your data, then let the LLM answer.
# MAGIC * **Chunking matters**: How you split documents affects retrieval quality.
# MAGIC * **AI Search is managed**: Databricks computes embeddings, maintains the index, and serves queries.
# MAGIC * **Everything is governed**: Unity Catalog controls access to every component.
# MAGIC * **Start managed, go custom when needed**: Knowledge Assistant first, Custom RAG when you need control.

# COMMAND ----------

# DBTITLE 1,Cleanup - Decommission Demo Artifacts
# CLEANUP: Decommission everything created in this demo
from databricks.sdk import WorkspaceClient
import time

w = WorkspaceClient()

index_name = "module5a_demo3.rag.knowledge_base_index"
endpoint_name = "demo3_vs_endpoint"

# 1. Delete the vector search index
print("Deleting index...")
try:
    w.vector_search_indexes.delete_index(index_name=index_name)
    print(f"  Deleted index: {index_name}")
except Exception as e:
    print(f"  Index deletion: {e}")

time.sleep(3)

# 2. Delete the vector search endpoint
print("Deleting endpoint...")
try:
    w.vector_search_endpoints.delete_endpoint(endpoint_name=endpoint_name)
    print(f"  Deleted endpoint: {endpoint_name}")
except Exception as e:
    print(f"  Endpoint deletion: {e}")

# 3. Drop the schema and catalog (cascades to tables)
print("Dropping schema and catalog...")
spark.sql("DROP VOLUME IF EXISTS module5a_demo3.rag.pdf_documents")
spark.sql("DROP SCHEMA IF EXISTS module5a_demo3.rag CASCADE")
spark.sql("DROP CATALOG IF EXISTS module5a_demo3 CASCADE")
print("  Dropped schema and catalog")

print("\nCleanup complete!")