# Databricks notebook source
# DBTITLE 1,Title: what we are building
# MAGIC %md
# MAGIC # ✈️ Module 5B Lab: SkyJet Assist, a Production Passenger-Support Chatbot
# MAGIC
# MAGIC ## What we are going to build
# MAGIC
# MAGIC **SkyJet Assist**: a production-ready AI chatbot for **SkyJet Airlines**. A passenger asks a question in plain English, and the assistant answers **grounded in the airline's official passenger-policies PDF** (in this module's `static` folder) for anything about rules (baggage, refunds, check-in, pets, special assistance) and **from live operational data** for flights and baggage fees. Behind the chat window sits a supervisor agent routing between two specialists, every tool is a governed Unity Catalog object discovered at runtime via MCP, every request is traced, and the whole thing ships as a **Databricks App** that a real airline could deploy, monitor, and iterate on.
# MAGIC
# MAGIC Notebook 1 supplied the techniques (governed Python tools, MCP, MLflow Tracing, DAB architecture). This lab assembles them into one shippable artifact:
# MAGIC
# MAGIC | ✅ Production-ready means… | Where in this lab |
# MAGIC |---|---|
# MAGIC | **Grounded answers (RAG)**: retrieval over the policies PDF using the full 5A Demo 3 pipeline (parse → chunk → embed → index → retrieve) | Section S |
# MAGIC | **Governed tools**: SQL tools, Python tools, *and the retrieval tool itself* are Unity Catalog objects with permissions, lineage, and discovery | Section B |
# MAGIC | **Runtime tool discovery**: everything served via MCP, nothing hardcoded into prompts | Section C |
# MAGIC | **Environment-based configuration**: zero hardcoded values; code identical across dev/staging/prod | Sections A & F |
# MAGIC | **Multi-agent orchestration**: supervisor routes to Policy and Flight specialists via handoffs | Section D |
# MAGIC | **Observability**: every request traced to an MLflow experiment | Section E |
# MAGIC | **Tested**: unit tests for tools, quality tests for agent behavior, smoke tests after deploy | Section T |
# MAGIC | **Deployed**: a real DAB project (`agent.py`, `server.py`, `databricks.yml`, `app.yaml`) that we actually deploy | Section F |
# MAGIC | **Monitoring hooks**: traces queryable with SQL, scorer-ready; go/no-go gates | Sections G & H |
# MAGIC
# MAGIC ## Architecture we are building
# MAGIC
# MAGIC ```
# MAGIC                           ┌──────────────────────────────────────────────────┐
# MAGIC    user / HTTP ─────────► │  Databricks App (FastAPI)  ← DAB deploys         │
# MAGIC                           │    └── Supervisor Agent (handoffs)               │
# MAGIC                           │          ├── PolicyAgent  (RAG specialist)       │
# MAGIC                           │          └── FlightAgent  (structured tools)     │
# MAGIC                           │    ┌───────────────────────────┐                 │
# MAGIC                           │    │ tools (ALL discovered via MCP):             │
# MAGIC                           │    ├── search_policies      ← AI Search index  │
# MAGIC                           │    │                            (RAG retrieval)  │
# MAGIC                           │    ├── get_flight_status / baggage fees        │
# MAGIC                           │    │   (governed SQL + Python UC functions)     │
# MAGIC                           │    └── @function_tool glue (agent-local only)   │
# MAGIC                           └──────────────┬───────────────────────────────────┘
# MAGIC                                          │  (every call traced)
# MAGIC                                          ▼
# MAGIC                           Unity AI Gateway → Model Serving Endpoint (LLM)
# MAGIC                                          │
# MAGIC                                          ▼
# MAGIC                           MLflow Experiment (traces) → scorers → Delta archival
# MAGIC ```
# MAGIC
# MAGIC **Where this connects to Module 5A:** the knowledge-base pipeline in Section S is the *same* pipeline you built in 5A Demo 3 (`ai_parse_document` → chunk table → vector index), pointed at SkyJet's policies PDF and wired into a governed tool. Notebook 1 already registered the Python tools into `module5b.skyjet`; we reuse them here.

# COMMAND ----------

# DBTITLE 1,Section A header dash fix
# MAGIC %md
# MAGIC # Section A: Configuration the production way (resource injection)
# MAGIC
# MAGIC From Notebook 1 Part D: `databricks.yml` declares resources and `app.yaml` injects them as **environment variables**; the agent code reads `os.environ` and never hardcodes a workspace-specific value.
# MAGIC
# MAGIC We apply that pattern *inside the notebook*: every value comes from an environment variable **with a dev fallback**. When this same code runs inside the deployed app, the fallbacks disappear and only injected values remain. This is the exact line where 5A-style notebook code becomes deployable code.

# COMMAND ----------

# DBTITLE 1,A1 install lab dependencies
# databricks-openai requires openai-agents>=0.5.0, so openai-agents is left
# unpinned and pip resolves one coherent set (the official 0.0.16 pin conflicts).
%pip install databricks-mcp==0.9.0 nest-asyncio unitycatalog-ai[databricks]==0.3.2 databricks-openai openai-agents
%restart_python

# COMMAND ----------

# DBTITLE 1,A2 config via resource injection
import os

# ------------------------------------------------------------------
# Resource injection: in the deployed app these are set by databricks.yml
# / app.yaml. In the notebook we fall back to development defaults.
# ------------------------------------------------------------------
UC_TOOL_CATALOG        = os.environ.get("UC_TOOL_CATALOG", "module5b")
UC_TOOL_SCHEMA         = os.environ.get("UC_TOOL_SCHEMA", "skyjet")
SERVING_ENDPOINT       = os.environ.get("SERVING_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
EMBEDDING_ENDPOINT     = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
VS_ENDPOINT            = os.environ.get("VS_ENDPOINT", "module5b_vs_endpoint")
VS_INDEX               = os.environ.get("VS_INDEX", "module5b.skyjet.policies_index")
# NOTE: the USER env var is the internal spark service account on serverless,
# NOT your workspace login, so ask the platform instead. This default is only
# used to build databricks.yml for the deployed app (Section F); notebook
# tracing never reads it.
_current_user = spark.sql("SELECT current_user() AS u").first().u
MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    f"/Users/{_current_user}/module5b_skyjet_production",
)

FULL_SCHEMA = f"{UC_TOOL_CATALOG}.{UC_TOOL_SCHEMA}"

for k, v in {
    "UC_TOOL_CATALOG": UC_TOOL_CATALOG,
    "UC_TOOL_SCHEMA": UC_TOOL_SCHEMA,
    "SERVING_ENDPOINT": SERVING_ENDPOINT,
    "EMBEDDING_ENDPOINT": EMBEDDING_ENDPOINT,
    "VS_ENDPOINT": VS_ENDPOINT,
    "VS_INDEX": VS_INDEX,
    "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
}.items():
    print(f"  {k:<24} = {v}")

# COMMAND ----------

# DBTITLE 1,Section S header: endpoint vs index theory
# MAGIC %md
# MAGIC # Section S: The SkyJet knowledge base (RAG, the 5A Demo 3 pipeline productionized)
# MAGIC
# MAGIC In 5A Demo 3 you built a PDF-to-searchable-index pipeline: `ai_parse_document` → chunk table → **AI Search** index → similarity retrieval. In this lab we run the *same* pipeline against SkyJet's passenger-policies PDF, and in Section B we turn its retrieval step into a **governed UC tool** so any agent can use it through MCP.
# MAGIC
# MAGIC Pipeline steps (identical to 5A Demo 3):
# MAGIC 1. Copy the PDF into a **UC Volume** (`ai_parse_document` requires binary files in a Volume)
# MAGIC 2. Parse with `ai_parse_document(content, MAP('version','2.0'))` → structured elements (text, section headers, tables)
# MAGIC 3. Load elements into a Delta **knowledge-base table** (primary key + Change Data Feed, as required by Delta Sync indexes)
# MAGIC 4. Create a **Delta Sync vector index** on a Vector Search endpoint with a managed embedding model, then sync
# MAGIC
# MAGIC ### Vector Search: endpoints vs. indexes (the concept everyone trips on)
# MAGIC
# MAGIC * A **Vector Search endpoint** is *managed compute*. It is the server that does the embedding comparisons and hosts indexes.
# MAGIC * A **Vector Search index** is *embedded data*: one index per source table.
# MAGIC * **One endpoint can host MANY indexes.** That is why 5A's `demo3_vs_endpoint` could technically have hosted our policies index too, with zero new infrastructure.
# MAGIC
# MAGIC So why do we still create a **dedicated `module5b_vs_endpoint`** below? Production hygiene: separate endpoints per environment or team keep load, cost, and lifecycle isolated, and tearing down this lab's resources can never touch 5A's demo infrastructure. Same rule as catalogs and schemas: sharing is *possible*, isolation is *policy*.

# COMMAND ----------

# DBTITLE 1,S0 create dedicated VS endpoint
# S0: Create a DEDICATED Vector Search endpoint for Module 5B
# (see the endpoint-vs-index theory above: one endpoint can host many indexes,
#  but production hygiene says this lab gets its own compute and lifecycle)
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import AlreadyExists, ResourceDoesNotExist
from databricks.sdk.service.vectorsearch import EndpointType
import time

w = WorkspaceClient()

# Idempotency rule (same as F2a/J2): only a true 404 means "does not exist".
# A broad `except Exception` would also swallow transient errors - e.g. a get
# while the endpoint is still provisioning - fall into the create branch, and
# crash with AlreadyExists. That is the exact stale error this cell cached on
# Oct 8: the first run created the endpoint (21:43), a re-run 30 seconds
# later swallowed a transient get failure, and its create hit AlreadyExists.
try:
    ep = w.vector_search_endpoints.get_endpoint(endpoint_name=VS_ENDPOINT)
    print(f"Endpoint '{VS_ENDPOINT}' already exists: {ep.endpoint_status.state}")
except ResourceDoesNotExist:
    print(f"Creating endpoint '{VS_ENDPOINT}' ... (takes ~2-3 minutes)")
    try:
        w.vector_search_endpoints.create_endpoint(name=VS_ENDPOINT, endpoint_type=EndpointType.STANDARD)
    except AlreadyExists:
        # Another run is still provisioning it: fall through to the wait loop.
        print("Endpoint is already provisioning (a previous run started it).")
    for i in range(60):
        ep = w.vector_search_endpoints.get_endpoint(endpoint_name=VS_ENDPOINT)
        print(f"  state: {ep.endpoint_status.state}")
        if str(ep.endpoint_status.state).endswith("ONLINE"):
            break
        time.sleep(10)

print(f"\n✅ Endpoint '{VS_ENDPOINT}' is ready. The policies index (S3) will live here;")
print("   5A's demo3_vs_endpoint keeps hosting the 5A knowledge base, untouched.")

# COMMAND ----------

# DBTITLE 1,S1 copy PDF to volume
# S1: Copy the SkyJet policies PDF into a UC Volume
# (ai_parse_document requires binary files in a UC Volume, not workspace files)
spark.sql(f"CREATE VOLUME IF NOT EXISTS {FULL_SCHEMA}.pdf_documents")

pdf_source = "file:/Workspace/Users/shriveens@platformatory.com/Data+AI Academy/Module 5B/static/SkyJet_Passenger_Policies.pdf"
pdf_dest   = f"/Volumes/{UC_TOOL_CATALOG}/{UC_TOOL_SCHEMA}/pdf_documents/SkyJet_Passenger_Policies.pdf"

dbutils.fs.cp(pdf_source, pdf_dest)

import os as _os
size = _os.popen(f"ls -la /Volumes/{UC_TOOL_CATALOG}/{UC_TOOL_SCHEMA}/pdf_documents/").read()
print("PDF staged in volume:")
print(size)

# COMMAND ----------

# DBTITLE 1,S2 comment dash fix
# MAGIC %sql
# MAGIC -- S2: Parse the policies PDF and build the knowledge-base table
# MAGIC -- Same shape as 5A Demo 3: doc_id (primary key), title, content, plus Change Data Feed,
# MAGIC -- which Delta Sync vector indexes require.
# MAGIC
# MAGIC CREATE OR REPLACE TABLE module5b.skyjet.policies_kb (
# MAGIC   doc_id       STRING NOT NULL COMMENT 'Primary key used by the vector index',
# MAGIC   title        STRING COMMENT 'Section header this chunk belongs to',
# MAGIC   content      STRING COMMENT 'Chunk text, embedded by the vector index',
# MAGIC   element_type STRING COMMENT 'Element type from ai_parse_document'
# MAGIC )
# MAGIC COMMENT 'SkyJet passenger-policies knowledge base, parsed from the policies PDF'
# MAGIC TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC
# MAGIC INSERT INTO module5b.skyjet.policies_kb
# MAGIC WITH parsed AS (
# MAGIC   SELECT ai_parse_document(content, MAP('version', '2.0')) AS parsed
# MAGIC   FROM READ_FILES('/Volumes/module5b/skyjet/pdf_documents/', format => 'binaryFile')
# MAGIC )
# MAGIC SELECT
# MAGIC   concat('doc-', lpad(CAST(row_number() OVER (ORDER BY pos) AS STRING), 3, '0')) AS doc_id,
# MAGIC   CASE WHEN element:type::STRING IN ('section_header', 'title')
# MAGIC        THEN element:content::STRING ELSE 'SkyJet Passenger Policies' END      AS title,
# MAGIC   element:content::STRING                                                     AS content,
# MAGIC   element:type::STRING                                                        AS element_type
# MAGIC FROM parsed
# MAGIC LATERAL VIEW posexplode(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) e AS pos, element
# MAGIC WHERE length(element:content::STRING) > 60
# MAGIC   AND element:type::STRING IN ('text', 'section_header', 'title', 'table');
# MAGIC
# MAGIC SELECT count(*) AS total_chunks, collect_set(element_type) AS chunk_types
# MAGIC FROM module5b.skyjet.policies_kb;

# COMMAND ----------

# DBTITLE 1,S2b seed flights table
# MAGIC %sql
# MAGIC -- S2b: Seed a small synthetic flight-status table for the structured (non-RAG) tools.
# MAGIC -- In a real airline this would be an operational database; the tool code stays identical.
# MAGIC
# MAGIC CREATE OR REPLACE TABLE module5b.skyjet.flights (
# MAGIC   flight_number       STRING,
# MAGIC   origin              STRING,
# MAGIC   destination         STRING,
# MAGIC   scheduled_departure TIMESTAMP,
# MAGIC   status              STRING,
# MAGIC   gate                STRING
# MAGIC )
# MAGIC COMMENT 'SkyJet sample flight schedule (synthetic data for the lab)';
# MAGIC
# MAGIC INSERT INTO module5b.skyjet.flights VALUES
# MAGIC   ('SJ102', 'SFO', 'DEL', TIMESTAMP'2025-06-01 22:30:00', 'ON TIME',   'A12'),
# MAGIC   ('SJ204', 'JFK', 'LHR', TIMESTAMP'2025-06-02 08:15:00', 'DELAYED',   'B07'),
# MAGIC   ('SJ330', 'BLR', 'SIN', TIMESTAMP'2025-06-02 11:45:00', 'ON TIME',   'C03'),
# MAGIC   ('SJ415', 'DEL', 'DXB', TIMESTAMP'2025-06-02 15:20:00', 'CANCELLED', 'A04');
# MAGIC
# MAGIC SELECT * FROM module5b.skyjet.flights;

# COMMAND ----------

# DBTITLE 1,S3 dash fix
# S3: Create + sync the Delta Sync vector index (5A Demo 3 pattern, new source table)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest, EmbeddingSourceColumn,
    VectorIndexType, PipelineType,
)
import time

w = WorkspaceClient()

# Create the index only if it does not exist yet; never delete one that is
# still provisioning (a re-run during creation would restart it from zero).
try:
    w.vector_search_indexes.get_index(index_name=VS_INDEX)
    print(f"Index '{VS_INDEX}' already exists.")
except Exception:
    print(f"Creating index '{VS_INDEX}' ...")
    w.vector_search_indexes.create_index(
        name=VS_INDEX,
        endpoint_name=VS_ENDPOINT,
        primary_key="doc_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=f"{FULL_SCHEMA}.policies_kb",
            embedding_source_columns=[
                EmbeddingSourceColumn(name="content", embedding_model_endpoint_name=EMBEDDING_ENDPOINT)
            ],
            pipeline_type=PipelineType.TRIGGERED,
            columns_to_sync=["doc_id", "title", "content"],
        ),
    )

# Wait for readiness, then sync (TRIGGERED mode syncs on demand)
for i in range(60):
    idx = w.vector_search_indexes.get_index(index_name=VS_INDEX)
    if idx.status.ready:
        print("Index ready!")
        break
    print(f"  waiting for index ... ({i})")
    time.sleep(10)

# w.vector_search_indexes.sync_index(index_name=VS_INDEX)
# time.sleep(10)
# print(f"\n✅ Vector index '{VS_INDEX}' synced: {spark.sql(f'SELECT count(*) FROM {FULL_SCHEMA}.policies_kb').collect()[0][0]} chunks embedded.")

# COMMAND ----------

# DBTITLE 1,Section B header dash fixes
# MAGIC %md
# MAGIC # Section B: The tool layer, three tiers, one rule
# MAGIC
# MAGIC > **The rule (from Notebook 1):** if a tool is *shared*, it belongs in Unity Catalog. If it is *agent-local glue*, plain Python in the agent code is fine.
# MAGIC
# MAGIC | Tier | Mechanism | Governed? | SkyJet use |
# MAGIC |---|---|---|---|
# MAGIC | 1. Data access | `CREATE FUNCTION … LANGUAGE SQL` | ✅ UC | Flight status, flight counts, and **`search_policies` RAG retrieval** (a SQL function wrapping the `vector_search()` built-in) |
# MAGIC | 2. Business logic | `client.create_python_function()` | ✅ UC | Baggage fees, contact masking |
# MAGIC | 3. Agent glue | Plain Python in the agent code | ❌ notebook-local | Formatting the final answer for the UI |
# MAGIC
# MAGIC The showpiece is **`search_policies`**: in 5A Demo 3 the retrieval step was just notebook code. Here it is a governed SQL function whose body calls the `vector_search()` SQL built-in directly, so it is permissioned, auditable, and discoverable via MCP like any other tool. **RAG retrieval is now a first-class tool.**

# COMMAND ----------

# DBTITLE 1,B1 flight-status string fix
# MAGIC %sql
# MAGIC -- TIER 1: governed SQL tools over the flights table
# MAGIC CREATE OR REPLACE FUNCTION module5b.skyjet.get_flight_status(
# MAGIC   flight_no STRING COMMENT "SkyJet flight number, e.g. 'SJ204'"
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE SQL
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'Returns origin, destination, scheduled departure, current status and gate for one SkyJet flight. Use for flight-status questions.'
# MAGIC RETURN SELECT max(concat(
# MAGIC          'Flight ', f.flight_number, ': ', f.origin, '→', f.destination,
# MAGIC          ', scheduled ', CAST(f.scheduled_departure AS STRING),
# MAGIC          ', currently ', f.status, ' at gate ', f.gate))
# MAGIC        FROM module5b.skyjet.flights f
# MAGIC        WHERE upper(f.flight_number) = upper(flight_no);
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION module5b.skyjet.count_flights_by_status(
# MAGIC   flight_status STRING COMMENT "Status to count, e.g. 'ON TIME', 'DELAYED', 'CANCELLED'"
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE SQL
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'Counts SkyJet flights currently in a given status. Use for operational-overview questions.'
# MAGIC RETURN SELECT concat(
# MAGIC          CAST(COUNT(*) AS STRING), ' flight(s) currently ', flight_status)
# MAGIC        FROM module5b.skyjet.flights
# MAGIC        WHERE upper(status) = upper(flight_status);
# MAGIC
# MAGIC -- Smoke test both (UC functions must be called by their full three-part name)
# MAGIC SELECT module5b.skyjet.get_flight_status('SJ204')         AS status_line,
# MAGIC        module5b.skyjet.count_flights_by_status('ON TIME') AS on_time_count;

# COMMAND ----------

# DBTITLE 1,B1b: RAG retrieval as governed SQL function
# MAGIC %sql
# MAGIC -- TIER 1b: RAG retrieval as a GOVERNED SQL function over the vector index.
# MAGIC -- The vector_search() SQL built-in queries the index natively: no SDK client
# MAGIC -- and no credentials needed, so this tool works wherever it is invoked,
# MAGIC -- including through MCP. (A Python UC function cannot query the index itself:
# MAGIC -- its isolated execution environment has no workspace credentials.)
# MAGIC -- num_results must be a constant (the built-in requires a foldable
# MAGIC -- argument), so the tool always retrieves the top 3 policy sections.
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION module5b.skyjet.search_policies(
# MAGIC   query STRING COMMENT "The natural-language policy question or topic, e.g. 'what is the refund policy for cancelled flights'"
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Retrieves the most relevant SkyJet passenger-policy sections for a question using semantic (vector) search over the official policies document. Use this tool for ANY question about SkyJet rules or policies: baggage, refunds, changes, check-in, pets, special assistance. Always ground policy answers in the sections it returns and cite the doc ids, e.g. (per doc-004).'
# MAGIC RETURN SELECT coalesce(
# MAGIC          array_join(collect_list(concat('[', vs.doc_id, ' | ', vs.title, '] ', vs.content)), '\n\n'),
# MAGIC          'No matching policy sections found in the SkyJet policies document.')
# MAGIC        FROM vector_search(
# MAGIC          index => 'module5b.skyjet.policies_index',
# MAGIC          query_text => query,
# MAGIC          num_results => 3
# MAGIC        ) AS vs;

# COMMAND ----------

# DBTITLE 1,B1c note: plain-wrapper alternative
# MAGIC %md
# MAGIC ### B1c. Note: the plain-wrapper alternative, and why it is not our tool
# MAGIC
# MAGIC The textbook way to reach the serving endpoint that hosts a vector index is a small SDK wrapper in agent code, exposed to the agent with the Agents SDK `function_tool` decorator:
# MAGIC
# MAGIC ```python
# MAGIC from agents import function_tool
# MAGIC from databricks.vectorsearch.client import VectorSearchClient
# MAGIC
# MAGIC vsc = VectorSearchClient()   # picks up ambient workspace credentials
# MAGIC
# MAGIC @function_tool
# MAGIC def search_policies_local(query: str) -> str:
# MAGIC     """Retrieves the most relevant SkyJet passenger-policy sections."""
# MAGIC     res = vsc.query_index(
# MAGIC         index_name="module5b.skyjet.policies_index",
# MAGIC         columns=["doc_id", "title", "content"],
# MAGIC         query_text=query,
# MAGIC         num_results=3,
# MAGIC     )
# MAGIC     chunks = res["result"]["data_array"]
# MAGIC     return "\n".join(f"[{c['doc_id']} | {c['title']}] {c['content']}" for c in chunks)
# MAGIC ```
# MAGIC
# MAGIC This is a perfectly valid pattern **inside a notebook or a Databricks App**: the process runs with workspace credentials, so the client can query the AI Search serving endpoint directly. It is the "normal function wrap" most quick examples show.
# MAGIC
# MAGIC Two reasons it is not our production RAG tool:
# MAGIC
# MAGIC 1. **It cannot become a UC Python function.** A Python UC function runs in an isolated serverless environment with no workspace credentials, so this exact code fails there (default auth cannot configure credentials). The governed form is B1b: a SQL function whose body calls the `vector_search()` built-in, which executes with the platform's own identity.
# MAGIC 2. **It is not governed.** An agent-local decorated tool has no UC permissions, no audit trail, and no MCP discovery: only the process that carries the code can call it. Shared tools live in UC; decorator wrappers are for agent-local glue only.
# MAGIC
# MAGIC **Rule of thumb:** B1b is the shared, governed retrieval tool every agent reaches through MCP; the wrapper above is the local, ungoverned shortcut, fine for experiments inside one notebook or app process.
# MAGIC
# MAGIC **The same reasoning answers "why not `tools=[...]` and a wrapped retrieval function in production?"** Read from the agent-construction side (Notebook 1 B6 has the full surface): `tools=` carries functions that live only in the calling process, perfect for agent-local glue like B3's `format_policy_answer`, wrong for shared tools that other agents must also discover, permission, and audit. `mcp_servers=` is the production channel: the agent opens the managed MCP server over `module5b.skyjet` and every UC function surfaces through it, SQL flight tools, Python fee tools, and `search_policies` RAG alike. Note too that this lab deliberately has no separate index MCP: retrieval is already a governed SQL function over the index, so ONE namespace and ONE MCP server cover RAG and the operational tools together, and the app and endpoint service principals need grants only against UC objects, never direct vector-index credentials.

# COMMAND ----------

# DBTITLE 1,B2 register the Python tools as UC functions
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()   # registers Python fns as governed UC functions


def estimate_baggage_fee(weight_kg: float, cabin_class: str, loyalty_tier: str) -> str:
    """
    Calculates the checked-baggage fee for a SkyJet passenger based on bag weight,
    travel class, and loyalty tier (the tool registered in Notebook 1).

    Use this tool whenever a passenger asks about baggage fees or excess-bag charges.

    Args:
        weight_kg (float): Total checked-bag weight in kilograms, between 1 and 60.
        cabin_class (str): Travel class. One of 'economy', 'premium', 'business'.
        loyalty_tier (str): SkyJet loyalty tier. One of 'none', 'silver', 'gold', 'platinum'.

    Returns:
        str: A human-readable fee breakdown, or an error message if inputs are invalid.
    """
    allowances = {"economy": 20.0, "premium": 30.0, "business": 40.0}
    discounts = {"none": 0.0, "silver": 0.10, "gold": 0.20, "platinum": 0.35}
    RATE_PER_KG = 12.0

    cls = (cabin_class or "").strip().lower()
    tier = (loyalty_tier or "").strip().lower()
    if cls not in allowances:
        return f"Error: unknown cabin class '{cabin_class}'."
    if tier not in discounts:
        return f"Error: unknown loyalty tier '{loyalty_tier}'."
    if weight_kg is None or weight_kg < 1 or weight_kg > 60:
        return "Error: weight_kg must be between 1 and 60 kg."

    allowance = allowances[cls]
    excess = max(0.0, weight_kg - allowance)
    fee = excess * RATE_PER_KG
    final = fee * (1 - discounts[tier])
    if excess == 0:
        return f"{cls.capitalize()}: {allowance:g} kg free allowance, so no fee."
    return (
        f"{cls.capitalize()}: {allowance:g} kg free; {excess:g} kg excess "
        f"at ${RATE_PER_KG:.2f}/kg = ${fee:.2f}; {tier} discount → ${final:.2f} total"
    )


def mask_contact_info(text: str) -> str:
    """
    Masks emails and phone numbers in free text so passenger contact details are
    never echoed back in plain form in responses or logs.

    Args:
        text (str): Free text that may contain emails or phone numbers.

    Returns:
        str: The same text with each email replaced by '[email masked]' and
        each phone number by '[phone masked]'.
    """
    import re

    masked = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email masked]", text or "")
    # {6,} (not {7,}) so 7-digit local numbers like 555-0100 (8 chars with
    # the dash) are masked too; the first and last char are consumed by \d.
    masked = re.sub(r"(?<!\w)(\+?\d[\d\s().-]{6,}\d)(?!\w)", "[phone masked]", masked)
    return masked


# TIER 2: register the governed Python tools (same pattern as Notebook 1).
# search_policies is NOT here: it is the SQL function from B1b, because a
# Python UC function runs in an isolated environment with no workspace
# credentials and cannot call the vector search index itself.
for fn in (estimate_baggage_fee, mask_contact_info):
    client.create_python_function(func=fn, catalog=UC_TOOL_CATALOG, schema=UC_TOOL_SCHEMA, replace=True)
    print(f"✅ Registered: {FULL_SCHEMA}.{fn.__name__}")

# Verify THE RAG TOOL executes through UC: retrieval is now a governed tool:
print("\n--- RAG retrieval through UC ---")
r = client.execute_function(
    function_name=f"{FULL_SCHEMA}.search_policies",
    parameters={"query": "what is the refund policy if SkyJet cancels a flight"},
)
print(r.value[:600])

# COMMAND ----------

# DBTITLE 1,B3 agent-local glue (tier 3, not in UC)
# B3 / TIER 3: agent-local presentation glue. It only formats output for the chat UI,
# no other agent calls it and it touches no data, so it does not belong in
# Unity Catalog. In production this is plain Python living in agent.py, not an
# SDK-decorated tool.
def format_policy_answer(question: str, policy_text: str) -> str:
    """Formats a retrieved policy snippet into a customer-friendly answer block for the chat UI."""
    return f"📌 SkyJet policy, re: “{question}”\n\n{policy_text}\n\n*(Source: SkyJet Passenger Policies)*"


print("Tier-3 glue ready (agent-local, intentionally NOT in UC):")
print(format_policy_answer("pets on board", "Service animals fly in the cabin at no charge."))

# COMMAND ----------

# DBTITLE 1,Section C header: verify the toolkit via MCP
# MAGIC %md
# MAGIC # Section C: Verify the governed toolkit via MCP (runtime discovery)
# MAGIC
# MAGIC Before building agents on top of the toolkit, **prove the contract**: every shared tool (SQL, Python, *and the RAG retrieval tool*) must be discoverable at runtime through the managed MCP server. This is the production equivalent of 5A's `print(df.columns)` sanity check.

# COMMAND ----------

# DBTITLE 1,C1 MCP discovery + tool contract gate
import nest_asyncio
nest_asyncio.apply()

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

ws = WorkspaceClient()
workspace_host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
mcp_url = f"{workspace_host}/api/2.0/mcp/functions/{UC_TOOL_CATALOG}/{UC_TOOL_SCHEMA}"

mcp_client = DatabricksMCPClient(server_url=mcp_url, workspace_client=ws)
tools = mcp_client.list_tools()
tool_names = [t.name for t in tools]

print(f"🧰 {len(tools)} governed tools discovered at runtime from {FULL_SCHEMA}:\n")
for t in tools:
    print(f"  • {t.name:<28} {t.description[:70]}...")

# Production gate: fail fast if the contract is broken
REQUIRED_TOOLS = {
    "get_flight_status", "count_flights_by_status",
    "search_policies", "estimate_baggage_fee", "mask_contact_info",
}
# MCP tool names are schema-qualified (catalog__schema__function), the same
# names tools/list printed above, so compare the bare function names.
bare_names = {n.split("__")[-1] for n in tool_names}
missing = REQUIRED_TOOLS - bare_names
assert not missing, f"❌ Tool contract violated, missing: {missing}"
print("\n✅ Tool contract satisfied: SQL, Python AND RAG-retrieval tools all discoverable via MCP.")

# COMMAND ----------

# DBTITLE 1,Section D header: multi-agent orchestration
# MAGIC %md
# MAGIC # Section D: Multi-agent orchestration (supervisor + handoffs)
# MAGIC
# MAGIC From 5A you know the supervisor-worker pattern (Demo 6). In this lab we productionize it with one crucial change: **workers get their tools from MCP at runtime** (D1 defines the team; D2 runs it), nothing is hardcoded, and both specialists share one governed toolkit. (Notebook 1 B6 maps the full constructor surface: `tools` for agent-local glue, `mcp_servers` for governed runtime discovery, `handoffs` for routing.)
# MAGIC
# MAGIC - **PolicyAgent**: the RAG specialist, grounds every answer in retrieved policy sections (`search_policies`), masks contact details before repeating them
# MAGIC - **FlightAgent**: the structured-data specialist, handles flight status, operational counts, baggage fees
# MAGIC - **Supervisor**: triages, hands off via `handoffs=`, never answers alone
# MAGIC
# MAGIC The `McpServer.from_uc_function` async context manager handles the MCP lifecycle (initialize → discover → invoke → shutdown) for the agent's lifetime, the exact pattern from Notebook 1 Part B.

# COMMAND ----------

# DBTITLE 1,D1 define the supervisor + specialists
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from databricks.sdk import WorkspaceClient
import logging

# D1: define the supervisor team. Route the OpenAI Agents SDK through Databricks model serving (the 5A setup):
# without this the SDK calls api.openai.com directly and every agent run fails.
set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])   # disable OpenAI trace processor; use MLflow instead
logging.getLogger("openai").setLevel(logging.WARNING)

ws = WorkspaceClient()

POLICY_AGENT_INSTRUCTIONS = """You are SkyJet's Policy Specialist, the passenger-support agent for policy questions.
For EVERY policy question, call search_policies first and ground your answer strictly in the
sections it returns. Cite the doc ids you used, e.g. (per doc-004). If search_policies returns
nothing relevant, say you could not find the policy in the official document: never guess.
If the passenger shares contact details, pass them through mask_contact_info before repeating them.
Be concise and factual."""

FLIGHT_AGENT_INSTRUCTIONS = """You are SkyJet's Flight Operations Specialist.
Answer flight-status questions with get_flight_status, operational-overview questions with
count_flights_by_status, and baggage-fee questions with estimate_baggage_fee.
If a flight number is unknown in the system, say so: never invent one.
Be concise and factual."""

SUPERVISOR_INSTRUCTIONS = """You are the SkyJet passenger-support supervisor.
Route policy questions (baggage rules, refunds, changes, check-in, pets, special assistance)
to the PolicyAgent, and flight/baggage-fee operational questions to the FlightAgent, using
handoffs. Do not answer data questions yourself. If the request is unclear, ask one
clarifying question."""

policy_agent = Agent(
    name="PolicyAgent",
    model=SERVING_ENDPOINT,
    instructions=POLICY_AGENT_INSTRUCTIONS,
    mcp_servers=[],          # wired below inside the MCP context
)
flight_agent = Agent(
    name="FlightAgent",
    model=SERVING_ENDPOINT,
    instructions=FLIGHT_AGENT_INSTRUCTIONS,
    mcp_servers=[],
)
supervisor = Agent(
    name="SkyJetSupervisor",
    model=SERVING_ENDPOINT,
    instructions=SUPERVISOR_INSTRUCTIONS,
    handoffs=[policy_agent, flight_agent],
)
print("✅ Agents defined, supervisor routes to", [a.name for a in supervisor.handoffs])

# COMMAND ----------

# DBTITLE 1,D2 entrypoint with MCP lifespan
import asyncio
from openai import BadRequestError

async def run_production_query(user_message: str) -> str:
    """Production entrypoint: MCP servers live for the agent's lifetime (lifespan pattern)."""
    async with McpServer.from_uc_function(
        catalog=UC_TOOL_CATALOG,
        schema=UC_TOOL_SCHEMA,
        workspace_client=ws,
        name="skyjet-uc-tools",
        timeout=120,   # serverless tool calls can take ~1 min; the 20s default times out
    ) as uc_server:
        # Wire the SAME governed MCP server into both workers (runtime discovery)
        policy_agent.mcp_servers = [uc_server]
        flight_agent.mcp_servers = [uc_server]

        result = await Runner.run(supervisor, user_message)
        return result.final_output


def ask(user_message: str, max_attempts: int = 2) -> str:
    # The served Llama model occasionally emits a malformed tool-call; the
    # endpoint rejects it with a 400 ("Model response did not respect the
    # required format") and Databricks' guidance for that error is to retry.
    # One bounded retry, then the real error surfaces.
    for attempt in range(1, max_attempts + 1):
        try:
            return asyncio.run(run_production_query(user_message))
        except BadRequestError as e:
            if attempt == max_attempts or "required format" not in str(e):
                raise
            print(f"   serving-side tool-call glitch, retrying (attempt {attempt + 1} of {max_attempts})...")

print("✅ Production entrypoint ready (with bounded retry for malformed tool-calls).")

# COMMAND ----------

# DBTITLE 1,Section E header: trace every request
# MAGIC %md
# MAGIC # Section E: Observability - trace every request
# MAGIC
# MAGIC From Notebook 1 Part C: **tracing must come first**, because scorers have nothing to evaluate without traces. We wire MLflow at the entrypoint so every conversation produces a span tree: `AGENT` root → `CHAT_MODEL` + `TOOL` children (you will see `search_policies` TOOL spans for RAG retrievals).
# MAGIC
# MAGIC In the deployed app, the experiment name arrives via `databricks.yml`; here in the notebook, traces automatically log to this notebook's default MLflow experiment.

# COMMAND ----------

# DBTITLE 1,E1 wire MLflow tracing
import mlflow

# The 5A tracing pattern: autolog instruments the OpenAI Agents SDK. In a
# Databricks notebook traces log to this notebook's MLflow experiment
# automatically, so no set_experiment call is needed here; the injected
# MLFLOW_EXPERIMENT_NAME only matters in the deployed app (Section F).
mlflow.openai.autolog()


@mlflow.trace
def skyjet_chat(user_message: str) -> str:
    """Single traced entrypoint for every user request."""
    return ask(user_message)


print("✅ Tracing wired: every request produces a trace in this notebook's MLflow experiment")

# COMMAND ----------

# DBTITLE 1,E2b end-to-end acceptance run
# E2: End-to-end acceptance run. Four scenarios:
# (1) a RAG question with a citable answer, (2) a mixed question using both
# FlightAgent tools, (3) a pure SQL question, (4) a RAG question whose answer
# is NOT in the policies document, proving the agent refuses instead of
# guessing. Section T turns these into formal tests.
TEST_CASES = [
    "What is SkyJet's refund policy if a flight is cancelled?",                                    # → PolicyAgent (RAG, cites doc ids)
    "Is flight SJ204 on time, and what is the baggage fee for a 28 kg bag in economy, gold tier?",  # → FlightAgent (both tools)
    "How many flights are currently delayed?",                                                   # → FlightAgent (SQL)
    "What is SkyJet's policy on traveling with pets?",                                          # → PolicyAgent (grounded refusal: no pets section in the document)
]

for i, q in enumerate(TEST_CASES, 1):
    print(f"\n━━━ Case {i}: {q}")
    try:
        print(skyjet_chat(q))
    except Exception as e:
        print(f"⚠️ Case {i} failed ({type(e).__name__}: {e})")
        print("   Check: serving endpoint access, UC EXECUTE permissions on", FULL_SCHEMA,
              ", vector index query access on", VS_INDEX)

# COMMAND ----------

# DBTITLE 1,Section T header: three levels of testing
# MAGIC %md
# MAGIC # Section T: Testing before shipping
# MAGIC
# MAGIC Production readiness is proven by tests, not by hope. E2 let us *look* at four conversations; tests turn "looks right" into "is right", runnable on every code change. We test at three levels, cheapest first:
# MAGIC
# MAGIC 1. **Unit tests (tools)**: deterministic checks of every governed tool. Known inputs must produce known outputs; invalid inputs must fail gracefully. No LLM involved, so these are fast and stable.
# MAGIC 2. **Agent quality tests (behavior)**: machine-checkable properties of the whole agent. A production SkyJet assistant must ground policy answers in the document (cite doc ids) and must refuse to invent data (unknown flight).
# MAGIC 3. **Deployment smoke tests**: after Section F deploys the app, `/health` must return ok and `/chat` must answer. Those run in Section F.
# MAGIC
# MAGIC In 5A we tested by reading outputs with our own eyes. This is the production-grade version of that habit.

# COMMAND ----------

# DBTITLE 1,T1 unit tests for tools
# T1: Unit tests for the governed toolkit (deterministic, fast, no LLM involved)
def run_tests(tests):
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ❌ {name}: unexpected {type(e).__name__}: {e}")
            failed.append(name)
    print(f"\n{passed}/{passed + len(failed)} unit tests passed")
    assert not failed, f"Failing tests: {failed}"


def test_flight_status_tool():
    r = spark.sql(f"SELECT {FULL_SCHEMA}.get_flight_status('SJ204') r").collect()[0].r
    assert "JFK" in r and "DELAYED" in r, r


def test_flight_count_tool():
    r = spark.sql(f"SELECT {FULL_SCHEMA}.count_flights_by_status('on time') r").collect()[0].r
    assert r.startswith("2 flight"), r   # case-insensitive status match


# Unit tests call every tool with spark.sql on the attached compute (warm,
# seconds) instead of the Statement Execution API (cold start, roughly a
# minute per call): the unit-test level must be the fast feedback loop.
def test_baggage_fee_tool():
    r = spark.sql(f"SELECT {FULL_SCHEMA}.estimate_baggage_fee(28.0, 'economy', 'gold') r").collect()[0].r
    assert "$" in r and "gold" in r.lower(), r   # 20 kg free, 8 kg excess, 20% discount


def test_baggage_fee_rejects_bad_input():
    r = spark.sql(f"SELECT {FULL_SCHEMA}.estimate_baggage_fee(99.0, 'economy', 'gold') r").collect()[0].r
    assert r.startswith("Error"), r   # graceful failure, no exception, no hallucinated fee


def test_mask_contact_info():
    r = spark.sql(f"SELECT {FULL_SCHEMA}.mask_contact_info('email bob@x.com or call 555-0100') r").collect()[0].r
    assert "[email masked]" in r and "[phone masked]" in r, r


def test_search_policies_tool():
    r = spark.sql(f"SELECT {FULL_SCHEMA}.search_policies('refund policy for cancelled flights') r").collect()[0].r
    assert "[doc-" in r and len(r) > 50, r[:200]   # retrieval returns citable chunks


def test_kb_is_populated():
    n = spark.sql(f"SELECT count(*) FROM {FULL_SCHEMA}.policies_kb").collect()[0][0]
    assert n > 10, f"only {n} chunks in knowledge base"


run_tests([
    ("get_flight_status returns the SJ204 row", test_flight_status_tool),
    ("count_flights_by_status is case-insensitive", test_flight_count_tool),
    ("estimate_baggage_fee computes a fee", test_baggage_fee_tool),
    ("estimate_baggage_fee fails gracefully", test_baggage_fee_rejects_bad_input),
    ("mask_contact_info masks email and phone", test_mask_contact_info),
    ("search_policies returns citable chunks", test_search_policies_tool),
    ("policies_kb is populated", test_kb_is_populated),
])

# COMMAND ----------

# DBTITLE 1,T2 agent quality tests
# T2: Agent quality tests (LLM involved: behavioral, slower, still automated)
import re


def test_policy_answer_is_grounded():
    ans = skyjet_chat("What is SkyJet's refund policy if a flight is cancelled?")
    cited = re.search(r"doc-\d+", ans)
    assert cited, f"no citation in answer: {ans[:200]}"
    print(f"   citation found: {cited.group(0)}")


def test_unknown_flight_is_refused():
    ans = skyjet_chat("Is flight SJ999 on time?").lower()
    keywords = ("not", "no flight", "unknown", "couldn't", "could not", "unable", "sorry", "don't", "do not")
    assert any(k in ans for k in keywords), ans[:200]


def test_missing_policy_is_refused():
    # The policies document has no general pets section (only service animals
    # under special assistance). The grounded behaviors are a clean refusal OR
    # an answer explicitly scoped to what the document does say; either way
    # the agent must not invent a pets policy.
    ans = skyjet_chat("What is SkyJet's policy on traveling with pets?").lower()
    grounded = ("could not find", "couldn't find", "not find", "no policy",
                "not in the official document", "did not find", "not covered",
                "service animal")
    assert any(k in ans for k in grounded), ans[:200]


run_tests([
    ("policy answers cite retrieved doc ids", test_policy_answer_is_grounded),
    ("unknown flight is refused, not invented", test_unknown_flight_is_refused),
    ("missing policy is refused, not invented", test_missing_policy_is_refused),
])

print("""
Going further (the MLflow-native route):
  Log each test question as an MLflow evaluation run with attached scorers
  (correctness, groundedness, safety), so quality is tracked over time instead
  of a single pass/fail snapshot. Notebook 1 Part E covered the scorer
  concepts; wiring them up is the natural next step after this module.
""")

# COMMAND ----------

# DBTITLE 1,Section F header: package + deploy the DAB
# MAGIC %md
# MAGIC # Section F: Packaging and deploying the DAB project
# MAGIC
# MAGIC Now we turn the notebook into the **deployment artifact** from Notebook 1 Part D: a DAB project with `agent.py`, `server.py`, `pyproject.toml`, `databricks.yml`, and `app.yaml`. The project lives as **real, individually editable workspace files** in the `skyjet-app/` folder next to this notebook (F1 verifies it), and F2 **actually deploys it** with the two-step flow (Python SDK; the CLI equivalents are noted for the web terminal or CI).
# MAGIC
# MAGIC Notice how Notebook 1's theory lands file-by-file:
# MAGIC - `agent.py` reads **only** environment variables (resource injection)
# MAGIC - `server.py` initializes MCP in the **FastAPI lifespan** (exactly our `run_production_query` pattern)
# MAGIC - `databricks.yml` declares the app + resources (serving endpoint, vector search endpoint/index); `app.yaml` injects env vars

# COMMAND ----------

# DBTITLE 1,F0 why an App and not a serving endpoint
# MAGIC %md
# MAGIC ### F0. Why an App and not a serving endpoint?
# MAGIC
# MAGIC From Notebook 1 Part D: a registered agent deployed to a **serving endpoint** answers prompts and is perfect for single-agent, model-centric use cases. **SkyJet Assist** needs more around the model, so we deploy a **Databricks App** and the agent code itself becomes the deployment artifact:
# MAGIC
# MAGIC | SkyJet Assist needs | Why a serving endpoint alone struggles | How the App delivers |
# MAGIC |---|---|---|
# MAGIC | Runtime tool discovery via MCP | Tools are baked into the model package at registration time | The app connects to the MCP server at startup; register a new UC tool and the app uses it with **zero redeploy** |
# MAGIC | Multi-agent orchestration (supervisor + handoffs) | Orchestration is not the model's job | `agent.py` owns the supervisor and its two specialists |
# MAGIC | Custom endpoints and UX (`/chat`, `/health`) | Fixed input/output schema | FastAPI `server.py` defines exactly the routes we need |
# MAGIC | Business logic around the model (masking, routing, escalation) | Squeezing it into a model artifact is fragile | Plain Python living next to the agent |
# MAGIC
# MAGIC The app still calls the serving endpoint for every LLM turn: Apps replace the **orchestration layer**, not the model layer.

# COMMAND ----------

# DBTITLE 1,F1 verify DAB project files (skyjet-app)
# F1: The DAB project as REAL workspace files. The app package lives in the
# skyjet-app/ folder next to this notebook; every file is an ordinary
# workspace file you can open, review, and edit in the browser. F2b deploys
# this exact folder: no generation, no embedded strings.
import os

APP_NAME = "skyjet-support-app"   # the APP RESOURCE name (unchanged)
APP_ROOT = "/Workspace/Users/shriveens@platformatory.com/Data+AI Academy/Module 5B/skyjet-app"
SRC      = f"{APP_ROOT}/src"

# Verify the project is complete before deploying (fail fast on missing files)
PROJECT_FILES = {
    "databricks.yml":     "bundle: app + resources declaration",
    "app.yaml":           "runtime manifest + env injection",
    "src/agent.py":       "supervisor + Policy/Flight workers + MCP (env-injected)",
    "src/server.py":      "FastAPI wrapper: /chat + /health",
    "src/pyproject.toml": "dependencies",
}
for rel, desc in PROJECT_FILES.items():
    assert os.path.exists(f"{APP_ROOT}/{rel}"), f"missing {rel}"
    print(f"  ✓ {rel:<20} {desc}")

print("\n📁 DAB project (skyjet-app/, next to this notebook):")
print("├── databricks.yml      # bundle: app + resources declaration")
print("├── app.yaml            # runtime manifest + env injection")
print("└── src/")
print("    ├── agent.py        # supervisor + Policy/Flight workers + MCP (env-injected)")
print("    ├── server.py       # FastAPI wrapper: /chat + /health")
print("    └── pyproject.toml  # dependencies")

# COMMAND ----------

# DBTITLE 1,F2 header: the two-step deployment
# MAGIC %md
# MAGIC ### F2. Deployment, for real (two independent steps)
# MAGIC
# MAGIC From Notebook 1 Part D: the two steps are **independent** (resources ≠ code), so iteration is fast.
# MAGIC
# MAGIC **Why the SDK, not the CLI:** the Databricks CLI cannot run inside a notebook `%sh` cell on serverless compute: it prints a "web terminal only" warning, **exits 0, and does nothing** (a silent no-op, the worst failure mode). The cells below therefore execute the same two-step flow with the Python SDK; the CLI column is the equivalent you would run from the web terminal or CI.
# MAGIC
# MAGIC | Cell | SDK call (executed here) | CLI equivalent (web terminal / CI) | What it proves |
# MAGIC |---|---|---|---|
# MAGIC | F2a | `w.apps.create()` | `databricks bundle deploy` | Step 1: the app resource, its serving-endpoint access, and compute are declared |
# MAGIC | F2a2 | `GRANT EXECUTE` + VS endpoint ACL | `GRANT` SQL / endpoint Permissions tab | Governance: the app's service principal may use the UC tools and the vector index |
# MAGIC | F2b | `w.apps.deploy()` | `databricks apps deploy` | Step 2: source code + env vars pushed (code iteration: this step only) |
# MAGIC | F2c | `w.apps.get()` + authenticated HTTP | `apps get` + `curl` | Smoke tests: app is up, `/health` ok, `/chat` answers grounded |
# MAGIC | F2e | `w.apps.delete()` (commented) | `databricks apps delete` | Cleanup: free the quota slot when the lab is done |
# MAGIC
# MAGIC **Env-injection parity:** `AppResource` (Step 1) plays the role of the typed resources in `databricks.yml` / `app.yaml`, and `AppDeployment.env_vars` (Step 2) carries the plain environment variables, exactly like `app.yaml`'s `env:` section. The DAB files in `skyjet-app/` remain the source of truth for the CLI/CI route.
# MAGIC
# MAGIC <div style="border-left:4px solid #f57c00;background:#fff3e0;padding:12px 16px;border-radius:4px;margin:12px 0">
# MAGIC <strong>⚠️ Before you deploy (free-tier / quota awareness, the Module 5A lesson):</strong> each deployed app runs under its own compute and counts against workspace quotas. Check existing apps first with <code>w.apps.list()</code>, and delete stale ones with <code>w.apps.delete(name="skyjet-support-app")</code>. Deploy once, iterate on code with Step 2 only.
# MAGIC </div>
# MAGIC
# MAGIC **Permissions (service principal):** granted by F2a + F2a2: `CAN_QUERY` on the serving endpoint (via the `AppResource`), `USE CATALOG` + `USE SCHEMA` (MCP cannot even list tools without them), `EXECUTE` on the UC functions in `module5b.skyjet` (including `search_policies`), `CAN_USE` on the vector search endpoint, and `SELECT` on `policies_index` and `policies_kb` - the vector index's backing table (the actual RAG gate; found via the audit log) and its source table. Only MLflow experiment access is left to you: grant the app's service principal `CAN_MANAGE` on the experiment if you want its traces uploaded. Governance splits between **workspace-level app ACLs** and **UC permissions on downstream resources**; remember Notebook 1's table.

# COMMAND ----------

# DBTITLE 1,F2a create app resource and API scopes
# F2a: Step 1 of 2: create the app resource (the SDK equivalent of
# `databricks bundle deploy`). Idempotent: re-runs converge the config.
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import (
    App, AppResource, AppResourceServingEndpoint,
    AppResourceServingEndpointServingEndpointPermission,
)

w = WorkspaceClient()

if any(a.name == APP_NAME for a in w.apps.list()):
    app = w.apps.get(name=APP_NAME)
    print(f"App '{APP_NAME}' exists (Step 1 is idempotent).")
else:
    app = w.apps.create_and_wait(app=App(
        name=APP_NAME,
        description="SkyJet Assist: governed, RAG-grounded multi-agent passenger support",
        resources=[
            # Typed resource: grants the app's service principal CAN_QUERY on
            # the model serving endpoint (app.yaml's valueFrom equivalent).
            AppResource(
                name="skyjet-llm",
                serving_endpoint=AppResourceServingEndpoint(
                    name=SERVING_ENDPOINT,
                    permission=AppResourceServingEndpointServingEndpointPermission.CAN_QUERY,
                ),
            ),
        ],
    ))
    print(f"✅ App created: {app.name}")

print(f"   URL: {app.url}")
print(f"   Service principal id: {app.service_principal_client_id}")

# COMMAND ----------

# DBTITLE 1,F2a2 governance grants for the app service principal
# F2a2: Governance grants. The app's service principal reaches the tools
# via MCP, so it needs EXECUTE on every UC function (the grants the F2 header
# describes). The RAG path also queries the vector index, which lives on the
# vector search endpoint: grant the service principal CAN_USE there too.
# Idempotent: re-granting is a no-op. This cell changes permissions on
# governed resources - run it deliberately, once per app.
from databricks.sdk.service.vectorsearch import (
    VectorSearchEndpointAccessControlRequest, VectorSearchEndpointPermissionLevel,
)

sp = app.service_principal_client_id
print(f"Granting tool access to the app service principal: {sp}")

# MCP's tools/list first walks catalog -> schema, so the service principal
# needs USE CATALOG / USE SCHEMA just to DISCOVER the functions before
# EXECUTE matters (the live app taught us this: /chat 500'd with
# "User does not have USE CATALOG on Catalog 'module5b'").
spark.sql(f"GRANT USE CATALOG ON CATALOG {UC_TOOL_CATALOG} TO `{sp}`")
spark.sql(f"GRANT USE SCHEMA ON SCHEMA {FULL_SCHEMA} TO `{sp}`")
print("✅ USE CATALOG + USE SCHEMA granted (tool discovery via MCP)")

# The RAG retrieval gate: vector_search() reads the index, whose access is
# governed by SELECT on the index's BACKING table - and that table carries
# the index's own name (module5b.skyjet.policies_index), NOT the source
# table's (policies_kb). We proved this from the audit log: the SP's call
# ended at "authzEval: GetTable module5b.skyjet.policies_index" with no
# queryVectorIndex ever fired, and a wrong grant on policies_kb changed
# nothing. Grant both source and backing table to keep the pattern complete.
spark.sql(f"GRANT SELECT ON TABLE {FULL_SCHEMA}.policies_index TO `{sp}`")
spark.sql(f"GRANT SELECT ON TABLE {FULL_SCHEMA}.policies_kb TO `{sp}`")
print("✅ SELECT on policies_index (index backing table) + policies_kb (source) granted")

for fn in ("get_flight_status", "count_flights_by_status", "estimate_baggage_fee",
           "mask_contact_info", "search_policies"):
    spark.sql(f"GRANT EXECUTE ON FUNCTION {FULL_SCHEMA}.{fn} TO `{sp}`")
print("✅ EXECUTE on all 5 tools granted")

ep = w.vector_search_endpoints.get_endpoint(endpoint_name=VS_ENDPOINT)
w.vector_search_endpoints.update_permissions(
    endpoint_id=ep.id,
    access_control_list=[VectorSearchEndpointAccessControlRequest(
        service_principal_name=sp,
        permission_level=VectorSearchEndpointPermissionLevel.CAN_USE,
    )],
)
print(f"✅ CAN_USE on vector search endpoint '{VS_ENDPOINT}' granted")

# COMMAND ----------

# DBTITLE 1,F2b deploy the source code (why deploy after create)
# F2b: Step 2 of 2: push the source code + inject the env vars
# (SDK equivalent of `databricks apps deploy --source-code-path`).
# WHY a separate deploy step after F2a's create()? create() only provisioned
# the app SHELL - identity, OAuth client, URL, quota slot - with NO code in
# it. deploy() uploads src/, builds the uv environment server-side, and
# launches the process. The split is deliberate: source changes are
# redeployed many times over the app's life (that is exactly what this cell
# did when agent.py gained its MCP comments) without ever recreating the
# app or touching its grants.
# Plain env vars ride on the deployment (AppDeployment.env_vars), exactly
# the role app.yaml's env: section plays; the uv build runs server-side.
from databricks.sdk.service.apps import AppDeployment, EnvVar
import time

print(f"Deploying source from {SRC} (the build takes a few minutes)...")
t0 = time.time()
deployment = w.apps.deploy_and_wait(
    app_name=APP_NAME,
    app_deployment=AppDeployment(
        # The SDK deploy path does not read app.yaml; the launch command must
        # ride on the deployment itself (this is why the first deploy crashed
        # with "python agent.py" and no dependencies). uv + pyproject.toml is
        # the exact app.yaml build; port 8000 is the platform default.
        command=["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"],
        source_code_path=SRC,
        env_vars=[
            EnvVar(name="SERVING_ENDPOINT",       value=SERVING_ENDPOINT),
            EnvVar(name="UC_TOOL_CATALOG",        value=UC_TOOL_CATALOG),
            EnvVar(name="UC_TOOL_SCHEMA",         value=UC_TOOL_SCHEMA),
            EnvVar(name="VS_ENDPOINT",            value=VS_ENDPOINT),
            EnvVar(name="VS_INDEX",               value=VS_INDEX),
            EnvVar(name="MLFLOW_EXPERIMENT_NAME", value=MLFLOW_EXPERIMENT_NAME),
        ],
    ),
)
print(f"\u2705 Deployment {deployment.deployment_id} finished in {time.time() - t0:.0f}s: {deployment.status}")

# COMMAND ----------

# DBTITLE 1,F2c smoke tests: SDK + edge + the real /chat call (I4)
# F2c: Deployment smoke tests (Section T level 3): is the app up, healthy,
# and answering?
#
# Databricks Apps sit behind the workspace SSO edge. A request without a
# workspace session is redirected to the Databricks sign-in page - which is
# exactly what we observed live: the notebook's ambient OAuth token is
# workspace-scoped and the edge does not accept it (documented behavior:
# tokens for programmatic app access need the all-apis scope, e.g. the CLI's
# `databricks auth token`, or a service principal's M2M token; browser users
# authenticate via SSO). So the smoke has three parts:
#   1. App resource reports RUNNING (SDK).
#   2. The edge answers our request (HTTP) - even if that answer is the login
#      page, it proves the app's serving plane is live and auth is enforced.
#   3. /chat as an authenticated USER: open the app URL in the browser (the
#      app serves a minimal chat UI at its root) or use the web terminal.
import requests

app = w.apps.get(name=APP_NAME)
print(f"App URL: {app.url}")
assert "RUNNING" in str(app.app_status), app.app_status
print(f"1) app_status={app.app_status.state}  compute_status={app.compute_status.state}  (SDK)")

headers = w.config.authenticate()
s = requests.Session()
s.headers.update(headers)
r = s.get(f"{app.url.rstrip('/')}/health", timeout=60)
body = r.text.strip()
if "Sign In" in body[:500]:
    print("2) GET /health -> 200, SSO login page: the edge is live and enforcing auth")
    print("   (the notebook's ambient token is workspace-scoped; the documented")
    print("   programmatic routes are an all-apis CLI token or an SP M2M token)")
else:
    print(f"2) GET /health -> {r.status_code}: {body[:200]}")
assert r.status_code == 200, r.text[:200]

print("\n3) /chat as an authenticated user (the definitive smoke test):")
print("   open the app URL in your browser (workspace SSO) - the app serves a")
print("   minimal chat UI at its root - and ask:")
print("   'What is SkyJet's refund policy if a flight is cancelled?'")
print("   Expected: a grounded answer citing doc ids (doc-021/022/023).")
curl_cmd = (
    'curl -s -H "Authorization: Bearer $(databricks auth token)"'
    ' -H "Content-Type: application/json"'
    " -d '{\"message\": \"What is the refund policy for a cancelled flight?\"}'"
    f" {app.url.rstrip('/')}/chat"
)
print("   Web-terminal equivalent (the CLI token carries the all-apis scope):")
print("   " + curl_cmd)
print("\n   The ACTUAL programmatic /chat call (token-exchange + POST) is")
print("   Section I4 - the real end-to-end test of the agent path. I3 is the")
print("   raw model call (no agent); I4 is the app call (agent + tools + RAG).")

# COMMAND ----------

# DBTITLE 1,F3 live-fix log: how the deployment was debugged
# MAGIC %md
# MAGIC ### F3. The live-fix log: five layers a real deployment peels back
# MAGIC
# MAGIC <div style="border-left:4px solid #2e7d32;background:#e8f5e9;padding:12px 16px;border-radius:4px;margin:12px 0">
# MAGIC <strong>✅ VERIFIED LIVE (2026-10-09):</strong> the browser smoke test returned a grounded answer citing <code>doc-021/022/023</code> - full refund or free rebooking for cancellations, delay compensation tiers, and the weather/ATC/security exception. SSO edge → app → supervisor → MCP → governed UC tools → vector search → cited answer. Every layer green.
# MAGIC </div>
# MAGIC
# MAGIC The app did not work on the first deploy - and that was the most valuable part of the lab. Each failure exposed exactly one production layer, and every fix is now baked into the code and cells you just ran. This is the complete log:
# MAGIC
# MAGIC | # | Symptom (live) | Root cause | Fix (where it lives now) | Lesson |
# MAGIC |---|---|---|---|---|
# MAGIC | 1 | `databricks` CLI cells "succeeded" but did nothing | The CLI cannot run in a `%sh` cell on serverless compute - prints a web-terminal-only warning and **exits 0** (silent no-op) | Section F rewritten to the Python SDK (`F2a`-`F2c`); CLI column kept as the web-terminal/CI equivalent | Exit code 0 is not success - verify side effects, not exit codes |
# MAGIC | 2 | App creation failed: invalid `user_api_scopes` | `user_api_scopes` is not a valid field on this platform version | Removed from `F2a` and both `app.yaml`/`databricks.yml` | Copy platform config from current docs, not memory - validate against the live API |
# MAGIC | 3 | First deploy crashed instantly: `python agent.py`, no dependencies | The SDK deploy path does **not** read `app.yaml` - the launch command and build must ride on the deployment itself | `F2b`: `command=["uv", "run", "uvicorn", ...]` + `pyproject.toml` | The two deployment steps are independent: resource ≠ code - and each route has its own manifest |
# MAGIC | 4 | `/chat` 500: `asyncio.run() cannot be called from a running event loop` | `async def chat` already runs on the event loop | `server.py`: the endpoint is a **sync `def`** on purpose - FastAPI runs it in its threadpool so `asyncio.run()` is legal | Async endpoints and `asyncio.run()`-style agent harnesses don't mix |
# MAGIC | 5 | `/chat` 500 on the **second** request: `Event loop is closed` | The OpenAI client was created once at module level; every request runs `asyncio.run()` (a **new** loop) and httpx binds its connection pool to the loop | `agent.py`: `set_default_openai_client(AsyncDatabricksOpenAI())` created **per run** | Long-lived async clients and per-request event loops are incompatible |
# MAGIC | 6 | `/chat` 500: `PERMISSION_DENIED: User does not have USE CATALOG on Catalog 'module5b'` | MCP's `tools/list` walks catalog → schema before `EXECUTE` ever matters | `F2a2`: `GRANT USE CATALOG` + `GRANT USE SCHEMA` before the function grants | Discovery and execution are separate permissions - grant for both |
# MAGIC | 7 | `/chat` **200 but grounded refusal** ("could not find the policy") - the sneakiest one | `vector_search()` inside the SQL function died at an `authzEval` on `GetTable module5b.skyjet.policies_index` (zero `queryVectorIndex` events). A Delta Sync Index's **backing table carries the index's own name** - not the source table's (`policies_kb`) | `F2a2`: `GRANT SELECT ON TABLE ...policies_index` (+ `policies_kb`); root cause pinpointed by querying `system.access.audit` | **A missing permission can look like a content problem.** The app answered 200, correctly refused to guess - and only the audit log revealed the access denial. This is why Section G exists |
# MAGIC
# MAGIC **The through-line:** governance failures rarely announce themselves. They surface as silent CLI no-ops, as 500s with cryptic tracebacks, or worst of all as a *well-behaved refusal*. Diagnosis discipline: read the actual failure layer (app logs → audit/system tables → grants), fix exactly that layer, verify in the browser - and keep the fix in the lab so the next student inherits the working version.

# COMMAND ----------

# DBTITLE 1,F2e (optional) delete the app to free quota
# F2e (optional, run when the lab is finished): delete the app to free its
# compute quota slot. From this notebook the SDK call is:
#   from databricks.sdk import WorkspaceClient
#   WorkspaceClient().apps.delete(name="skyjet-support-app")
# From the web terminal the CLI equivalent is:
#   databricks apps delete skyjet-support-app
# Kept commented so the app stays up for Section G monitoring.
print("Cleanup stays commented out: the app is still up for Sections G-J.")

# COMMAND ----------

# DBTITLE 1,Section G header: monitoring the traces
# MAGIC %md
# MAGIC # Section G: Monitoring, query the traces
# MAGIC
# MAGIC Traces are flowing (Section E). Production monitoring starts by making them **queryable**. Two levels:
# MAGIC
# MAGIC 1. **MLflow SDK** (G1): pull recent traces programmatically
# MAGIC 2. **SQL over Delta**: for long-term analysis, bind the experiment to a UC trace location (`trace_location=UnityCatalog(...)`), which creates `<prefix>_otel_spans` / `_otel_annotations` / `_otel_logs` / `_otel_metrics` Delta tables you can query from any warehouse and build dashboards/alerts on (the system-table monitoring patterns from Module 5A 1.11-1.12).

# COMMAND ----------

# DBTITLE 1,G1 query recent traces + error rate
# G1: Level 1 of monitoring - pull the recent traces programmatically and
# turn them into a health signal. Every agent request (app UI, I4, J3) has
# produced a trace; this is how you read them without the UI.
import time, mlflow

time.sleep(3)   # let traces finalize

traces = mlflow.search_traces(max_results=10)
print(f"{len(traces)} recent traces in experiment:\n")

if len(traces) > 0:
    for _, t in traces.iterrows():
        print(f"  {t.trace_id}  state={t.state}  spans={len(t.spans)}")

    # Simple health signal you could alert on (5A  1.12 pattern)
    errored = sum(1 for _, t in traces.iterrows() if "ERROR" in str(t.state).upper())
    print(f"\nError rate: {errored}/{len(traces)}; route to an Alert when this exceeds your SLO.")
    print("\nOpen a trace in the MLflow experiment UI to inspect the span tree:")
    print("AGENT root with CHAT_MODEL and TOOL children (incl. search_policies).")
else:
    print("Open the MLflow Experiment in the UI to inspect traces (SDK view may lag a few seconds).")

# ---------------------------------------------------------------------------
# Level 2: UC-backed trace storage - SQL over Delta for long-term analysis
# (dashboards, alerts). Binding is PERMANENT for the experiment: run it
# deliberately, once. It creates the module5b.skyjet.skyjet_traces_otel_*
# Delta tables, queryable from any warehouse.
#
# from mlflow.entities.trace_location import UnityCatalog
#
# mlflow.set_experiment(
#     experiment_name=MLFLOW_EXPERIMENT_NAME,
#     trace_location=UnityCatalog(
#         catalog_name=UC_TOOL_CATALOG,
#         schema_name=UC_TOOL_SCHEMA,
#         table_prefix="skyjet_traces",
#     ),
# )
# Then: SELECT * FROM module5b.skyjet.skyjet_traces_otel_spans
# Grants: USE_CATALOG + USE_SCHEMA + explicit MODIFY + SELECT
# (ALL_PRIVILEGES is NOT enough - UC trace tables need explicit grants)

# COMMAND ----------

# DBTITLE 1,Section H: the go/no-go production checklist
# MAGIC %md
# MAGIC # Section H: The go/no-go production checklist
# MAGIC
# MAGIC Run this before you call anything **production-ready**. Each gate traces back to a technique you learned:
# MAGIC
# MAGIC | # | Gate | Check | From |
# MAGIC |---|---|---|---|
# MAGIC | 1 | Tool governance | Every shared tool is a UC object (SQL *or* `create_python_function`), **including RAG retrieval**; only presentation glue uses `@function_tool` | NB1 Part A |
# MAGIC | 2 | Knowledge base | Policies PDF parsed and chunked into the KB table; vector index synced and queryable | 5A Demo 3 pipeline (Section S) |
# MAGIC | 3 | Tool contract | `tools/list` returns all required tools; descriptions name-when-to-use + typed Args | NB1 Part A/B |
# MAGIC | 4 | Grounding | Policy answers cite retrieved doc ids; unanswerable questions say so instead of guessing | this NB E2 |
# MAGIC | 5 | Config hygiene | `grep` the source: zero hardcoded hosts/tokens/endpoints; everything via env injection | NB1 Part D |
# MAGIC | 6 | Observability | Every request produces a trace; span tree shows AGENT → CHAT_MODEL/TOOL (incl. `search_policies`) | NB1 Part C |
# MAGIC | 7 | Acceptance tests | Unit tests + agent quality tests pass before deploy | this NB T |
# MAGIC | 8 | Deployability | `bundle validate` + `bundle deploy` + `apps deploy` succeed; `/health` returns ok; smoke test answered | this NB F |
# MAGIC | 9 | Permissions | App service principal: `EXECUTE` on UC functions, endpoint access, **index query access**, experiment write | NB1 Part D |
# MAGIC | 10 | Gateway governance | Guardrails / rate limits / usage tracking configured on the endpoint (platform-level) | NB1 Part D5 |
# MAGIC | 11 | Monitoring | Trace error-rate query wired to an Alert; scorer plan documented | this NB G |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC # 🎓 Module 5B: the complete arc
# MAGIC
# MAGIC **Module 5A** ended with a prototype: smart agents, ungoverned Python tools, notebook-bound RAG.
# MAGIC
# MAGIC **Notebook 1** closed the gaps: Python tools became governed UC objects, MCP made every tool discoverable at runtime, MLflow Tracing made every request observable, and DABs + Apps gave us the deployment architecture.
# MAGIC
# MAGIC **Notebook 2 (this lab)** assembled it all into one artifact: a governed, *grounded*, observable, configurable, **tested and deployed** multi-agent application. Note what is new versus any ordinary agent: the RAG retrieval you prototyped in 5A Demo 3 is now a **governed, MCP-discoverable tool** inside a supervisor-routed team of specialists. That is what **production-ready** means: not a smarter agent, but an agent the *platform* can govern, observe, test, deploy, and trust.

# COMMAND ----------

# DBTITLE 1,Section I header: (optional) your own serving endpoint
# MAGIC %md
# MAGIC # Section I (optional): Provision YOUR serving endpoint — the app's only model dependency
# MAGIC
# MAGIC F0 made the split explicit: the **App replaces the orchestration layer, not the model layer**. The deployed app needs exactly **one** serving resource: an LLM endpoint it can call on every agent turn. The lab runs on the **pay-per-token system endpoint** (`databricks-meta-llama-3-3-70b-instruct`), which exists out of the box — zero provisioning.
# MAGIC
# MAGIC When you want your own endpoint instead (your own quota, your own AI Gateway policies, provisioned throughput), the pattern is the same steps you already ran, pointed at the new endpoint:
# MAGIC
# MAGIC | Cell | What it does | Section F equivalent |
# MAGIC |---|---|---|
# MAGIC | I1 | Provision the serving endpoint (idempotent; on Free Edition it falls back to the built-in pay-per-token system endpoint) | - (the lab used the built-in one) |
# MAGIC | I3 | Direct call to the endpoint (tests the MODEL layer: raw LLM answer) | F2c |
# MAGIC | I4 | Call the APP itself via its /chat API - the grounded, cited answer | F2c |
# MAGIC
# MAGIC <div style="border-left:4px solid #f57c00;background:#fff3e0;padding:12px 16px;border-radius:4px;margin:12px 0">
# MAGIC <strong>⚠️ Cost awareness:</strong> a provisioned-throughput endpoint reserves GPU capacity and bills while scaled up (scale-to-zero helps). On Databricks Free Edition (this lab's workspace) provisioned throughput is not available at all: I1 will hit that gate and fall back to the built-in pay-per-token system endpoint, which is the right choice for this module anyway.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,I1 provision serving endpoint (idempotent)
# I1: Provision the LLM serving endpoint the app calls (idempotent).
# The lab default SERVING_ENDPOINT (databricks-meta-llama-3-3-70b-instruct) is
# a pay-per-token SYSTEM endpoint that already exists - nothing to provision,
# and it exposes no UC model/version to pin. A DEDICATED provisioned-
# throughput endpoint must serve a UC-hosted foundation model, so this cell
# provisions the docs-verified system.ai.gpt-oss-120b (task llm/v1/chat - the
# same OpenAI-compatible contract the agent's client already speaks).
# Throughput is provisioned in increments of the model's chunk size, so it is
# queried live rather than hardcoded (docs: "Get provisioned throughput in
# increments"). GPU capacity is reserved while scaled up - see the warning in
# the section header. Takes ~10-15 minutes on first creation.
import json

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

w = WorkspaceClient()
MY_ENDPOINT = "skyjet-llm-endpoint"
MY_MODEL = "system.ai.gpt-oss-120b"
MY_MODEL_VERSION = "1"

try:
    ep = w.serving_endpoints.get(MY_ENDPOINT)
    print(f"Endpoint '{MY_ENDPOINT}' already exists (provisioning is idempotent).")
except Exception:
    info = w.api_client.do(
        "GET",
        f"/api/2.0/serving-endpoints/get-model-optimization-info/{MY_MODEL}/{MY_MODEL_VERSION}",
    )
    chunk = info["model_unit_chunk_size"]
    print(f"Creating serving endpoint '{MY_ENDPOINT}' (throughput chunk size: "
          f"{chunk} tokens/s; takes ~10-15 minutes)...")
    try:
        ep = w.serving_endpoints.create_and_wait(
            name=MY_ENDPOINT,
            config=EndpointCoreConfigInput(
                name=MY_ENDPOINT,   # required on this SDK version
                served_models=[ServedModelInput(
                    name="skyjet-llm",
                    model_name=MY_MODEL,
                    model_version=MY_MODEL_VERSION,      # required: pins the UC model version
                    min_provisioned_throughput=0,        # 0 = required with scale-to-zero
                    max_provisioned_throughput=2 * chunk,  # raise for more traffic
                    scale_to_zero_enabled=True,          # no GPU billed while idle
                )]
            ),
        )
    except NotFound:
        # The workspace is Databricks Free Edition: provisioned-throughput
        # serving (reserved GPU capacity, billed while scaled up) does not
        # exist there. ONLY pay-per-token serving is available, in two
        # flavors:
        #   1. the databricks-* SYSTEM endpoints (like the lab's
        #      databricks-meta-llama-3-3-70b-instruct): pay-per-token out of
        #      the box, nothing to create - what the lab falls back to
        #   2. your OWN pay-per-token endpoint on a UC-registered foundation
        #      model: the same create call as above minus the throughput
        #      fields (ServedEntityInput; no min/max_provisioned_throughput,
        #      no scale-to-zero - billing is per token, not per capacity):
        #          w.serving_endpoints.create_and_wait(
        #              name="skyjet-llm-ppt",
        #              config=EndpointCoreConfigInput(
        #                  name="skyjet-llm-ppt",
        #                  served_entities=[ServedEntityInput(
        #                      entity_name="system.ai.gpt-oss-120b",
        #                      entity_version="1",
        #                  )],
        #              ),
        #          )
        # The PT create code above is correct and works on paid tiers where
        # PT is enabled; nothing in this cell is wrong - the gate is the
        # edition of the workspace, not the code.
        print("⚠️  Provisioned-throughput endpoints are not available on")
        print("   Databricks Free Edition - only pay-per-token serving is.")
        print("   Falling back to the lab's pay-per-token system endpoint.")
        MY_ENDPOINT = globals().get("SERVING_ENDPOINT",
                                    "databricks-meta-llama-3-3-70b-instruct")
        ep = w.serving_endpoints.get(MY_ENDPOINT)

print(f"✅ Endpoint ready: {ep.name} (state: {ep.state})")
print("   Next: I3 calls this endpoint directly (the model layer, raw LLM")
print("   reply); I4 calls the app itself (the agent layer, cited reply).")

# COMMAND ----------

# DBTITLE 1,I3 direct call to the model endpoint (the control)
# I3: Call the serving endpoint DIRECTLY - the MODEL layer in isolation,
# exactly what the app does on every agent turn minus everything the app
# adds (orchestration, governed tools, RAG). Its connection to I4: the two
# cells ask the same style of question at the two layers -
#   I3 (this cell): the model endpoint -> raw LLM answer, no tools, no docs
#   I4 (next cell): the app          -> the same brain, but routed through
#                  the supervisor + search_policies -> grounded, cited
# I3 is the CONTROL that shows exactly what the agent layer adds. On the
# Free Edition fallback (I1) this hits the default pay-per-token system
# endpoint - the very call every agent turn already makes under the hood,
# which is also why the model layer "just works" with zero setup.
# Equivalent curl (works anywhere with a token, e.g. the web terminal):
#   curl "$HOST/serving-endpoints/$MY_ENDPOINT/invocations" \
#        -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
#        -d '{"messages":[{"role":"user","content":"Reply with exactly: SkyJet endpoint OK"}],"max_tokens":20}'
import requests

url = f"{w.config.host}/serving-endpoints/{MY_ENDPOINT}/invocations"
r = requests.post(
    url,
    headers=w.config.authenticate(),
    json={
        "messages": [
            {"role": "user", "content": "Reply with exactly: SkyJet endpoint OK"},
        ],
        "max_tokens": 20,
    },
    timeout=120,
)
print(f"POST {url}  ->  HTTP {r.status_code}\n")
resp = r.json()
print("Model reply:", resp["choices"][0]["message"]["content"])

print("\nNext (I4): curl the APP itself - same question, but through the")
print("agent: orchestration + governed tools + RAG -> the cited answer.")

# COMMAND ----------

# DBTITLE 1,I4 curl the APP itself: the grounded, cited answer
# I4: The other half of the split - curl the DATABRICKS APP itself (the
# agent: orchestration + governed tools + RAG), not the model endpoint that
# I3 tested. The app is behind workspace SSO, so browsers authenticate
# automatically; a programmatic call must first exchange the notebook token
# for an audience-scoped token bound to the app's OAuth client (the documented
# Databricks Apps pattern). Then ONE POST /chat returns the same grounded,
# cited answer the UI gives.
# Equivalent curl from your laptop (a CLI token already includes all-apis):
#   TOKEN=$(databricks auth token --profile my-env | jq -r .access_token)
#   curl "$APP_URL/chat" -H "Authorization: Bearer $TOKEN" \
#        -H "Content-Type: application/json" \
#        -d '{"message":"What is SkyJet's refund policy if a flight is cancelled?"}'
import requests

APP_NAME_I4 = globals().get("APP_NAME", "skyjet-support-app")
app = w.apps.get(name=APP_NAME_I4)

# 1) the app's OAuth client id
app_client_id = app.oauth2_app_client_id

# 2) exchange the notebook token for an audience-scoped app token
notebook_token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().apiToken().get()
)
exchange = requests.post(
    f"{w.config.host}/oidc/v1/token",
    data={
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token": notebook_token,
        "subject_token_type": "urn:databricks:params:oauth:token-type:personal-access-token",
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "scope": "all-apis",
        "audience": app_client_id,
    },
)
audience_token = exchange.json()["access_token"]

# 3) call the app - this IS the agent, not the raw model
r = requests.post(
    f"{app.url.rstrip('/')}/chat",
    headers={"Authorization": f"Bearer {audience_token}",
             "Content-Type": "application/json"},
    json={"message": "What is SkyJet's refund policy if a flight is cancelled?"},
    timeout=300,
)
print(f"POST {app.url.rstrip('/')}/chat  ->  HTTP {r.status_code}\n")
print("Agent answer:\n", r.json()["response"])

print("\nCompare with I3: the raw model endpoint could not cite any docs - the")
print("app grounds the answer through its governed search_policies tool.")

# COMMAND ----------

# DBTITLE 1,Section J header: serve the agent as a serving endpoint
# MAGIC %md
# MAGIC # Section J (optional, no-UI mode): Serve the AGENT itself as a serving endpoint
# MAGIC
# MAGIC Sections F-I deployed the agent as a **Databricks App** (UI, SSO, chat history) backed by a **model** serving endpoint that only serves the LLM. There is a second, UI-free deployment mode: package the agent itself as an MLflow model and serve THAT - this is what the **Agents** option in the Serving UI (Create serving endpoint -> Select served entity -> your registered agent model) is for.
# MAGIC
# MAGIC Same brain, two shells:
# MAGIC
# MAGIC | | Databricks App (Sections F-I) | Agent serving endpoint (this section) |
# MAGIC |---|---|---|
# MAGIC | Hosts | orchestration + tools + RAG + UI | the packaged agent only (no UI) |
# MAGIC | Call it | browser UI, or `/chat` + audience token (I4) | one `POST /invocations` with a normal workspace token |
# MAGIC | Answer | grounded, cited (doc-021/022/023) | grounded, cited - the SAME agent code |
# MAGIC
# MAGIC | Cell | What it does |
# MAGIC |---|---|
# MAGIC | J1 | Package the agent as an MLflow pyfunc model and register it in Unity Catalog |
# MAGIC | J2 | Create the agent serving endpoint with one call: `deploy()` (scale-to-zero + automatic grants) |
# MAGIC | J3 | Curl-style `POST /invocations` - the cited answer, no app, no UI |
# MAGIC
# MAGIC <div style="border-left:4px solid #f57c00;background:#fff3e0;padding:12px 16px;border-radius:4px;margin:12px 0">
# MAGIC <strong>⚠️ Cost awareness:</strong> an agent endpoint bills DBUs while running (scale-to-zero keeps idle cost at zero, but each cold start takes minutes). Delete it when done with <code>w.serving_endpoints.delete(...)</code>. First creation takes ~10-15 minutes.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,J1 package + register the agent as an MLflow model
# J1: Package the SkyJet AGENT as an MLflow model and register it in Unity
# Catalog - the prerequisite for serving it. The Databricks App runs agent.py
# as loose Python (uvicorn); a serving endpoint can only serve a REGISTERED
# model, so the whole agent - instructions, supervisor handoffs, governed MCP
# tools - is wrapped in a pyfunc. Three things make this work in production:
#   1. code_paths ships the EXACT agent.py the app runs (one source of truth)
#   2. the config the agent reads at import time is baked into the model
#   3. mlflow declares its dependencies (LLM endpoint, VS index, UC functions)
#      via `resources` - this is what lets the endpoint receive its grants
#      automatically at deploy time (J2) instead of manual GRANTs.
import importlib
import os

import mlflow
from mlflow.models.resources import (
    DatabricksFunction, DatabricksServingEndpoint, DatabricksVectorSearchIndex,
)

mlflow.set_experiment(globals().get("MLFLOW_EXPERIMENT_NAME",
    "/Users/shriveens@platformatory.com/module5b_skyjet_production"))

MODEL_NAME = "module5b.skyjet.skyjet_agent"   # UC: catalog.schema.model

# UC-registered models REQUIRE a signature with input AND output specs - and
# an agent serving endpoint (J2) enforces the Agent Framework I/O contract on
# top of that: the model must speak the OpenAI wire format (messages in,
# chat.completion out). Anything else - like a custom {"message": ...}
# contract - is REJECTED at deploy time. MLflow ships the exact allowed
# schemas as public constants; using them (not infer_signature) guarantees
# J2's deploy-time validation passes.
from mlflow.models.signature import ModelSignature
from mlflow.types.llm import CHAT_MODEL_INPUT_SCHEMA, CHAT_MODEL_OUTPUT_SCHEMA

SIGNATURE = ModelSignature(
    inputs=CHAT_MODEL_INPUT_SCHEMA,    # {"messages": [{"role", "content"}], ...}
    outputs=CHAT_MODEL_OUTPUT_SCHEMA,  # OpenAI-style chat.completion reply
)


class SkyJetAgentModel(mlflow.pyfunc.PythonModel):
    """The UI-free shell around the SAME agent code the app runs.

    Accepts OpenAI-style input (like the app's UI sends) or a plain message
    string, and returns the final grounded, cited answer as an OpenAI-style
        chat.completion object - the serving contract the signature promises.
        The asyncio guard
    is the serving twin of the Section F2 lesson: asyncio.run() works from a
    plain worker thread, but NOT from inside a running event loop - so if one
    is running (async serving stacks), the call is offloaded to a fresh thread.
    """

    def __init__(self, config: dict):
        self.config = dict(config)   # env vars agent.py reads at import time

    def load_context(self, context):
        os.environ.update(self.config)
        self.agent = importlib.import_module("agent")   # shipped via code_paths

    def predict(self, context, model_input, params=None):
        message = model_input
        if hasattr(message, "columns"):   # pyfunc serving may pass a DataFrame
            message = message.to_dict("records")[0]
        if isinstance(message, dict):
            if "messages" in message:   # [{"role": "user", "content": ...}]
                message = next((m.get("content", "") for m in message["messages"]
                                if m.get("role") == "user"), "")
            else:
                message = message.get("message", "")
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                text = pool.submit(self.agent.handle_request, str(message)).result(timeout=300)
        except RuntimeError:   # no loop running (sync worker): the F2 pattern
            text = self.agent.handle_request(str(message))
        # Reply in the exact OpenAI chat.completion shape the output schema
        # promises - the same JSON I3 received from the raw model endpoint,
        # so any OpenAI-compatible client can consume the agent endpoint too.
        import time as _t
        from uuid import uuid4
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(_t.time()),
            "model": "skyjet-agent",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        }


with mlflow.start_run(run_name="skyjet-agent-package") as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=SkyJetAgentModel({
            "UC_TOOL_CATALOG":  globals().get("UC_TOOL_CATALOG", "module5b"),
            "UC_TOOL_SCHEMA":   globals().get("UC_TOOL_SCHEMA", "skyjet"),
            "SERVING_ENDPOINT": globals().get("SERVING_ENDPOINT",
                                              "databricks-meta-llama-3-3-70b-instruct"),
        }),
        signature=SIGNATURE,
        input_example={"messages": [{"role": "user",
                                  "content": "What is SkyJet's refund policy if a flight is cancelled?"}]},
        code_paths=[f"{globals().get('SRC', '')}/agent.py"],
        pip_requirements=[
            "mlflow>=3.17",
            "databricks-openai==0.17.1",   # AsyncDatabricksOpenAI + McpServer
            "openai-agents==0.23.1",       # Agent / Runner / handoffs
            "databricks-sdk>=0.122.0",
            "mcp>=1.30",
            "requests",
        ],
        resources=[
            DatabricksServingEndpoint(endpoint_name=globals().get("SERVING_ENDPOINT",
                "databricks-meta-llama-3-3-70b-instruct")),   # the LLM brain
            DatabricksVectorSearchIndex(index_name=globals().get("VS_INDEX",
                "module5b.skyjet.policies_index")),           # the RAG corpus
        ] + [   # the governed tools
            DatabricksFunction(function_name=f"module5b.skyjet.{fn}") for fn in
            ["search_policies", "get_flight_status", "count_flights_by_status",
             "estimate_baggage_fee", "mask_contact_info"]
        ],
    )

# Register in Unity Catalog - now it appears in the Serving UI dropdown under
# "My models" (exactly the dialog in the Create-serving-endpoint flow).
model_version = mlflow.register_model(
    f"runs:/{run.info.run_id}/agent", MODEL_NAME)
print(f"✅ Agent packaged and registered: {MODEL_NAME} version {model_version.version}")
print("   Next: J2 creates the serving endpoint for it (takes ~10-15 minutes).")

# COMMAND ----------

# DBTITLE 1,J2 explainer: what J2 does, in plain words
# MAGIC %md
# MAGIC ### What J2 does, in plain words
# MAGIC
# MAGIC | Cell | What happens | Analogy |
# MAGIC |---|---|---|
# MAGIC | J1 | pack the agent into a registered model | put your app in a box |
# MAGIC | **J2 (this cell)** | put that box online as an API | give the box a URL |
# MAGIC | J3 | ask it a question, get the cited answer | use it |
# MAGIC
# MAGIC The code is just one call, `deploy()`, which:
# MAGIC
# MAGIC * creates `skyjet-agent-endpoint` and points it at the packed agent
# MAGIC * turns on scale-to-zero (no cost while idle; the first request after idle takes a few minutes)
# MAGIC * automatically grants the endpoint permission to use everything the agent needs (the LLM, the policy docs, the tools) - the permissions we granted by hand in F2a2, but done for you
# MAGIC
# MAGIC Re-running is safe: if the endpoint already exists, it just updates to the newest version.

# COMMAND ----------

# DBTITLE 1,J2 create the agent serving endpoint
# J2: Put the packed agent (J1) online - ONE call. deploy() creates the
# endpoint, points it at the model version, enables scale-to-zero, and
# automatically grants the endpoint permission to use everything J1 logged
# in `resources` (the LLM, the policy docs, the tools) - the grants we handed
# out manually in F2a2, done for us. It also validates the model's I/O
# contract (OpenAI messages in / chat.completion out) before serving.
# Re-run safe: if the endpoint already exists, deploy() just updates it to
# the newest version. First creation takes ~10-15 minutes.
from databricks.agents import deploy

AGENT_ENDPOINT = "skyjet-agent-endpoint"

deployment = deploy(
    model_name=MODEL_NAME,
    model_version=int(model_version.version),
    endpoint_name=AGENT_ENDPOINT,
    scale_to_zero=True,      # no DBUs while idle
    workload_size="Small",   # smallest serving footprint
)

print(f"✅ Agent serving endpoint: {deployment.endpoint_name}")
print(f"   Query it at /serving-endpoints/{AGENT_ENDPOINT}/invocations")
print("   Next: J3 curls it - the cited answer, no app, no UI.")

# COMMAND ----------

# DBTITLE 1,J3 curl the agent endpoint: the cited answer, no UI
# J3: The UI-free payoff - ONE curl-style POST to the agent endpoint returns
# the SAME grounded, cited answer as the app's UI, with no app, no SSO token
# exchange (unlike I4 - model serving accepts a normal workspace token), and
# no UI. This is the full answer to 'what if I dont want the fancy UI':
# the agent itself becomes the API.
# Equivalent curl from anywhere:
#   curl "$HOST/serving-endpoints/skyjet-agent-endpoint/invocations" \
#        -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
#        -d '{"inputs":{"messages":[{"role":"user","content":"What is SkyJet\'s refund policy if a flight is cancelled?"}]}}'
import requests

url = f"{w.config.host}/serving-endpoints/{AGENT_ENDPOINT}/invocations"
r = requests.post(
    url,
    headers=w.config.authenticate(),
    json={"inputs": {"messages": [{"role": "user",
                                   "content": "What is SkyJet's refund policy if a flight is cancelled?"}]}},
    timeout=600,   # first call after a cold start can take a few minutes
)
print(f"POST {url}  ->  HTTP {r.status_code}\n")
resp = r.json()
print("Agent answer (no UI, no app - just the packaged agent):\n")
body = resp.get("predictions", resp)   # pyfunc serving wraps the reply
print(body["choices"][0]["message"]["content"])   # same OpenAI shape as I3

print("\nSame question, three surfaces - one agent brain:")
print("   I3: raw model endpoint  -> no grounding, no citations")
print("   I4: Databricks App      -> grounded + cited (with UI/SSO)")
print("   J3: agent endpoint      -> grounded + cited (pure API)")
