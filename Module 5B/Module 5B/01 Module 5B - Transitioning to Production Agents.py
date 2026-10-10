# Databricks notebook source
# DBTITLE 1,Title cell rewrite
# MAGIC %md
# MAGIC
# MAGIC # 🌉 Notebook 1 : From Module 5A's Prototype Tools to Production-Grade Agents
# MAGIC
# MAGIC ## The 5A → 5B Transition
# MAGIC
# MAGIC In **Module 5A** you built agents the *prototype* way:
# MAGIC
# MAGIC | What you did in 5A | How the tool was built | Where it lived | Governed by UC? |
# MAGIC |---|---|---|---|
# MAGIC | **SQL tools** (Demo 4, registered with `CREATE FUNCTION`) | SQL, stored in UC | ✅ Unity Catalog | ✅ Yes: permissions, lineage, discovery |
# MAGIC | **Python tools** (`mask_pii`, `sanitize_input`, …) | `@function_tool` decorator | ❌ Only inside the notebook | ❌ **No**: invisible to UC, not reusable |
# MAGIC | **RAG retrieval** (Demo 3, parsed PDF → index → search) | Prototype pipeline code | ❌ Only inside the notebook | ❌ **No**: not a governed, callable tool |
# MAGIC
# MAGIC The `@function_tool` decorator is fantastic for **rapid prototyping**, but those functions were **trapped in the notebook**. No permissions, no lineage, no discovery by other agents, no MCP exposure. That was the deliberate cliffhanger of 5A.
# MAGIC
# MAGIC > 🎯 **Module 5B closes that gap.** We take the Python tools, **register them into Unity Catalog**, expose **everything**, including RAG retrieval, through the **Model Context Protocol (MCP)**, make the agent **observable with MLflow Tracing**, and learn the **deployment architecture** (Databricks Apps + Declarative Automation Bundles) that turns a notebook prototype into a production application.
# MAGIC
# MAGIC ## Today's Roadmap
# MAGIC
# MAGIC | Part | Topic | Mode |
# MAGIC |---|---|---|
# MAGIC | **A** | Registering Python tools into Unity Catalog (`DatabricksFunctionClient`) | Theory + Demo + Test |
# MAGIC | **B** | The Model Context Protocol: discovery, invocation, lifecycle | Theory + Demo + Test |
# MAGIC | **C** | Observability: MLflow Tracing in depth | Theory + Demo + Test |
# MAGIC | **D** | Deployment architecture: Databricks Apps, DABs, resource injection, AI Gateway | Theory (applied in Notebook 2) |
# MAGIC | **E** | Evaluation & monitoring: scorers, judges, backfill | Theory + pointers |
# MAGIC
# MAGIC **Our domain for Module 5B: SkyJet Airlines.** We are building a **passenger-support assistant** for SkyJet, grounded in the airline's passenger-policies PDF (already in this module's `static` folder). Notebook 1 gives that assistant governed, discoverable, observable tools; Notebook 2 assembles the full production application with RAG retrieval over the policies document, using the same parse → chunk → index → retrieve pipeline you built in 5A Demo 3.
# MAGIC
# MAGIC *Built on the techniques from Module 5A: agent tools (Demo 4), the PDF-to-searchable-index RAG pipeline (Demo 3), and multi-agent orchestration (Demo 6).*

# COMMAND ----------

# DBTITLE 1,Part A theory final dash fixes
# MAGIC %md
# MAGIC # Part A: Registering Python Tools into Unity Catalog
# MAGIC
# MAGIC ## A1. The Bridge: from `@function_tool` to a governed UC function
# MAGIC
# MAGIC In 5A, a Python tool looked like this:
# MAGIC
# MAGIC ```python
# MAGIC from agents import function_tool
# MAGIC
# MAGIC @function_tool
# MAGIC def mask_pii(text: str) -> str:
# MAGIC     """Mask emails and phone numbers in free text."""
# MAGIC     ...
# MAGIC ```
# MAGIC
# MAGIC The decorator reads the **name, type hints, and docstring** to build the tool schema the LLM sees, but the function lives **only in your notebook's memory**. Compare that with the SQL tools you registered with `CREATE FUNCTION` in 5A Demo 4, which live in Unity Catalog with permissions, comments, and lineage.
# MAGIC
# MAGIC The bridge is the **Databricks Function Client** (`unitycatalog-ai`):
# MAGIC
# MAGIC ```python
# MAGIC from unitycatalog.ai.core.databricks import DatabricksFunctionClient
# MAGIC
# MAGIC client = DatabricksFunctionClient()
# MAGIC function_info = client.create_python_function(
# MAGIC     func=my_python_tool,        # the actual Python function object
# MAGIC     catalog=CATALOG,
# MAGIC     schema=SCHEMA,
# MAGIC     replace=True,
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC After this call, the Python function is a **first-class Unity Catalog function**, exactly the same kind of object as your SQL functions. It appears in Catalog Explorer, honors UC permissions, and (as we'll see in Part B) is automatically exposed through Databricks' **managed MCP server**.
# MAGIC
# MAGIC ## A2. Best practices for Python agent tools
# MAGIC
# MAGIC Here is the one lesson that separates a working tool from a broken one: **an agent is only as good as its tool descriptions**. When you register a Python function to UC:
# MAGIC
# MAGIC 1. **Explicit type hints** on every argument and the return value: the client infers the UC function signature from them.
# MAGIC 2. **A rich docstring** with an `Args:` section: this is what the LLM reads to decide *when* and *how* to call the tool. A vague docstring ⇒ the agent calls the wrong tool or the wrong arguments.
# MAGIC 3. **Self-contained imports**: put `import json`, `import re`, etc. *inside* the function body, because the function is serialized and executed remotely by UC.
# MAGIC 4. **Return strings**, not exotic objects: LLMs consume text.
# MAGIC 5. **Test at notebook level first**, register second (fail fast before it becomes a governed object).
# MAGIC
# MAGIC <div style="border-left:4px solid #1976d2;background:#e3f2fd;padding:12px 16px;border-radius:4px;margin:12px 0">
# MAGIC <strong>Why this matters for production:</strong> in 5A's prototype, a bad docstring only affected one notebook. In production, a badly documented UC tool misleads <em>every</em> agent that discovers it, across teams and workspaces. Governance raises the blast radius, so documentation becomes part of your API contract.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,A3 header rewrite
# MAGIC %md
# MAGIC ### A3. Setup: our SkyJet workspace
# MAGIC
# MAGIC We create a fresh home for Module 5B: the `module5b.skyjet` catalog and schema. Everything we build in this module (tools, tables, indexes, traces) lives here. We also verify the 5A Demo 3 RAG legacy is still in place, since Notebook 2 reuses that pipeline pattern.

# COMMAND ----------

# DBTITLE 1,A3.1 install unitycatalog-ai
# MAGIC %pip install unitycatalog-ai[databricks]==0.3.2
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,A3.2 config
# --- Module 5B configuration ---
CATALOG = "module5b"
SCHEMA  = "skyjet"
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

# In production (Notebook 2) these come from environment variables instead:
#   CATALOG = os.environ["UC_TOOL_CATALOG"]
#   SCHEMA  = os.environ["UC_TOOL_SCHEMA"]
print(f"Working in Unity Catalog schema: {FULL_SCHEMA}")

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FULL_SCHEMA}")
print("✅ Ready.")

# COMMAND ----------

# DBTITLE 1,A3.3 dash fix
# Verify what Module 5A left behind (Demo 3: the PDF-to-searchable-index pipeline)
kb = spark.sql("SELECT count(*) AS c FROM module5a_demo3.rag.knowledge_base").collect()[0].c
print(f"5A Demo 3 knowledge base (parsed eBook PDF): {kb} indexed document chunks")
print("5A Demo 3 vector index: module5a_demo3.rag.knowledge_base_index  ✓")
print("5A Demo 4 SQL tools: registered with CREATE FUNCTION, governed by UC  ✓")
print("""
What was MISSING from 5A (and what we fix today):
  ✗ Python tools built with @function_tool lived only inside the notebook
  ✗ The RAG retrieval code from Demo 3 was also just notebook code,
    not a governed, callable tool any agent could discover
Today we make both first-class Unity Catalog citizens.
""")

# COMMAND ----------

# DBTITLE 1,A4 header rewrite
# MAGIC %md
# MAGIC ### A4. Define a NEW Python tool: business logic SQL cannot express
# MAGIC
# MAGIC In 5A, a passenger-support agent could look up flight records with SQL. But a real airline assistant also needs logic that is awkward in pure SQL, like a **baggage fee calculator** with class allowances and loyalty discounts. This is the exact category of tool that in 5A we could only build with `@function_tool` (ungoverned). Now we build it the production way.
# MAGIC
# MAGIC Notice the two things that make it agent-ready: **type hints** and a **rich docstring with Args**. Without them, the LLM guesses the tool's contract and calls it wrongly.

# COMMAND ----------

# DBTITLE 1,A4.1 smoke-test dash fix
def estimate_baggage_fee(weight_kg: float, cabin_class: str, loyalty_tier: str) -> str:
    """
    Calculates the checked-baggage fee for a SkyJet passenger based on bag weight,
    travel class, and loyalty tier.

    Use this tool whenever a passenger or agent asks about baggage fees, or before
    quoting a total trip cost that must include checked-bag charges.

    Args:
        weight_kg (float): Total checked-bag weight in kilograms, between 1 and 60
            (e.g. 23.5).
        cabin_class (str): Travel class. One of 'economy', 'premium', 'business'
            (case-insensitive).
        loyalty_tier (str): SkyJet loyalty tier. One of 'none', 'silver', 'gold',
            'platinum' (case-insensitive).

    Returns:
        str: A human-readable fee breakdown, e.g.
        'Economy: 20 kg free allowance; 3.5 kg excess at $12.00/kg = $42.00; silver discount 10% → $37.80 total'.
        Or an error message if inputs are invalid.
    """
    allowances = {"economy": 20.0, "premium": 30.0, "business": 40.0}
    discounts = {"none": 0.0, "silver": 0.10, "gold": 0.20, "platinum": 0.35}
    RATE_PER_KG = 12.0

    cls = (cabin_class or "").strip().lower()
    tier = (loyalty_tier or "").strip().lower()

    if cls not in allowances:
        return f"Error: unknown cabin class '{cabin_class}'. Use economy, premium, or business."
    if tier not in discounts:
        return f"Error: unknown loyalty tier '{loyalty_tier}'. Use none, silver, gold, or platinum."
    if weight_kg is None or weight_kg < 1 or weight_kg > 60:
        return "Error: weight_kg must be between 1 and 60 kg."

    allowance = allowances[cls]
    excess = max(0.0, weight_kg - allowance)
    fee = excess * RATE_PER_KG
    discount = discounts[tier]
    final = fee * (1 - discount)

    if excess == 0:
        return f"{cls.capitalize()}: {allowance:g} kg free allowance, so no fee (bag: {weight_kg} kg)."
    return (
        f"{cls.capitalize()}: {allowance:g} kg free allowance; "
        f"{excess:g} kg excess at ${RATE_PER_KG:.2f}/kg = ${fee:.2f}; "
        f"{tier} discount {discount:.0%} → ${final:.2f} total"
    )


def mask_contact_info(text: str) -> str:
    """
    Masks emails and phone numbers in free text so customer contact details are
    never echoed back in plain form in agent responses or logs.

    Args:
        text (str): Free text that may contain emails or phone numbers,
            e.g. 'Call me at 555-0100 or jane@example.com'.

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


# Smoke test at notebook level BEFORE registering (catch errors early;
# never promote an untested function into governed UC)
print(estimate_baggage_fee(23.5, "economy", "silver"))
print(estimate_baggage_fee(15, "BUSINESS", "platinum"))
print(estimate_baggage_fee(99, "first", "gold"))     # graceful error path
print(mask_contact_info("Call me at 555-0100 or jane.doe@example.com"))

# COMMAND ----------

# DBTITLE 1,A5 register header
# MAGIC %md
# MAGIC ### A5. Register the Python tool into Unity Catalog
# MAGIC
# MAGIC One call moves it from *notebook memory* to *governed platform object*. Then we prove it is real UC by executing it **through UC**, not through the local Python object.

# COMMAND ----------

# DBTITLE 1,A5.1 register
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

for fn in (estimate_baggage_fee, mask_contact_info):
    info = client.create_python_function(func=fn, catalog=CATALOG, schema=SCHEMA, replace=True)
    print(f"✅ Registered: {CATALOG}.{SCHEMA}.{fn.__name__}")

# COMMAND ----------

# DBTITLE 1,A5.2 execute via UC
# Test it THROUGH Unity Catalog (not the local function object).
# This is the exact execution path an agent or MCP server would take.
result = client.execute_function(
    function_name=f"{CATALOG}.{SCHEMA}.estimate_baggage_fee",
    parameters={"weight_kg": 28.0, "cabin_class": "economy", "loyalty_tier": "gold"},
)
print(result.value)

result2 = client.execute_function(
    function_name=f"{CATALOG}.{SCHEMA}.mask_contact_info",
    parameters={"text": "Passenger contact: bob@skyjet-demo.com, +1 555 0100"},
)
print(result2.value)

# COMMAND ----------

# DBTITLE 1,A5.3 dash fix
# MAGIC %sql
# MAGIC -- And it is callable from plain SQL too: one namespace for SQL and Python tools.
# MAGIC -- NOTE: a UC function must be called by its fully qualified three-part name,
# MAGIC -- because SQL only searches system.* and the default schema, never module5b.skyjet.
# MAGIC SELECT module5b.skyjet.estimate_baggage_fee(25.5, 'premium', 'silver') AS baggage_quote

# COMMAND ----------

# DBTITLE 1,Part B theory dash fixes
# MAGIC %md
# MAGIC # Part B: The Model Context Protocol (MCP)
# MAGIC
# MAGIC ## B1. What is MCP?
# MAGIC
# MAGIC The **Model Context Protocol (MCP)** is an open protocol for connecting AI applications to external systems. Think of it like a **USB-C port for AI**: just as USB-C standardizes how peripherals connect to computers, MCP standardizes how agents connect to tools.
# MAGIC
# MAGIC Without MCP, every tool integration needs custom code: API wrappers, auth handling, response parsing. MCP eliminates this with a common protocol any compliant client and server can speak.
# MAGIC
# MAGIC **Client-server architecture, three participants:**
# MAGIC
# MAGIC | Participant | Role |
# MAGIC |---|---|
# MAGIC | **MCP Host** | The AI application (e.g., your agent app) that manages one or more MCP clients |
# MAGIC | **MCP Client** | A component inside the host holding a dedicated connection to one MCP server |
# MAGIC | **MCP Server** | A program exposing tools, resources, or prompts to connected clients |
# MAGIC
# MAGIC Servers expose three primitives: **Tools** (executable functions), **Resources** (read-only data), and **Prompts** (reusable templates). Tools are the integration point for agents.
# MAGIC
# MAGIC ## B2. MCP on Databricks: three server categories
# MAGIC | Type | Description | Example |
# MAGIC |---|---|---|
# MAGIC | **Managed MCP** | Pre-configured servers with immediate access to Databricks features | UC functions, Genie spaces, AI Search indexes |
# MAGIC | **External MCP** | Managed connections to MCP servers hosted outside Databricks | Third-party APIs, SaaS integrations |
# MAGIC | **Custom MCP** | An MCP server you write and host as a Databricks App | Domain-specific tools, internal services |
# MAGIC
# MAGIC The most common starting point is a **Managed MCP server backed by Unity Catalog functions**: you write a function (SQL *or* Python), register it in UC, and the managed MCP server **automatically exposes it as a tool** any connected agent can discover and invoke: no custom integration code.
# MAGIC
# MAGIC <div style="border-left:4px solid #1976d2;background:#e3f2fd;padding:12px 16px;border-radius:4px;margin:12px 0">
# MAGIC <strong>The 5A → 5B payoff:</strong> a single host can connect to multiple servers simultaneously, each through its own client instance. Your 5A SQL tools and today's registered Python tool now live in the <em>same</em> governed namespace, exposed through the <em>same</em> protocol.
# MAGIC </div>
# MAGIC
# MAGIC ## B3. The MCP connection lifecycle
# MAGIC
# MAGIC **Initialize → Discover → Invoke → Shutdown**
# MAGIC
# MAGIC 1. **Initialization**: the client sends an `initialize` request declaring protocol version and capabilities; the server responds with its own; the client confirms with an `initialized` notification.
# MAGIC 2. **Tool discovery**: the client sends `tools/list`; the server returns all tools with names, descriptions, and input schemas. The agent uses these descriptions to decide when to call each tool.
# MAGIC 3. **Tool invocation**: the client sends `tools/call` with the tool name and arguments; the server executes and returns the result.
# MAGIC 4. **Shutdown**: either side closes the connection gracefully.
# MAGIC
# MAGIC > 💡 Tools are **discovered at runtime, not hardcoded**. Register a new UC function on the server and connected agents can use it **without redeployment**.

# COMMAND ----------

# DBTITLE 1,B4 header final dash fix
# MAGIC %md
# MAGIC ### B4. Demo: discover and call tools through the Managed MCP server
# MAGIC
# MAGIC The managed MCP server for UC functions lives at a predictable URL:
# MAGIC `{workspace_host}/api/2.0/mcp/functions/{catalog}/{schema}`
# MAGIC
# MAGIC We point `DatabricksMCPClient` at the `module5b.skyjet` schema and watch it discover **the registered Python tools through one protocol**. In Part B we only have Python tools so far; Notebook 2 adds SQL tools and the RAG retrieval tool to the same schema, and they all surface the same way.

# COMMAND ----------

# DBTITLE 1,B4.0 install databricks-mcp
# MAGIC %pip install databricks-mcp
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,B4.1 MCP discovery (tools/list)
import nest_asyncio
nest_asyncio.apply()

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient

ws = WorkspaceClient()
workspace_host = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"

# The managed MCP server exposing every UC function in the schema
mcp_url = f"{workspace_host}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}"
print(f"Managed MCP server: {mcp_url}\n")

mcp_client = DatabricksMCPClient(server_url=mcp_url, workspace_client=ws)

tools = mcp_client.list_tools()
print(f"🧰 Found {len(tools)} tools discovered at runtime:")
for t in tools:
    print(f"  • {t.name}")

# COMMAND ----------

# DBTITLE 1,B4.2 invoke
# Invoke a tool through MCP exactly as an agent would (tools/call).
# MCP tool names are schema-qualified: catalog__schema__function
# (the same names tools/list printed above). A bare "estimate_baggage_fee" is rejected.
# If the very first call raises a ReadTimeout, the serverless execution
# environment was warming up: simply run the cell again.
result = mcp_client.call_tool("module5b__skyjet__estimate_baggage_fee",
                              {"weight_kg": 28.0, "cabin_class": "economy", "loyalty_tier": "gold"})
print(f"MCP result: {result}")

# And the PII-masking tool through the same protocol:
result2 = mcp_client.call_tool("module5b__skyjet__mask_contact_info",
                               {"text": "Contact passenger at jane.doe@example.com"})
print(f"\nMCP result (PII masking): {result2}")

# COMMAND ----------

# DBTITLE 1,B6 Agent constructor surface: tools, mcp_servers, handoffs
# MAGIC %md
# MAGIC ### B6. From protocol to agent: the Agent constructor surface (tools, mcp_servers, handoffs)
# MAGIC
# MAGIC Part B connected an agent to UC functions through the MCP *protocol*. You already know the OpenAI Agents SDK from 5A (`Agent`, `Runner`), but in 5A you only ever passed one option: `tools=[...]` with `@function_tool` functions. Before Notebook 2 productionizes this, here is the full constructor surface you will see in production `agent.py`:
# MAGIC
# MAGIC | Option | Receives | Job | 5A prototype | 5B production |
# MAGIC |---|---|---|---|---|
# MAGIC | `model` | serving endpoint name | the LLM, routed through Databricks model serving | used | unchanged (env-injected) |
# MAGIC | `instructions` | the system prompt | behavior, grounding, and routing rules | used | unchanged (now acceptance-tested) |
# MAGIC | `tools=[...]` | in-process Python functions (`@function_tool`) | agent-local glue | the only option you used | rare: only helpers that live in one process (e.g. formatting a chat-UI answer) |
# MAGIC | `mcp_servers=[...]` | MCP server objects | governed, runtime-discovered tools from Unity Catalog | never used | the default: where shared tools come from |
# MAGIC | `handoffs=[...]` | other `Agent`s | supervisor-worker routing to specialists | Demo 6 (prototype) | unchanged, now with MCP-wired workers |
# MAGIC | `output_type` | a dataclass type | structured final output | unused | optional |
# MAGIC
# MAGIC Four clarifications that prevent the classic confusion when reading Notebook 2's production `agent.py`:
# MAGIC
# MAGIC 1. **`tools=` is not prototype-only, it is local-only.** A decorated function is carried by the process that defines it: fine (and correct) for glue only one agent needs; wrong for shared tools, which no other agent can see, permission, or audit. Shared means UC (Part A), and an agent reaches UC through `mcp_servers=`.
# MAGIC 2. **`mcp_servers=` is how an agent uses what Part B taught.** `McpServer.from_uc_function(catalog, schema)` opens the same managed MCP server B4 hit (`/api/2.0/mcp/functions/{catalog}/{schema}`) and hands the agent every UC function in the schema, SQL and Python alike, discovered at runtime with `tools/list`, invoked with `tools/call`. Because an MCP server is an async context manager, production agents attach it per request inside `run()`, not at construction (you will see exactly that in Notebook 2).
# MAGIC 3. **Where does the vector index MCP fit?** Databricks also exposes a managed MCP server over an AI Search (Vector Search) index, so an agent can query an index directly through MCP. Both routes reach the same index. The UC-function route (Notebook 2's choice) makes retrieval itself a governed tool: one namespace, grants, lineage, audit; and it works wherever the calling identity has no direct index access. The index MCP is the shorter path for code that already holds index access.
# MAGIC 4. **Who gets which tools?** `mcp_servers=` is set per agent. Attach the same schema server to several specialists and steer selection with their `instructions` (Notebook 2 attaches one server to both specialists), or attach different per-schema servers to different agents for hard tool isolation.
# MAGIC

# COMMAND ----------

# DBTITLE 1,B5 key takeaway: MCP for production agents
# MAGIC %md
# MAGIC ### B5. Key takeaway
# MAGIC
# MAGIC <div style="border-left:4px solid #1976d2;background:#e3f2fd;padding:12px 16px;border-radius:4px;margin:12px 0">
# MAGIC <strong>B4/B5, key takeaway:</strong> SQL tools and Python tools are now <strong>equal citizens</strong> in one governed UC namespace, exposed through one protocol. An agent discovers them at <strong>runtime</strong> via <code>tools/list</code>: registering a new function requires <strong>no agent redeployment</strong>. This is the mechanism that makes 5A's prototype tools reusable across every agent in the organization.
# MAGIC </div>
# MAGIC
# MAGIC For **production agents**, the OpenAI Agents SDK connects to the same servers with `McpServer.from_uc_function` (async context manager: initialization and shutdown handled for you):
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_openai.agents import McpServer
# MAGIC from agents import Agent, Runner
# MAGIC
# MAGIC workspace_client = WorkspaceClient()
# MAGIC
# MAGIC async with McpServer.from_uc_function(
# MAGIC     catalog="module5b",
# MAGIC     schema="skyjet",
# MAGIC     workspace_client=workspace_client,
# MAGIC     name="uc-functions",
# MAGIC ) as uc_server:
# MAGIC     agent = Agent(
# MAGIC         name="skyjet_agent",
# MAGIC         model="databricks-claude-sonnet-4-5",
# MAGIC         mcp_servers=[uc_server],
# MAGIC     )
# MAGIC     result = await Runner.run(agent, "What's the baggage fee for a 28 kg bag in economy with a gold loyalty tier?")
# MAGIC ```
# MAGIC
# MAGIC We wire this into the full production application in **Notebook 2**.

# COMMAND ----------

# DBTITLE 1,Part C: observability with MLflow tracing
# MAGIC %md
# MAGIC # Part C: Observability with MLflow Tracing
# MAGIC
# MAGIC ## C1. Why 5A's testing wasn't enough
# MAGIC
# MAGIC In 5A we tested agents by printing final answers. In production that is not enough: you must see **server-side behavior**, which tools fired, in what order, with what latency, and where it failed. *(Remember from 5A: GenAI apps need deeper insight into latency/cost metrics and API logging than traditional ML batch workloads.)*
# MAGIC
# MAGIC **MLflow Tracing** records each agent request as a structured **trace**: a tree of **spans**. Each span captures:
# MAGIC - **Inputs and outputs**: what went in, what came out
# MAGIC - **Timing**: start, end, duration
# MAGIC - **Span type**: `AGENT`, `CHAT_MODEL`, `TOOL`, `RETRIEVER`
# MAGIC - **Status**: success or failure
# MAGIC
# MAGIC ```
# MAGIC AGENT  handle_request
# MAGIC   ├── CHAT_MODEL  chat.completions
# MAGIC   ├── TOOL        estimate_baggage_fee
# MAGIC   └── CHAT_MODEL  chat.completions
# MAGIC ```
# MAGIC
# MAGIC If the agent calls supported libraries (OpenAI SDK, LangChain, etc.), child spans are generated **automatically**. For custom logic, add manual spans with `mlflow.start_span()`.
# MAGIC
# MAGIC ## C2. Traces in production
# MAGIC
# MAGIC Traces are logged to an **MLflow Experiment** backed by Unity Catalog: governed objects with lineage and access control. The experiment is the hub where observability and evaluation converge:
# MAGIC
# MAGIC - **Scorers evaluate traces**, not raw requests, so any traced agent can be evaluated with the same scorer infrastructure (Part E).
# MAGIC - **Trace archival** streams traces to Delta tables for SQL analysis and dashboards.
# MAGIC - **Metric backfill** applies new scorers retroactively to old traces.
# MAGIC
# MAGIC ```python
# MAGIC mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])   # injected via databricks.yml
# MAGIC ```
# MAGIC
# MAGIC For long-term retention you can bind an experiment to a **UC trace location**, which creates four Delta tables (`<prefix>_otel_spans`, `_otel_annotations`, `_otel_logs`, `_otel_metrics`): SQL-queryable, UC-governed. (Note: `ALL_PRIVILEGES` is **not** sufficient, `MODIFY` + `SELECT` must be granted explicitly, and the binding cannot be reassigned later.)

# COMMAND ----------

# DBTITLE 1,C3 header: mini agent loop demo
# MAGIC %md
# MAGIC ### C3. Demo: instrument a mini agent loop and inspect the trace (in this notebook's experiment)

# COMMAND ----------

# DBTITLE 1,C3.1 instrument the mini agent loop
import mlflow

# MLflow tracing in a Databricks notebook automatically logs traces to this
# notebook's MLflow experiment (the experiment icon at the top of the notebook page).
# In the deployed app (Notebook 2) the experiment is instead injected via
# databricks.yml as MLFLOW_EXPERIMENT_NAME.


@mlflow.trace
def preprocess(user_message: str) -> str:
    # Manual child span for custom logic, powered by the governed mask_contact_info tool
    with mlflow.start_span(name="pii_scrub") as span:
        cleaned = client.execute_function(
            function_name=f"{CATALOG}.{SCHEMA}.mask_contact_info",
            parameters={"text": user_message},
        ).value
        span.set_inputs({"raw": user_message})
        span.set_outputs({"cleaned": cleaned})
    return cleaned


@mlflow.trace
def mini_agent(user_message: str) -> str:
    """Toy agent loop: preprocess → call UC tool → compose answer."""
    cleaned = preprocess(user_message)

    # TOOL span: the exact governed tool we registered in Part A, executed via UC
    result = client.execute_function(
        function_name=f"{CATALOG}.{SCHEMA}.estimate_baggage_fee",
        parameters={"weight_kg": 28.0, "cabin_class": "economy", "loyalty_tier": "gold"},
    )
    with mlflow.start_span(name="compose_answer") as span:
        answer = f"Regarding '{cleaned}': {result.value}"
        span.set_outputs({"answer": answer})
    return answer


print(mini_agent("Email me at shriveens@platformatory.com, what's the baggage fee for a 28 kg economy bag, gold tier?"))
print("\n✅ Trace logged to this notebook's MLflow experiment")

# COMMAND ----------

# DBTITLE 1,C3.2 read the trace back
# C3.2: pull the trace we just logged back from the experiment
import time; time.sleep(3)   # allow trace finalization

traces = mlflow.search_traces(max_results=1)
if len(traces) > 0:
    print("Latest trace:", traces.iloc[0].trace_id)
    print("\nOpen the MLflow experiment (experiment icon at the top of this notebook)")
    print("and click the trace to inspect the span tree:")
    print("  mini_agent → preprocess → pii_scrub, plus the governed tool calls.")
else:
    print("Trace not visible yet: open the MLflow Experiment in the UI to see it.")

# COMMAND ----------

# DBTITLE 1,Part D: deployment architecture
# MAGIC %md
# MAGIC # Part D: Deployment Architecture *(theory here, applied in Notebook 2)*
# MAGIC
# MAGIC ## D1. The agent lifecycle: four stages
# MAGIC
# MAGIC **Deployment → Observability → Evaluation → Monitoring**
# MAGIC
# MAGIC | Stage | Tooling | What it does |
# MAGIC |---|---|---|
# MAGIC | **Deployment** | Declarative Automation Bundles + Databricks Apps | Package the agent, deploy as a managed web app with its own compute, service principal, and URL |
# MAGIC | **Observability** | MLflow Tracing | Every request produces a structured trace: inputs, outputs, latency, execution flow *(Part C)* |
# MAGIC | **Evaluation** | Scorers + Judges | Attach scorers to the MLflow Experiment that automatically assess trace quality |
# MAGIC | **Monitoring** | Backfill + Archival | Manage scorer lifecycles, backfill historical traces, archive to Delta tables |
# MAGIC
# MAGIC These stages are **not strictly sequential**: the lifecycle is incremental. You can add tracing before deployment, attach scorers to a live agent, or backfill weeks of old traces.
# MAGIC
# MAGIC ## D2. Where does an agent live in production? Endpoints vs. Apps
# MAGIC
# MAGIC Databricks gives you two deployment targets, and choosing between them is a real architecture decision:
# MAGIC
# MAGIC **Option 1: the model serving endpoint (a deployed agent).** You log the agent as an MLflow model, register it in Unity Catalog, and deploy it to a **serving endpoint**. The registered model is the artifact; the endpoint exposes a REST API (prompt in, answer out). This is the natural evolution of Module 5A's MLflow `PythonModel`: it works, it scales, and it is fully governed by UC. Many single-agent, model-centric use cases stop here, and that is fine.
# MAGIC
# MAGIC **Option 2: the Databricks App.** The agent **code itself** is the deployment artifact, packaged in a DAB. A managed web app runs your `agent.py` + `server.py` on platform compute, with its own service principal and workspace URL.
# MAGIC
# MAGIC | Concern | Serving endpoint (registered model/agent) | Databricks App (code artifact) |
# MAGIC |---|---|---|
# MAGIC | **What is deployed** | A registered model (MLflow/UC) | Application code (DAB) |
# MAGIC | **Inference** | The endpoint runs the model directly | The app **calls** a serving endpoint for inference |
# MAGIC | **Scaling** | Managed by serving infrastructure | Managed by the Apps platform |
# MAGIC | **Customization** | Limited to model configuration | Full control: custom UI, middleware, routing |
# MAGIC | **Governance** | UC model permissions | App-level ACLs + UC permissions on accessed resources |
# MAGIC | **Tool integration** | Defined in the registered model | MCP servers, direct API calls, any Python library |
# MAGIC
# MAGIC **Why transition to an App?** A registered agent answers prompts, but a production *assistant* needs much more around the model: **runtime tool discovery via MCP** (register a new UC tool and the app uses it with zero redeploy), **multi-agent orchestration** (supervisor + specialists with handoffs), custom HTTP endpoints and UX (`/chat`, `/health`), and business logic such as PII masking and escalation rules. Squeezing all of that into a model artifact is fragile. The App is the agent's *home*; orchestration, tools, and observability wiring live in app code.
# MAGIC
# MAGIC Note what does **not** change: the app still calls a model serving endpoint for the LLM. Apps replace the *orchestration* layer, not the model layer.
# MAGIC
# MAGIC **Key characteristics of a Databricks App:** managed compute (no clusters/VMs/containers), its own **service principal** identity, a **unique workspace URL**, and first-class workspace presence on the Apps page.
# MAGIC
# MAGIC ## D3. Declarative Automation Bundles (DABs)
# MAGIC
# MAGIC A DAB is the project structure describing the app and its resources:
# MAGIC
# MAGIC **User-authored source files**: `agent.py` (agent logic), `server.py` (FastAPI wrapper exposing HTTP endpoints), `pyproject.toml` (dependencies).
# MAGIC
# MAGIC **Configuration files**: `databricks.yml` (bundle declaration: app name, resources, permissions) and `app.yaml` (runtime manifest: startup command + `env:` injection).
# MAGIC
# MAGIC Deployment is a **two-step process**:
# MAGIC
# MAGIC ```bash
# MAGIC # Step 1: Deploy the bundle (creates/updates resources)
# MAGIC databricks bundle deploy
# MAGIC
# MAGIC # Step 2: Deploy the app source code
# MAGIC databricks apps deploy my-agent-app --source-code-path /Workspace/Users/<you>/my-agent-app
# MAGIC ```
# MAGIC
# MAGIC The two steps are **independent**: redeploy code without touching resource declarations, or update resources without redeploying code. This separation makes iteration fast.
# MAGIC
# MAGIC ## D4. Resource injection: decoupling code from infrastructure
# MAGIC
# MAGIC `databricks.yml` **declares** every resource the app depends on; `app.yaml` **injects** them as environment variables. Agent code reads configuration from the environment, never hardcoded:
# MAGIC
# MAGIC ```yaml
# MAGIC # databricks.yml (excerpt): declares resources
# MAGIC resources:
# MAGIC   apps:
# MAGIC     my_agent:
# MAGIC       name: my-agent-app
# MAGIC       source_code_path: ./src
# MAGIC       config:
# MAGIC         command: ["uv", "run", "start-server"]
# MAGIC         env:
# MAGIC           - name: SERVING_ENDPOINT
# MAGIC             value: my-llm-endpoint
# MAGIC           - name: UC_TOOL_CATALOG
# MAGIC             value: module5b
# MAGIC           - name: MLFLOW_EXPERIMENT_NAME
# MAGIC             value: /Users/me/module5b_skyjet_traces
# MAGIC ```
# MAGIC
# MAGIC ```python
# MAGIC # agent code: same code runs in dev, staging, production
# MAGIC import os
# MAGIC serving_endpoint = os.environ["SERVING_ENDPOINT"]
# MAGIC catalog          = os.environ["UC_TOOL_CATALOG"]
# MAGIC ```
# MAGIC
# MAGIC > 💡 The same agent code runs in development, staging, and production by changing **only** the `databricks.yml` declarations. This is exactly the `os.environ.get()` pattern we previewed in Parts A and C.
# MAGIC
# MAGIC ## D5. Unity AI Gateway: the governance layer in front of production traffic
# MAGIC
# MAGIC When a deployed agent calls a serving endpoint, the request passes through the **Unity AI Gateway**:
# MAGIC
# MAGIC **Databricks App (agent code) → Unity AI Gateway → Serving Endpoint (foundation model)**
# MAGIC
# MAGIC | Capability | Description |
# MAGIC |---|---|
# MAGIC | **Guardrails** | Safety and content policies enforced at the gateway level |
# MAGIC | **Rate limiting** | Consumption limits per endpoint: capacity and cost control |
# MAGIC | **Usage tracking** | Cost analysis by endpoint, model, principal, tags via billing system tables |
# MAGIC | **Traffic splitting** | Distribute requests across model backends for A/B tests, gradual rollouts, **fallback** |
# MAGIC | **Inference tables** | Log requests/responses to UC Delta tables for auditing |
# MAGIC
# MAGIC The gateway is **transparent to agent code**: no changes needed, it is configured at platform level. It also governs MCP servers and coding agents, giving platform teams one place to manage AI access. *(Module 5A introduced the AI Gateway conceptually; in 5B it becomes a deployment concern.)*

# COMMAND ----------

# DBTITLE 1,Part E: evaluation & monitoring (road ahead)
# MAGIC %md
# MAGIC # Part E: Evaluation & monitoring (the road ahead)
# MAGIC
# MAGIC Once an agent is deployed **and** traced, the remaining lifecycle stages become possible:
# MAGIC
# MAGIC - **Scorers**: attach automatic assessments to the MLflow Experiment: **LLM judges** score dimensions like correctness, relevance, groundedness, and safety (the “LLM-as-judge” you saw teased in 5A §6.5); **custom Python scorers** encode business rules; **guideline scorers** check policies.
# MAGIC - **Backfill**: apply new scorers **retroactively** to traces logged before the scorer existed.
# MAGIC - **Archival**: stream traces to Delta tables for SQL-based reporting and dashboards (the UC trace tables from Part C).
# MAGIC
# MAGIC > 🔑 **Tracing must come first.** If the agent doesn't produce traces, there is nothing for scorers to evaluate. The first step of any monitoring setup is ensuring traces flow to an MLflow Experiment, which is why Part C preceded this section.
# MAGIC
# MAGIC ## Hands-on practice after this notebook
# MAGIC
# MAGIC | Exercise | What it drills |
# MAGIC |---|---|
# MAGIC | **Fix a sloppy tool** | Take any 5A `@function_tool`, sharpen its docstring, register it to UC with `create_python_function()`, verify via MCP: the Part A skill, repeated until automatic |
# MAGIC | **Build a Supervisor Agent with Agent Bricks** | The no-code route: route questions to a Knowledge Assistant + UC tools from the Agent Bricks UI, and compare with the code-first supervisor pattern |
# MAGIC | **Same agent in LangChain** | Rebuild the Part C mini agent with `@tool` and `create_react_agent`: same governed tools, same tracing, different framework, to see how the platform patterns carry over |
# MAGIC | **Notebook 2 (the lab)** | **Everything above assembled into one production-ready application** |

# COMMAND ----------

# DBTITLE 1,Wrap-up: the 5A to 5B map
# MAGIC %md
# MAGIC # Wrap-up: the 5A → 5B map
# MAGIC
# MAGIC | Module 5A (what you knew) | Module 5B (what you know now) |
# MAGIC |---|---|
# MAGIC | SQL tools via `CREATE FUNCTION` (UC, Demo 4) | *unchanged*: they were already governed |
# MAGIC | Python tools via `@function_tool` (notebook-only) | **Python tools registered to UC** via `DatabricksFunctionClient.create_python_function()` |
# MAGIC | RAG pipeline as notebook code (Demo 3) | **Retrieval becomes a governed UC tool**, discoverable via MCP (built in Notebook 2) |
# MAGIC | “What is MCP?” (awareness only) | **MCP in depth**: managed/external/custom servers, `tools/list`, `tools/call`, lifecycle, `McpServer.from_uc_function`, plus the Agent constructor surface (`tools` vs `mcp_servers` vs `handoffs`, B6) |
# MAGIC | Print-and-pray testing | **MLflow Tracing**: spans, `@mlflow.trace`, production traces, UC-backed trace tables |
# MAGIC | MLflow PythonModel deployment | **Databricks Apps + DABs**: two-step deploy, resource injection, service principals, AI Gateway |
# MAGIC | LLM-as-judge teaser | **Scorers, backfill, archival**: the monitoring pipeline (applied in Notebook 2) |
# MAGIC
# MAGIC ## Self-check
# MAGIC
# MAGIC 1. Why is a `@function_tool` function *not* an acceptable production tool, even though the agent calls it perfectly well?
# MAGIC 2. What two things does `create_python_function()` read from your Python function to build the UC signature, and what does the LLM read from it?
# MAGIC 3. In the MCP lifecycle, which request discovers tools, and which invokes them?
# MAGIC 4. Why does a Databricks App still need a model serving endpoint?
# MAGIC 5. What must exist **before** any scorer can evaluate your agent?
# MAGIC
# MAGIC **In Notebook 2 (the lab):** we assemble everything (governed SQL tools + governed Python tools + a governed RAG retrieval tool over the SkyJet policies PDF + `@function_tool` glue + MCP discovery + supervisor handoffs + MLflow tracing + unit and quality tests + a real DAB deployment) into a single **production-ready SkyJet passenger-support application**.