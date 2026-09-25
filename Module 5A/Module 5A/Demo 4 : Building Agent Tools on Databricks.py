# Databricks notebook source
# DBTITLE 1,Demo 4 Title
# MAGIC %md
# MAGIC # Demo 4: Building Agent Tools on Databricks
# MAGIC
# MAGIC This demo covers agentic AI fundamentals: from single-call LLM interactions to multi-step agent loops, Unity Catalog functions as governed tools, common tool patterns, and the AI Gateway.

# COMMAND ----------

# DBTITLE 1,Setup - Catalog, Schema, Sample Data
# MAGIC %sql
# MAGIC -- SETUP: Create catalog, schema, sample data, and UC functions
# MAGIC -- We create a dedicated catalog/schema for this demo with a
# MAGIC -- customer_orders table that simulates an e-commerce system.
# MAGIC -- UC functions registered here serve as governed agent tools.
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS module5a_demo4;
# MAGIC CREATE SCHEMA IF NOT EXISTS module5a_demo4.agent_tools;
# MAGIC
# MAGIC -- Sample data table for agent tool demos
# MAGIC CREATE OR REPLACE TABLE module5a_demo4.agent_tools.customer_orders (
# MAGIC   order_id     STRING NOT NULL,
# MAGIC   customer_id STRING NOT NULL,
# MAGIC   customer_tier STRING,
# MAGIC   order_total  DOUBLE,
# MAGIC   status       STRING,
# MAGIC   order_date   DATE,
# MAGIC   CONSTRAINT pk_customer_orders PRIMARY KEY (order_id)
# MAGIC ) TBLPROPERTIES (
# MAGIC   delta.enableChangeDataFeed = true
# MAGIC );
# MAGIC
# MAGIC INSERT INTO module5a_demo4.agent_tools.customer_orders VALUES
# MAGIC ('ORD-001', 'CUST-001', 'premium', 249.99, 'shipped',    '2026-09-15'),
# MAGIC ('ORD-002', 'CUST-001', 'premium', 89.50,  'delivered',  '2026-09-18'),
# MAGIC ('ORD-003', 'CUST-002', 'standard', 159.00, 'pending',   '2026-09-20'),
# MAGIC ('ORD-004', 'CUST-002', 'standard', 45.00,  'cancelled', '2026-09-22'),
# MAGIC ('ORD-005', 'CUST-003', 'basic',   310.75, 'shipped',    '2026-09-24'),
# MAGIC ('ORD-006', 'CUST-003', 'basic',   78.25,  'processing', '2026-09-25'),
# MAGIC ('ORD-007', 'CUST-001', 'premium', 199.50, 'delivered',  '2026-09-10'),
# MAGIC ('ORD-008', 'CUST-002', 'standard', 525.00, 'pending',   '2026-09-25');
# MAGIC
# MAGIC SELECT * FROM module5a_demo4.agent_tools.customer_orders ORDER BY order_id;

# COMMAND ----------

# DBTITLE 1,Setup - Register UC Functions as Tools
# MAGIC %sql
# MAGIC -- SETUP: Register UC functions as governed agent tools
# MAGIC -- These functions are discovered and called by AI agents.
# MAGIC -- Unity Catalog governs access: GRANT/REVOKE EXECUTE on each function.
# MAGIC
# MAGIC -- Tool 1: Get order status by order ID
# MAGIC CREATE OR REPLACE FUNCTION module5a_demo4.agent_tools.get_order_status(order_id STRING)
# MAGIC RETURNS STRING
# MAGIC COMMENT 'Returns the current status of an order (shipped, delivered, pending, cancelled, processing)'
# MAGIC RETURN (SELECT max(status) FROM module5a_demo4.agent_tools.customer_orders
# MAGIC        WHERE order_id = get_order_status.order_id);
# MAGIC
# MAGIC -- Tool 2: Calculate discount based on order total and customer tier
# MAGIC CREATE OR REPLACE FUNCTION module5a_demo4.agent_tools.calculate_discount(order_total DOUBLE, customer_tier STRING)
# MAGIC RETURNS DOUBLE
# MAGIC COMMENT 'Calculates discount amount: premium=15%, standard=5%, basic=0%'
# MAGIC RETURN CASE
# MAGIC   WHEN customer_tier = 'premium'  THEN round(order_total * 0.15, 2)
# MAGIC   WHEN customer_tier = 'standard' THEN round(order_total * 0.05, 2)
# MAGIC   ELSE 0.0
# MAGIC END;
# MAGIC
# MAGIC -- Tool 3: Get order count for a customer
# MAGIC CREATE OR REPLACE FUNCTION module5a_demo4.agent_tools.get_order_count(customer_id STRING)
# MAGIC RETURNS INT
# MAGIC COMMENT 'Returns the total number of orders for a customer'
# MAGIC RETURN SELECT count(*) FROM module5a_demo4.agent_tools.customer_orders
# MAGIC        WHERE customer_id = get_order_count.customer_id;
# MAGIC
# MAGIC -- Tool 4: Get total spending for a customer
# MAGIC CREATE OR REPLACE FUNCTION module5a_demo4.agent_tools.get_total_spending(customer_id STRING)
# MAGIC RETURNS DOUBLE
# MAGIC COMMENT 'Returns total spending for a customer across all orders'
# MAGIC RETURN SELECT coalesce(sum(order_total), 0.0) FROM module5a_demo4.agent_tools.customer_orders
# MAGIC        WHERE customer_id = get_total_spending.customer_id;
# MAGIC
# MAGIC -- Verify all tools work
# MAGIC SELECT
# MAGIC   module5a_demo4.agent_tools.get_order_status('ORD-001') AS ord_001_status,
# MAGIC   module5a_demo4.agent_tools.calculate_discount(100.0, 'premium') AS premium_discount,
# MAGIC   module5a_demo4.agent_tools.get_order_count('CUST-001') AS cust_001_orders,
# MAGIC   module5a_demo4.agent_tools.get_total_spending('CUST-003') AS cust_003_spending;

# COMMAND ----------

# DBTITLE 1,4.1 Notes - From RAG to Agents
# MAGIC %md
# MAGIC ## 4.1 : From RAG to Agents
# MAGIC
# MAGIC ### Concepts
# MAGIC * **Classic RAG**: Fixed pipeline - retrieve context, then generate. One retrieve, one generate. No decisions in between.
# MAGIC * **Agent**: A system that can *reason*, *decide*, *act*, and *observe* in a loop. It chooses which tools to call and interprets the results.
# MAGIC * **Key difference**: RAG follows a fixed path (retrieve -> generate). An agent follows a dynamic path (reason -> act -> observe -> reason -> ... -> respond).
# MAGIC * **Why agents**: When the question requires multiple steps, tool calls, or conditional logic, a fixed RAG pipeline cannot adapt.
# MAGIC
# MAGIC RAG is a special case of an agent with exactly one tool (retrieval) and one step.

# COMMAND ----------

# DBTITLE 1,4.1 Demo - Single Call vs Agent Loop
# 4.1 Demo: Single-call (RAG-like) vs. multi-step agent loop
# A single call asks the LLM directly. An agent loop iterates:
# reason -> act (call tool) -> observe -> respond.

# --- Single call (like RAG: one generate, no tools) ---
print("=== Single Call (RAG-like) ===")
single = spark.sql("""
  SELECT ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    'What is the total revenue from all customer orders? Guess based on general knowledge.',
    modelParameters => named_struct('temperature', 0.0, 'max_tokens', 100)
  ) AS response
""").collect()[0][0]
print(f"  {single}")
print()

# --- Agent loop: reason -> act -> observe -> respond ---
print("=== Agent Loop (reason -> act -> observe -> respond) ===")

# Step 1: Reason - the agent decides it needs the actual order data
print("  Step 1 (Reason): Agent decides to call: get_total_revenue()")

# Step 2: Act - execute the tool (SQL query)
total_revenue = spark.sql("""
  SELECT sum(order_total) AS total
  FROM module5a_demo4.agent_tools.customer_orders
""").collect()[0][0]
order_count = spark.sql("""
  SELECT count(*) AS cnt
  FROM module5a_demo4.agent_tools.customer_orders
""").collect()[0][0]
print(f"  Step 2 (Act): Tool returned: revenue=${total_revenue:,.2f}, orders={order_count}")

# Step 3: Observe - feed the tool result back to the LLM
# Step 4: Respond - LLM generates the final answer using the tool output
print("  Step 3-4 (Observe & Respond):")
agent = spark.sql(f"""
  SELECT ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    'Tool result: total revenue = ${total_revenue:,.2f} across {order_count} orders. Summarize this for the user in one sentence.',
    modelParameters => named_struct('temperature', 0.0, 'max_tokens', 100)
  ) AS response
""").collect()[0][0]
print(f"  {agent}")

# COMMAND ----------

# DBTITLE 1,4.2 Notes - What is an AI Agent
# MAGIC %md
# MAGIC ## 4.2 : What is an AI Agent
# MAGIC
# MAGIC ### Concepts
# MAGIC An AI agent is a system that:
# MAGIC * **Perceives**: Reads user input and context
# MAGIC * **Reasons**: Uses an LLM to think about what to do next
# MAGIC * **Plans**: Decides which tools to call and in what order
# MAGIC * **Adapts**: Changes its plan based on tool results (if a tool fails, try another)
# MAGIC * **Acts**: Calls tools, queries data, interacts with external systems
# MAGIC * **Learns**: Uses conversation history and past results to improve
# MAGIC
# MAGIC Unlike a single LLM call (one prompt, one response), an agent iterates through multiple steps until it reaches a satisfactory answer.
# MAGIC
# MAGIC A traditional system follows a fixed path. An agent *chooses* its path based on the situation.

# COMMAND ----------

# DBTITLE 1,4.2 Demo - Agent Anatomy
# 4.2 Demo: Agent anatomy - the components that make up an agent
# Every agent has: system prompt, user message, tool definitions, and a response.

print("=== Agent Anatomy ===")
print()

# 1. System prompt - defines the agent's role and behavior
system_prompt = """You are a customer service agent for an e-commerce platform.
You help customers with order status, discounts, and account questions.
Always be concise and friendly."""
print(f"1. System Prompt:\n   {system_prompt[:120]}...")
print()

# 2. Available tools - what the agent can call
tools = [
    {"name": "get_order_status", "params": "order_id: string", "returns": "string"},
    {"name": "calculate_discount", "params": "order_total: double, customer_tier: string", "returns": "double"},
    {"name": "get_order_count", "params": "customer_id: string", "returns": "int"},
    {"name": "get_total_spending", "params": "customer_id: string", "returns": "double"},
]
print("2. Available Tools:")
for t in tools:
    print(f"   - {t['name']}({t['params']}) -> {t['returns']}")
print()

# 3. User message - what the user is asking
user_msg = "What is the status of order ORD-001?"
print(f"3. User Message:\n   {user_msg}")
print()

# 4. Agent response - the full interaction
# The agent reasons about which tool to call, calls it, and responds
print("4. Agent Interaction:")
print("   Reasoning: 'User wants order status. I should call get_order_status(ORD-001)'")
status = spark.sql("SELECT module5a_demo4.agent_tools.get_order_status('ORD-001')").collect()[0][0]
print(f"   Tool call: get_order_status('ORD-001') -> '{status}'")
print(f"   Response: 'Order ORD-001 is currently {status}.'")

# COMMAND ----------

# DBTITLE 1,4.3 Notes - Agent Framework Anatomy
# MAGIC %md
# MAGIC ## 4.3 : Agent Framework Anatomy
# MAGIC
# MAGIC ### Concepts
# MAGIC An agent framework provides the structure for building agents:
# MAGIC
# MAGIC 1. **System prompt**: Defines the agent's role, capabilities, and constraints
# MAGIC 2. **User prompt**: The user's question or request
# MAGIC 3. **Tool definitions**: Schema describing available tools (name, parameters, return type)
# MAGIC 4. **LLM as decision engine**: The model decides which tool to call and when
# MAGIC 5. **Tool execution**: The framework executes the tool and returns the result
# MAGIC 6. **Response synthesis**: The LLM combines tool results into a final answer
# MAGIC
# MAGIC The flow: `system_prompt + user_prompt -> LLM decides -> tool_call -> execute -> observe -> LLM responds`
# MAGIC
# MAGIC On Databricks, tools are often UC functions - governed, versioned, and auditable.

# COMMAND ----------

# DBTITLE 1,4.3 Demo - Full Agent Framework
# 4.3 Demo: Full agent framework interaction
# Show the complete flow: system prompt -> tool decision -> execution -> response

import json

print("=== Agent Framework: Full Interaction ===")
print()

# 1. System prompt with tool descriptions
system_prompt = """You are a customer service agent. You have these tools:
- get_order_status(order_id): Get order status
- calculate_discount(order_total, customer_tier): Calculate discount
- get_order_count(customer_id): Get order count
- get_total_spending(customer_id): Get total spending

Answer customer questions by calling the appropriate tools."""

# 2. User asks a multi-part question
user_question = "How many orders does CUST-001 have and what is their total spending?"
print(f"User: {user_question}")
print()

# 3. The agent framework parses the question and decides tool calls
# (In production, the LLM output includes structured tool-call instructions)
print("Agent reasoning: Need order count AND total spending for CUST-001")
print("Planning: Call get_order_count('CUST-001') AND get_total_spending('CUST-001')")
print()

# 4. Execute the tool calls
order_count = spark.sql("""
  SELECT module5a_demo4.agent_tools.get_order_count('CUST-001')
""").collect()[0][0]
total_spending = spark.sql("""
  SELECT module5a_demo4.agent_tools.get_total_spending('CUST-001')
""").collect()[0][0]

print(f"Tool results:")
print(f"  get_order_count('CUST-001') -> {order_count}")
print(f"  get_total_spending('CUST-001') -> ${total_spending:,.2f}")
print()

# 5. LLM synthesizes the final response from tool outputs
final = spark.sql(f"""
  SELECT ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    'Tool results: customer CUST-001 has {order_count} orders and total spending of ${total_spending:,.2f}. Write a brief, friendly response to the customer.',
    modelParameters => named_struct('temperature', 0.0, 'max_tokens', 100)
  ) AS response
""").collect()[0][0]
print(f"Agent response: {final}")

# COMMAND ----------

# DBTITLE 1,4.4 Notes - UC Functions as Agent Tools
# MAGIC %md
# MAGIC ## 4.4 : Unity Catalog Functions as Governed Agent Tools
# MAGIC
# MAGIC ### Concepts
# MAGIC * **UC functions** are the native way to create governed, reusable tools for agents on Databricks.
# MAGIC * **How it works**: Write a Python or SQL function, register it in Unity Catalog, and the agent discovers and calls it.
# MAGIC * **Governance**: UC controls who can `EXECUTE` each function - no ungoverned tool access.
# MAGIC * **Auditing**: Every tool call is logged through UC audit logs.
# MAGIC * **Versioning**: Functions can be updated without changing agent code.
# MAGIC * **Discovery**: Agents query UC information schema to find available tools.
# MAGIC
# MAGIC This is the Databricks equivalent of OpenAI function calling - but with enterprise governance built in.
# MAGIC
# MAGIC > UC functions support both SQL and Python. SQL functions are ideal for data queries. Python functions handle external API calls, transformations, and complex logic.

# COMMAND ----------

# DBTITLE 1,4.4 Demo - Discover UC Function Tools
# MAGIC %sql
# MAGIC -- 4.4 Demo: List and inspect registered UC functions as agent tools
# MAGIC -- Agents discover tools by querying the UC information schema.
# MAGIC -- Each function's comment serves as the tool description for the LLM.
# MAGIC
# MAGIC -- List all functions in our schema (this is what an agent does to discover tools)
# MAGIC SELECT
# MAGIC   routine_name AS function_name,
# MAGIC   comment AS tool_description,
# MAGIC   routine_type
# MAGIC FROM module5a_demo4.information_schema.routines
# MAGIC WHERE routine_schema = 'agent_tools'
# MAGIC ORDER BY function_name;
# MAGIC
# MAGIC -- Show the full function definition for one tool (agents use this to understand parameters)
# MAGIC DESCRIBE FUNCTION EXTENDED module5a_demo4.agent_tools.calculate_discount;

# COMMAND ----------

# DBTITLE 1,4.4b Demo - Agent Tool Calling
# 4.4b Demo: Agent tool calling - the LLM decides which UC function to call
# We simulate the tool-calling pattern: LLM -> tool decision -> execution -> response

print("=== Agent Tool Calling with UC Functions ===")
print()

# The agent receives a question and has access to UC function tools
question = "What discount would a premium customer get on a $200 order?"
print(f"User: {question}")
print()

# Step 1: LLM reasons about which tool to call
# (In a real agent framework, the LLM returns a structured tool-call JSON)
print("Step 1: LLM decides to call: calculate_discount(200.0, 'premium')")

# Step 2: Framework executes the UC function
discount = spark.sql("""
  SELECT module5a_demo4.agent_tools.calculate_discount(200.0, 'premium')
""").collect()[0][0]
print(f"Step 2: Tool returned: ${discount:.2f}")
print()

# Step 3: LLM generates a natural response from the tool result
response = spark.sql(f"""
  SELECT ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    'Tool result: calculate_discount(200.0, premium) returned {discount:.2f}. Answer the user: "What discount would a premium customer get on a $200 order?"',
    modelParameters => named_struct('temperature', 0.0, 'max_tokens', 80)
  ) AS response
""").collect()[0][0]
print(f"Step 3: Agent response: {response}")
print()

# Now a multi-tool scenario: check order status AND calculate discount
print("--- Multi-Tool Scenario ---")
print("User: What is the status of ORD-005 and what discount would CUST-003 get?")

status = spark.sql("SELECT module5a_demo4.agent_tools.get_order_status('ORD-005')").collect()[0][0]
tier = spark.sql("SELECT customer_tier FROM module5a_demo4.agent_tools.customer_orders WHERE customer_id = 'CUST-003' LIMIT 1").collect()[0][0]
order_total = spark.sql("SELECT order_total FROM module5a_demo4.agent_tools.customer_orders WHERE order_id = 'ORD-005'").collect()[0][0]
discount = spark.sql(f"SELECT module5a_demo4.agent_tools.calculate_discount({order_total}, '{tier}')").collect()[0][0]

print(f"  get_order_status('ORD-005') -> '{status}'")
print(f"  customer_tier for CUST-003 -> '{tier}'")
print(f"  calculate_discount({order_total}, '{tier}') -> ${discount:.2f}")

# COMMAND ----------

# DBTITLE 1,4.5 Notes - Common Tool Patterns
# MAGIC %md
# MAGIC ## 4.5 : Common Tool Patterns
# MAGIC
# MAGIC ### Concepts
# MAGIC Agents use tools for different purposes. The most common patterns:
# MAGIC
# MAGIC 1. **Structured data retrieval**: Query a SQL table or UC function for structured data
# MAGIC    - Example: `get_order_status('ORD-001')` returns a status string
# MAGIC
# MAGIC 2. **Unstructured retrieval (RAG)**: Search a vector index for semantically similar content
# MAGIC    - Example: `vector_search(index, query_text)` returns relevant document chunks
# MAGIC
# MAGIC 3. **Code interpreter**: Execute generated code in a sandbox
# MAGIC    - Example: Agent writes and runs Python to compute a custom metric
# MAGIC
# MAGIC 4. **External connections**: Call external APIs via UC connections
# MAGIC    - Example: Fetch weather data, call a payment API, send an email
# MAGIC
# MAGIC 5. **AI Function chaining**: Use ai_classify, ai_extract, ai_summarize as tools in a pipeline
# MAGIC    - Example: Classify a support ticket, then extract key fields, then summarize

# COMMAND ----------

# DBTITLE 1,4.5 Demo - Common Tool Patterns
# 4.5 Demo: Common tool patterns that agents use

print("=== Pattern 1: Structured Data Retrieval (UC Function) ===")
status = spark.sql("""
  SELECT module5a_demo4.agent_tools.get_order_status('ORD-003') AS status
""").collect()[0][0]
print(f"  Tool: get_order_status('ORD-003') -> '{status}'")
print()

print("=== Pattern 2: Unstructured Retrieval (RAG via ai_query) ===")
rag = spark.sql("""
  SELECT ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    'Context: "AI agents use tools to interact with external systems and data sources. UC functions provide governed, auditable tools that agents can discover and call." Question: What do UC functions provide for agents?',    modelParameters => named_struct('temperature', 0.0, 'max_tokens', 80)
  ) AS response
""").collect()[0][0]
print(f"  RAG response: {rag}")
print()

print("=== Pattern 3: AI Function as a Tool (Classification) ===")
category = spark.sql("""
  SELECT ai_classify(
    'Customer wants to cancel order ORD-004 and request a full refund immediately',
    '{"billing":"Billing or payment issues","cancellation":"Order cancellations and refunds","support":"General customer support","complaint":"Formal complaints"}',
    MAP('version', '2.1')
  ) AS category
""").collect()[0][0]
print(f"  Classification: {category}")
print()

print("=== Pattern 4: AI Function as a Tool (Extraction) ===")
extracted = spark.sql("""
  SELECT ai_extract(
    'Order ORD-005 was shipped on September 24, 2026 to customer CUST-003. Total was $310.75.',
    '{"order_id": {"type": "string"}, "customer_id": {"type": "string"}, "amount": {"type": "number"}, "status": {"type": "string"}}',
    MAP('version', '2.1')
  ) AS fields
""").collect()[0][0]
print(f"  Extracted: {extracted}")

# COMMAND ----------

# DBTITLE 1,4.6 Notes - AI Playground
# MAGIC %md
# MAGIC ## 4.6 : AI Playground
# MAGIC
# MAGIC ### Concepts
# MAGIC * The **AI Playground** is a no-code interface for testing prompts, models, and tools before writing code.
# MAGIC * **What you can test**:
# MAGIC   * Compare different foundation models side-by-side
# MAGIC   * Adjust temperature, max_tokens, and system prompts
# MAGIC   * Test UC functions as tools (the Playground discovers them automatically)
# MAGIC   * Save prompts as templates for reuse
# MAGIC * **Workflow**: Prototype in Playground -> verify behavior -> wire into production code or agent.
# MAGIC * **Key benefit**: Zero setup - the Playground has access to all foundation models and UC functions you can access.
# MAGIC
# MAGIC > The Playground is the fastest way to validate that a prompt + tool combination works before committing to agent code.

# COMMAND ----------

# DBTITLE 1,4.7 Notes - Unity AI Gateway
# MAGIC %md
# MAGIC ## 4.7 : Unity AI Gateway
# MAGIC
# MAGIC ### Concepts
# MAGIC The **Unity AI Gateway** is a managed service that sits between your applications and LLM endpoints, providing:
# MAGIC
# MAGIC * **Guardrails**: Filter harmful content in both requests (input) and responses (output). Detect PII, block prompt injections.
# MAGIC * **Rate limits**: Control requests per minute, per user, or per token. Prevents abuse and controls costs.
# MAGIC * **Usage tracking**: Every request is logged to an inference table in Unity Catalog. Query usage by user, model, token count, or cost.
# MAGIC * **Traffic splitting & fallback routing**: Route traffic across multiple models. If the primary model fails, automatically fall back to a secondary.
# MAGIC   * Example: 80% to `databricks-meta-llama-3-3-70b-instruct`, 20% to `databricks-llama-4-maverick`
# MAGIC   * Fallback: If primary returns 5xx, route to `databricks-gpt-oss-20b`
# MAGIC * **Governed**: All gateway configurations are UC entities with access controls.
# MAGIC
# MAGIC > The AI Gateway is the production layer between your agent and the model. Without it, you have no guardrails, no rate limits, and no usage visibility.

# COMMAND ----------

# DBTITLE 1,4.7 Demo - AI Gateway Concepts
# 4.7 Demo: AI Gateway concepts - guardrails, routing, usage tracking
# The AI Gateway sits between applications and LLM endpoints.
# Here we demonstrate the concepts that the Gateway provides.

print("=== Unity AI Gateway: Key Features ===")
print()

# 1. Guardrails - content filtering (the Gateway filters requests/responses)
print("1. Guardrails: Filter harmful content in requests and responses")
print("   - Input guardrails: detect PII, block prompt injection")
print("   - Output guardrails: filter harmful responses")
print("   - Configured per gateway endpoint")
print()

# 2. Usage tracking - monitor token consumption
print("2. Usage Tracking: Every request logged to UC inference table")
print("   - Track: user, model, token count, latency, cost")
print("   - Query: SELECT user, sum(tokens) FROM inference_table GROUP BY user")
print()

# 3. Rate limiting - control request volume
print("3. Rate Limits: Control requests per minute/user")
print("   - Example: 100 requests/min for standard users, 1000/min for premium")
print("   - Prevents abuse and controls costs")
print()

# 4. Traffic splitting & fallback routing
print("4. Traffic Splitting & Fallback Routing:")
print("   Primary:   databricks-meta-llama-3-3-70b-instruct (80%)")
print("   Secondary: databricks-llama-4-maverick (20%)")
print("   Fallback:  databricks-gpt-oss-20b (if primary fails)")
print()

# Show a real LLM call that would go through the Gateway in production
print("=== Example: LLM call (would route through Gateway in production) ===")
response = spark.sql("""
  SELECT ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    'In one sentence, what is the Unity AI Gateway?',
    modelParameters => named_struct('temperature', 0.0, 'max_tokens', 80)
  ) AS response
""").collect()[0][0]
print(f"  {response}")

# COMMAND ----------

# DBTITLE 1,4.7b Demo - AI Gateway in Action
# 4.7b Demo: AI Gateway in Action - Guardrails, Routing, Fallback
# This demo shows AI Gateway concepts with real function calls.

import time

print("=== AI Gateway: Hands-on Demonstration ===")
print()

# 1. GUARDRAILS: PII detection with ai_mask
# The Gateway would apply this filter to every request automatically.
# Here we demonstrate the concept using ai_mask on real data.
print("--- 1. Guardrails: PII Redaction ---")
pii_result = spark.sql("""
  SELECT ai_mask(
    'Please update my card 4532-1234-5678-9012 and email john.doe@company.com',
    array('credit_card', 'email')
  ) AS masked
""").collect()[0][0]
print(f"  Original:  Please update my card 4532-1234-5678-9012 and email john.doe@company.com")
print(f"  Masked:    {pii_result}")
print(f"  -> In production, the Gateway applies this automatically to every request.")
print()

# 2. TRAFFIC SPLITTING: Route to two models and compare
# The Gateway can split traffic: e.g., 80% to primary, 20% to secondary.
# Here we simulate by calling both models on the same prompt.
print("--- 2. Traffic Splitting: Two Models, Same Prompt ---")
prompt = "In one word, what is the status of order ORD-001?"

primary = spark.sql(f"""
  SELECT ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    '{prompt}',
    modelParameters => named_struct('temperature', 0.0, 'max_tokens', 30)
  ) AS response
""").collect()[0][0]

secondary = spark.sql(f"""
  SELECT ai_query(
    'databricks-llama-4-maverick',
    '{prompt}',
    modelParameters => named_struct('temperature', 0.0, 'max_tokens', 30)
  ) AS response
""").collect()[0][0]

print(f"  Prompt:     {prompt}")
print(f"  Primary:    {primary.strip()[:80]}")
print(f"  Secondary:  {secondary.strip()[:80]}")
print(f"  -> Gateway routes 80% traffic to primary, 20% to secondary for comparison.")
print()

# 3. FALLBACK ROUTING: Try primary, fall back if it fails
# The Gateway automatically falls back to a backup model if the primary errors.
print("--- 3. Fallback Routing ---")
print(f"  Primary endpoint:   databricks-meta-llama-3-3-70b-instruct")
print(f"  Fallback endpoint:  databricks-llama-4-maverick")
print(f"  Strategy: if primary returns error, route to fallback automatically.")
print()

# 4. RATE LIMITING: Simulate request counting
print("--- 4. Rate Limiting ---")
start = time.time()
requests_made = 0
for i in range(3):
    spark.sql(f"""
      SELECT ai_query(
        'databricks-meta-llama-3-3-70b-instruct',
        'Reply with only the word: ok',
        modelParameters => named_struct('temperature', 0.0, 'max_tokens', 5)
      )
    """).collect()
    requests_made += 1
elapsed = time.time() - start
print(f"  Made {requests_made} requests in {elapsed:.2f}s")
print(f"  -> Gateway enforces: max 100 req/min for standard users, 1000/min for premium.")
print()

# 5. USAGE TRACKING: What the inference table would log
print("--- 5. Usage Tracking (Inference Table Schema) ---")
print("  Every Gateway request logs to a UC inference table:")
print("    - timestamp: when the request was made")
print("    - user: who made the request")
print("    - model: which model was called")
print("    - tokens_in: input token count")
print("    - tokens_out: output token count")
print("    - latency_ms: response time")
print("    - cost: dollar cost of the request")
print("    - status: success / error / blocked_by_guardrail")
print()

print("=== Summary ===")
print("The AI Gateway provides 4 production capabilities:")
print("  1. Guardrails - automatic PII filtering and content safety")
print("  2. Traffic splitting - route across models for A/B testing")
print("  3. Fallback routing - automatic recovery from failures")
print("  4. Rate limiting - prevent abuse and control costs")
print("  5. Usage tracking - full audit trail in UC inference tables")

# COMMAND ----------

# DBTITLE 1,Learning Conclusion
# MAGIC %md
# MAGIC ## Learning Conclusion
# MAGIC
# MAGIC ### What we demonstrated
# MAGIC
# MAGIC | Topic | What was demoed | Key Takeaway |
# MAGIC |---|---|---|
# MAGIC | 4.1 | Single call vs. agent loop | Agents iterate: reason -> act -> observe -> respond; RAG is a 1-step agent |
# MAGIC | 4.2 | Agent anatomy (system prompt, tools, response) | Every agent has: system prompt, tool definitions, user message, LLM decisions |
# MAGIC | 4.3 | Full agent framework interaction | LLM decides which tools to call -> framework executes -> LLM synthesizes |
# MAGIC | 4.4 | UC functions as governed agent tools | Register SQL/Python functions in UC; agents discover and call them with governance |
# MAGIC | 4.5 | Common tool patterns | Structured retrieval, RAG, code interpreter, external connections, AI functions |
# MAGIC | 4.6 | AI Playground | No-code prototyping of prompts, models, and UC tools before wiring code |
# MAGIC | 4.7 | Unity AI Gateway | Guardrails, rate limits, usage tracking, traffic splitting & fallback routing |
# MAGIC
# MAGIC ### Key principles
# MAGIC * **Agents = LLM + tools + loop**: The LLM decides, tools execute, results feed back.
# MAGIC * **UC functions are the native tool format**: Governed, versioned, auditable.
# MAGIC * **The AI Gateway is the production layer**: Guardrails, routing, and monitoring for LLM calls.
# MAGIC * **Start in Playground, go to production**: Prototype first, then wire into agent code.

# COMMAND ----------

# DBTITLE 1,Cleanup - Decommission Demo Artifacts
# CLEANUP: Drop all resources created in this demo
print("Dropping UC functions...")
for fn in ['get_order_status', 'calculate_discount', 'get_order_count', 'get_total_spending']:
    spark.sql(f"DROP FUNCTION IF EXISTS module5a_demo4.agent_tools.{fn}")
print("  Functions dropped")

print("Dropping schema and catalog...")
spark.sql("DROP SCHEMA IF EXISTS module5a_demo4.agent_tools CASCADE")
spark.sql("DROP CATALOG IF EXISTS module5a_demo4 CASCADE")
print("  Schema and catalog dropped")

print("\nCleanup complete!")