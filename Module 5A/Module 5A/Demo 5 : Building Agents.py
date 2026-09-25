# Databricks notebook source
# DBTITLE 1,Demo 5a Title
# MAGIC %md
# MAGIC # Demo 5: Building Agents
# MAGIC
# MAGIC This demo covers agent anatomy (the reason-act-observe loop), building a single agent with UC function tools, the @function_tool decorator, and how a single agent orchestrates its own tools.

# COMMAND ----------

# DBTITLE 1,Setup - Catalog, Schema, Sample Data
# MAGIC %sql
# MAGIC -- SETUP: Create catalog, schema, sample data, and UC functions
# MAGIC -- for building and testing a single agent.
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS module5a_demo5;
# MAGIC CREATE SCHEMA IF NOT EXISTS module5a_demo5.agent_tools;
# MAGIC
# MAGIC -- Sample support tickets table
# MAGIC CREATE OR REPLACE TABLE module5a_demo5.agent_tools.support_tickets (
# MAGIC   ticket_id    STRING NOT NULL,
# MAGIC   customer_id  STRING NOT NULL,
# MAGIC   category     STRING,
# MAGIC   priority     STRING,
# MAGIC   subject      STRING,
# MAGIC   description  STRING,
# MAGIC   status       STRING,
# MAGIC   created_at   TIMESTAMP
# MAGIC );
# MAGIC
# MAGIC INSERT INTO module5a_demo5.agent_tools.support_tickets VALUES
# MAGIC ('TKT-001', 'CUST-001', 'billing',   'high',   'Overcharged on order ORD-001', 'I was charged $249.99 but the product was on sale for $199.99', 'open', '2026-09-25 08:00:00'),
# MAGIC ('TKT-002', 'CUST-002', 'shipping', 'medium', 'Order ORD-003 not delivered',   'My order from Sept 20 has not arrived yet',                     'open', '2026-09-25 09:30:00'),
# MAGIC ('TKT-003', 'CUST-003', 'technical','high',   'Cannot access my account',     'I keep getting a 403 error when trying to log in',             'open', '2026-09-25 10:15:00'),
# MAGIC ('TKT-004', 'CUST-001', 'general',   'low',    'Product recommendation',      'Can you recommend accessories for my recent purchase?',        'open', '2026-09-25 11:00:00'),
# MAGIC ('TKT-005', 'CUST-002', 'billing',   'medium', 'Discount not applied',        'My premium discount was not applied to order ORD-008',         'open', '2026-09-25 11:45:00');
# MAGIC
# MAGIC -- UC functions as agent tools
# MAGIC CREATE OR REPLACE FUNCTION module5a_demo5.agent_tools.get_ticket(ticket_id STRING)
# MAGIC RETURNS STRING
# MAGIC COMMENT 'Returns the subject and description of a support ticket'
# MAGIC RETURN (SELECT max(concat_ws(' | ', subject, description))
# MAGIC        FROM module5a_demo5.agent_tools.support_tickets
# MAGIC        WHERE ticket_id = get_ticket.ticket_id);
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION module5a_demo5.agent_tools.count_tickets_by_category(cat STRING)
# MAGIC RETURNS INT
# MAGIC COMMENT 'Returns the number of tickets in a given category'
# MAGIC RETURN SELECT count(*) FROM module5a_demo5.agent_tools.support_tickets
# MAGIC        WHERE category = count_tickets_by_category.cat;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION module5a_demo5.agent_tools.get_priority(ticket_id STRING)
# MAGIC RETURNS STRING
# MAGIC COMMENT 'Returns the priority of a support ticket'
# MAGIC RETURN (SELECT max(priority) FROM module5a_demo5.agent_tools.support_tickets
# MAGIC        WHERE ticket_id = get_priority.ticket_id);
# MAGIC
# MAGIC SELECT * FROM module5a_demo5.agent_tools.support_tickets ORDER BY ticket_id;

# COMMAND ----------

# DBTITLE 1,Agent Anatomy Notes
# MAGIC %md
# MAGIC ## Agent Anatomy: The Reason-Act-Observe Loop
# MAGIC
# MAGIC ### Concepts
# MAGIC An AI agent is not just a single LLM call. It follows a loop:
# MAGIC
# MAGIC 1. **Reason**: The LLM analyzes the user's request and decides what to do next.
# MAGIC    - "The user wants to know about TKT-001. I should call the get_ticket tool."
# MAGIC
# MAGIC 2. **Act**: The agent executes a tool (UC function, API call, SQL query).
# MAGIC    - Calls `get_ticket('TKT-001')` -> "Overcharged on order ORD-001 | I was charged..."
# MAGIC
# MAGIC 3. **Observe**: The agent sees the tool's output and incorporates it.
# MAGIC    - "The ticket is about a billing overcharge. The priority is high."
# MAGIC
# MAGIC 4. **Respond**: The agent synthesizes a natural language response.
# MAGIC    - "Ticket TKT-001 is a high-priority billing issue about an overcharge on order ORD-001."
# MAGIC
# MAGIC **Key components of an agent**:
# MAGIC * **System prompt**: Defines the agent's role, capabilities, and constraints
# MAGIC * **Tools**: Functions the agent can call (UC functions, APIs, code)
# MAGIC * **Reasoning loop**: The reason-act-observe cycle that continues until the agent has enough information to respond
# MAGIC
# MAGIC > A single LLM call is a question-answer pair. An agent is a **sequence of decisions** that may involve multiple tool calls before responding.

# COMMAND ----------

# DBTITLE 1,Agent Anatomy Demo
# Agent Anatomy Demo: The reason-act-observe loop
# Show each step of the loop explicitly for a single ticket query.

user_question = "What is the status of ticket TKT-001?"

print("=== Agent Reason-Act-Observe Loop ===")
print()
print(f"USER: {user_question}")
print()

# Step 1: REASON
print("--- STEP 1: REASON ---")
reasoning = spark.sql(f"""
  SELECT ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    'You are a support agent. A user asks: "{user_question}". You have these tools available: get_ticket(ticket_id), get_priority(ticket_id), count_tickets_by_category(category). Which tool should you call first? Respond with just the tool name and argument.',
    modelParameters => named_struct('temperature', 0.0, 'max_tokens', 50)
  ) AS reasoning
""").collect()[0][0]
print(f"Agent thinks: {reasoning.strip()}")
print()

# Step 2: ACT (call get_ticket)
print("--- STEP 2: ACT (call get_ticket) ---")
ticket_info = spark.sql("SELECT module5a_demo5.agent_tools.get_ticket('TKT-001')").collect()[0][0]
print(f"Tool output: {ticket_info}")
print()

# Step 3: ACT again (call get_priority)
print("--- STEP 2b: ACT (call get_priority) ---")
priority = spark.sql("SELECT module5a_demo5.agent_tools.get_priority('TKT-001')").collect()[0][0]
print(f"Tool output: {priority}")
print()

# Step 4: OBSERVE
print("--- STEP 3: OBSERVE ---")
print(f"Agent now knows: Ticket TKT-001 is about '{ticket_info[:50]}...' with priority '{priority}'")
print()

# Step 5: RESPOND
print("--- STEP 4: RESPOND ---")
response = spark.sql(f"""
  SELECT ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    'You are a support agent. Based on this information, respond to the user naturally. Ticket info: "{ticket_info}". Priority: "{priority}". User question: "{user_question}"',
    modelParameters => named_struct('temperature', 0.0, 'max_tokens', 100)
  ) AS response
""").collect()[0][0]
print(f"Agent responds: {response.strip()}")

# COMMAND ----------

# DBTITLE 1,5.1 Notes - Single vs Multi Agent
# MAGIC %md
# MAGIC ## 5.1 : Single-Agent vs. Multi-Agent
# MAGIC
# MAGIC ### When to use a single agent
# MAGIC A single agent is the right choice when:
# MAGIC * **Few tools** (< 8): The LLM can manage all tools effectively
# MAGIC * **One domain**: All questions fall within one area of expertise
# MAGIC * **Short instructions**: The system prompt fits comfortably in context
# MAGIC * **Simple flow**: The reason-act-observe loop doesn't need delegation
# MAGIC
# MAGIC ### When to split into multi-agent
# MAGIC * **Many tools** (8-10+): The LLM loses track of which tool to use
# MAGIC * **Multiple domains**: Billing, shipping, technical need different expertise
# MAGIC * **Long instructions**: System prompt exceeds ~2000 tokens
# MAGIC * **Team ownership**: Different teams own different domains
# MAGIC
# MAGIC > This demo focuses on building a strong single agent. Demo 6 covers multi-agent orchestration.

# COMMAND ----------

# DBTITLE 1,Building a Single Agent Notes
# MAGIC %md
# MAGIC ## Building a Single Agent
# MAGIC
# MAGIC ### Concepts
# MAGIC A production single agent has three key pieces:
# MAGIC
# MAGIC 1. **System Prompt**: Defines the agent's persona, rules, and available tools
# MAGIC    - "You are a customer support agent. You can look up tickets, check priorities, and count tickets by category."
# MAGIC
# MAGIC 2. **Tool Definitions**: UC functions the agent can call
# MAGIC    - Each tool has a name, description, and parameter schema
# MAGIC    - The LLM uses the description to decide which tool to call
# MAGIC
# MAGIC 3. **Agent Loop**: The reason-act-observe cycle
# MAGIC    - The agent may call multiple tools in sequence before responding
# MAGIC    - Each tool result feeds back into the reasoning for the next step
# MAGIC
# MAGIC > On Databricks, UC functions provide governed, auditable tools. The agent discovers them through their UC descriptions.

# COMMAND ----------

# DBTITLE 1,Building a Single Agent Demo
# Building a Single Agent: Complete agent with system prompt + tools + reasoning loop
# The agent handles different question types by choosing the right tools.

print("=== Single Agent: Handling Multiple Question Types ===")
print()

questions = [
    ("What is ticket TKT-003 about?", "TKT-003", "get_ticket"),
    ("How many billing tickets do we have?", "billing", "count_tickets_by_category"),
    ("What is the priority of TKT-002?", "TKT-002", "get_priority")
]

for question, arg, expected_tool in questions:
    print(f"USER: {question}")
    
    # Step 1: REASON - Agent decides which tool to call
    decision = spark.sql(f"""
      SELECT ai_query(
        'databricks-meta-llama-3-3-70b-instruct',
        'You are a support agent with these tools: get_ticket(ticket_id), get_priority(ticket_id), count_tickets_by_category(category). User asks: "{question}". Which tool should you call? Respond with just the tool name.',
        modelParameters => named_struct('temperature', 0.0, 'max_tokens', 30)
      ) AS decision
    """).collect()[0][0]
    print(f"  REASON -> Agent decides to call: {decision.strip()}")
    
    # Step 2: ACT - Execute the tool
    if 'get_ticket' in decision.lower():
        result = spark.sql(f"SELECT module5a_demo5.agent_tools.get_ticket('{arg}')").collect()[0][0]
    elif 'count' in decision.lower():
        result = str(spark.sql(f"SELECT module5a_demo5.agent_tools.count_tickets_by_category('{arg}')").collect()[0][0])
    elif 'priority' in decision.lower():
        result = spark.sql(f"SELECT module5a_demo5.agent_tools.get_priority('{arg}')").collect()[0][0]
    else:
        result = "Unknown tool"
    print(f"  ACT   -> Tool result: {str(result)[:70]}")
    
    # Step 3: OBSERVE + RESPOND
    response = spark.sql(f"""
      SELECT ai_query(
        'databricks-meta-llama-3-3-70b-instruct',
        'You are a support agent. Tool result: {str(result)[:200]}. Respond to: "{question}"',
        modelParameters => named_struct('temperature', 0.0, 'max_tokens', 80)
      ) AS response
    """).collect()[0][0]
    print(f"  RESPOND -> {response.strip()}")
    print()

# COMMAND ----------

# DBTITLE 1,5.4 Notes - Function Tool Decorator
# MAGIC %md
# MAGIC ## 5.4 : @function_tool Decorator
# MAGIC
# MAGIC ### Concepts
# MAGIC The `@function_tool` decorator converts a regular Python function into an agent tool:
# MAGIC
# MAGIC * **What it does**: Takes a Python function with type hints and a docstring, and generates a tool schema that the LLM can understand.
# MAGIC * **Type hints** become parameter types in the tool schema.
# MAGIC * **Docstring** becomes the tool description for the LLM.
# MAGIC * **Return type** becomes the tool's output type.
# MAGIC
# MAGIC This is the fastest way to create custom tools for agents - just write a function, decorate it, and the agent can call it.
# MAGIC
# MAGIC > On Databricks, UC functions are the governed equivalent. @function_tool is for rapid prototyping; UC functions are for production.

# COMMAND ----------

# DBTITLE 1,5.4 Demo - Function Tool Pattern
# 5.4 Demo: @function_tool pattern - turning Python functions into agent tools
# The decorator converts type hints + docstring into a tool schema.
# We simulate the pattern here (the actual decorator requires the OpenAI Agents SDK).

import json

# Simulate the @function_tool decorator
def function_tool(func):
    """Simulate the @function_tool decorator that converts a function into a tool."""
    tool_schema = {
        "name": func.__name__,
        "description": func.__doc__.strip() if func.__doc__ else "",
        "parameters": {}
    }
    hints = func.__annotations__
    for param, hint in hints.items():
        if param != "return":
            tool_schema["parameters"][param] = hint.__name__ if hasattr(hint, '__name__') else str(hint)
    func.tool_schema = tool_schema
    return func

# Define tools using the decorator pattern
@function_tool
def get_ticket_status(ticket_id: str) -> str:
    """Returns the current status of a support ticket."""
    result = spark.sql(f"SELECT status FROM module5a_demo5.agent_tools.support_tickets WHERE ticket_id = '{ticket_id}'").collect()[0][0]
    return result

@function_tool
def count_category_tickets(category: str) -> int:
    """Returns the number of open tickets in a given category."""
    return spark.sql(f"SELECT module5a_demo5.agent_tools.count_tickets_by_category('{category}')").collect()[0][0]

@function_tool
def escalate_ticket(ticket_id: str, reason: str) -> str:
    """Escalates a ticket to a higher priority level."""
    return f"Ticket {ticket_id} escalated. Reason: {reason}"

# Show the auto-generated tool schemas
print("=== @function_tool: Auto-Generated Tool Schemas ===")
print()
for tool in [get_ticket_status, count_category_tickets, escalate_ticket]:
    schema = tool.tool_schema
    print(f"Tool: {schema['name']}")
    print(f"  Description: {schema['description']}")
    print(f"  Parameters: {schema['parameters']}")
    print()

# Call the tools (the agent would do this automatically)
print("=== Tool Execution ===")
print(f"  get_ticket_status('TKT-001') -> '{get_ticket_status('TKT-001')}'")
print(f"  count_category_tickets('billing') -> {count_category_tickets('billing')}")
print(f"  escalate_ticket('TKT-003', 'system outage') -> {escalate_ticket('TKT-003', 'system outage')}")

# COMMAND ----------

# DBTITLE 1,5.3 Notes - Single Agent Orchestration
# MAGIC %md
# MAGIC ## 5.3 : Single-Agent Tool Orchestration
# MAGIC
# MAGIC ### Concepts
# MAGIC Even a single agent needs to orchestrate its tools:
# MAGIC
# MAGIC 1. **Sequential calls**: Call tool A, see the result, then call tool B.
# MAGIC    - "Get ticket TKT-001" -> see it's billing -> "Count billing tickets"
# MAGIC
# MAGIC 2. **Parallel calls**: Call multiple tools at once when independent.
# MAGIC    - "Get ticket TKT-001" + "Get priority TKT-001" at the same time
# MAGIC
# MAGIC 3. **Conditional calls**: Skip tools based on earlier results.
# MAGIC    - If the ticket is closed, don't check priority.
# MAGIC
# MAGIC **LLM-driven vs. code-driven** (applies to single agents too):
# MAGIC * **LLM-driven**: The model decides which tool to call next based on context.
# MAGIC * **Code-driven**: The developer hard-codes the tool sequence.
# MAGIC * **Hybrid**: Code defines the overall flow; the LLM fills in details.
# MAGIC
# MAGIC > In a single agent, the LLM is both the reasoner and the orchestrator. In multi-agent, a supervisor handles orchestration.

# COMMAND ----------

# DBTITLE 1,5.3 Demo - Chained Tool Calls
# 5.3 Demo: Single-agent tool orchestration
# Show how a single agent chains tool calls to answer a complex question.

print("=== Single Agent: Chained Tool Calls ===")
print()
print("User: 'Tell me about TKT-001 and how it compares to other billing tickets'")
print()

# Step 1: Get the ticket
print("--- Call 1: get_ticket('TKT-001') ---")
ticket = spark.sql("SELECT module5a_demo5.agent_tools.get_ticket('TKT-001')").collect()[0][0]
print(f"  Result: {ticket}")
print()

# Step 2: Get priority (independent of step 1, could be parallel)
print("--- Call 2: get_priority('TKT-001') ---")
priority = spark.sql("SELECT module5a_demo5.agent_tools.get_priority('TKT-001')").collect()[0][0]
print(f"  Result: {priority}")
print()

# Step 3: Count billing tickets (depends on knowing TKT-001 is billing)
print("--- Call 3: count_tickets_by_category('billing') ---")
billing_count = spark.sql("SELECT module5a_demo5.agent_tools.count_tickets_by_category('billing')").collect()[0][0]
print(f"  Result: {billing_count} billing tickets")
print()

# Step 4: Synthesize
print("--- Agent Synthesizes Response ---")
synthesis = spark.sql(f"""
  SELECT ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    'You are a support agent. Synthesize a response using these facts: Ticket TKT-001 details: {ticket}. Priority: {priority}. Total billing tickets: {billing_count}. User asked: Tell me about TKT-001 and how it compares to other billing tickets.',
    modelParameters => named_struct('temperature', 0.0, 'max_tokens', 120)
  ) AS response
""").collect()[0][0]
print(f"  {synthesis.strip()}")
print()

print("=== Orchestration Summary ===")
print("The agent made 3 tool calls in sequence:")
print("  1. get_ticket -> learned the ticket is about billing")
print("  2. get_priority -> learned it's high priority")
print("  3. count_tickets_by_category -> learned there are 2 billing tickets")
print("Then synthesized a complete response from all 3 results.")

# COMMAND ----------

# DBTITLE 1,Learning Conclusion
# MAGIC %md
# MAGIC ## Learning Conclusion
# MAGIC
# MAGIC ### What we demonstrated
# MAGIC
# MAGIC | Topic | What was demoed | Key Takeaway |
# MAGIC |---|---|---|
# MAGIC | Anatomy | Reason-act-observe loop | An agent is a sequence of decisions, not a single LLM call |
# MAGIC | 5.1 | Single vs. multi-agent decision | Start with single agent; split when tools >8 or domains differ |
# MAGIC | Building | Full single agent with system prompt + UC tools | System prompt + UC functions + reasoning loop = a working agent |
# MAGIC | 5.4 | @function_tool decorator | Type hints + docstring -> auto-generated tool schema |
# MAGIC | 5.3 | Single-agent tool orchestration | Agent chains tool calls: get ticket -> get priority -> count -> respond |
# MAGIC
# MAGIC ### Key principles
# MAGIC * **An agent is a loop, not a call**: Reason -> Act -> Observe -> Respond, repeating as needed.
# MAGIC * **UC functions are governed tools**: Register in Unity Catalog for production agents.
# MAGIC * **@function_tool for prototyping**: Quick tool creation; migrate to UC functions for governance.
# MAGIC * **Single agents can handle complex tasks**: Sequential and conditional tool calls within one agent.

# COMMAND ----------

# DBTITLE 1,Cleanup - Decommission Demo Artifacts
# CLEANUP: Drop all resources created in this demo
print("Dropping UC functions...")
for fn in ['get_ticket', 'count_tickets_by_category', 'get_priority']:
    spark.sql(f"DROP FUNCTION IF EXISTS module5a_demo5.agent_tools.{fn}")
print("  Functions dropped")

print("Dropping schema and catalog...")
spark.sql("DROP SCHEMA IF EXISTS module5a_demo5.agent_tools CASCADE")
spark.sql("DROP CATALOG IF EXISTS module5a_demo5 CASCADE")
print("  Schema and catalog dropped")

print("\nCleanup complete!")