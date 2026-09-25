# Databricks notebook source
# DBTITLE 1,Demo 5 Title
# MAGIC %md
# MAGIC # Demo 6: Multi-Agent Orchestration
# MAGIC
# MAGIC This demo covers the supervisor-worker architecture, LLM-driven vs. code-driven orchestration, Agent Bricks, Genie integration, and code-first vs. no-code decision frameworks.
# MAGIC
# MAGIC > **Prerequisite**: Complete Demo 5 (Building Agents) first to understand agent anatomy, the reason-act-observe loop, and @function_tool before moving to multi-agent systems.

# COMMAND ----------

# DBTITLE 1,Setup - Catalog, Schema, Sample Data
# MAGIC %sql
# MAGIC -- SETUP: Create catalog, schema, sample data, and UC functions
# MAGIC -- We create a dedicated catalog/schema for this demo with a
# MAGIC -- support_tickets table for multi-agent routing scenarios.
# MAGIC -- UC functions registered here serve as specialized agent tools.
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS module5a_demo6;
# MAGIC CREATE SCHEMA IF NOT EXISTS module5a_demo6.agent_systems;
# MAGIC
# MAGIC -- Sample support tickets table for multi-agent routing demos
# MAGIC CREATE OR REPLACE TABLE module5a_demo6.agent_systems.support_tickets (
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
# MAGIC INSERT INTO module5a_demo6.agent_systems.support_tickets VALUES
# MAGIC ('TKT-001', 'CUST-001', 'billing',   'high',   'Overcharged on order ORD-001', 'I was charged $249.99 but the product was on sale for $199.99', 'open', '2026-09-25 08:00:00'),
# MAGIC ('TKT-002', 'CUST-002', 'shipping', 'medium', 'Order ORD-003 not delivered',   'My order from Sept 20 has not arrived yet',                     'open', '2026-09-25 09:30:00'),
# MAGIC ('TKT-003', 'CUST-003', 'technical','high',   'Cannot access my account',     'I keep getting a 403 error when trying to log in',             'open', '2026-09-25 10:15:00'),
# MAGIC ('TKT-004', 'CUST-001', 'general',   'low',    'Product recommendation',      'Can you recommend accessories for my recent purchase?',        'open', '2026-09-25 11:00:00'),
# MAGIC ('TKT-005', 'CUST-002', 'billing',   'medium', 'Discount not applied',        'My premium discount was not applied to order ORD-008',         'open', '2026-09-25 11:45:00');
# MAGIC
# MAGIC -- UC functions as specialized agent tools
# MAGIC CREATE OR REPLACE FUNCTION module5a_demo6.agent_systems.get_ticket(ticket_id STRING)
# MAGIC RETURNS STRING
# MAGIC COMMENT 'Returns the subject and description of a support ticket'
# MAGIC RETURN (SELECT max(concat_ws(' | ', subject, description))
# MAGIC        FROM module5a_demo6.agent_systems.support_tickets
# MAGIC        WHERE ticket_id = get_ticket.ticket_id);
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION module5a_demo6.agent_systems.count_tickets_by_category(cat STRING)
# MAGIC RETURNS INT
# MAGIC COMMENT 'Returns the number of tickets in a given category'
# MAGIC RETURN SELECT count(*) FROM module5a_demo6.agent_systems.support_tickets
# MAGIC        WHERE category = count_tickets_by_category.cat;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION module5a_demo6.agent_systems.get_priority(ticket_id STRING)
# MAGIC RETURNS STRING
# MAGIC COMMENT 'Returns the priority of a support ticket'
# MAGIC RETURN (SELECT max(priority) FROM module5a_demo6.agent_systems.support_tickets
# MAGIC        WHERE ticket_id = get_priority.ticket_id);
# MAGIC
# MAGIC SELECT * FROM module5a_demo6.agent_systems.support_tickets ORDER BY ticket_id;

# COMMAND ----------

# DBTITLE 1,5.2 Notes - Supervisor-Worker Pattern
# MAGIC %md
# MAGIC ## 5.2 : Supervisor-Worker (Router-Delegate) Pattern
# MAGIC
# MAGIC ### Concepts
# MAGIC The supervisor-worker pattern is the most common multi-agent architecture:
# MAGIC
# MAGIC * **Supervisor**: Receives the user's question, decides which specialist to delegate to, and synthesizes the final response.
# MAGIC * **Workers**: Specialized agents, each with its own system prompt, tools, and domain expertise.
# MAGIC * **Router**: The supervisor's routing decision can be LLM-driven (the model decides) or rule-based (deterministic).
# MAGIC
# MAGIC **Flow**: User -> Supervisor (routes) -> Worker (executes) -> Supervisor (synthesizes) -> User
# MAGIC
# MAGIC **Common worker specializations**:
# MAGIC * Data Agent: Queries tables, runs SQL, returns structured data
# MAGIC * Research Agent: Searches knowledge base, retrieves documents
# MAGIC * General Agent: Handles FAQs, small talk, general questions

# COMMAND ----------

# DBTITLE 1,5.2 Demo - Supervisor Routing
# 5.2 Demo: Supervisor routes tickets to specialized agents
# The supervisor uses ai_query to classify the ticket,
# then delegates to the appropriate specialist.

print("=== Supervisor-Worker Pattern: Ticket Routing ===")
print()

tickets = spark.sql("SELECT ticket_id, subject, category FROM module5a_demo6.agent_systems.support_tickets ORDER BY ticket_id").collect()

for t in tickets:
    ticket_id = t['ticket_id']
    subject = t['subject']
    actual_category = t['category']
    
    # Supervisor: LLM classifies the ticket
    routing = spark.sql(f"""
      SELECT ai_query(
        'databricks-meta-llama-3-3-70b-instruct',
        'Classify this support ticket into exactly one category (billing, shipping, technical, general). Ticket subject: "{subject}". Respond with just the category name.',
        modelParameters => named_struct('temperature', 0.0, 'max_tokens', 20)
      ) AS category
    """).collect()[0][0]
    
    routed = routing.strip().lower()
    
    # Worker: execute the appropriate specialist's tool
    priority = spark.sql(f"SELECT module5a_demo6.agent_systems.get_priority('{ticket_id}')").collect()[0][0]
    ticket_info = spark.sql(f"SELECT module5a_demo6.agent_systems.get_ticket('{ticket_id}')").collect()[0][0]
    
    print(f"Ticket {ticket_id}: '{subject}'")
    print(f"  Supervisor routed to: {routed} agent (actual: {actual_category})")
    print(f"  Priority: {priority}")
    print(f"  Ticket info: {ticket_info[:80]}...")
    print()

# COMMAND ----------

# DBTITLE 1,5.5 Notes - Agent Bricks
# MAGIC %md
# MAGIC ## 5.5 : Agent Bricks
# MAGIC
# MAGIC ### Concepts
# MAGIC Agent Bricks is Databricks' no-code platform for building, optimizing, and governing production AI agents:
# MAGIC
# MAGIC * **What it is**: A managed platform that handles the infrastructure, evaluation, and deployment of AI agents.
# MAGIC * **Supported types**:
# MAGIC   * **Knowledge Assistant**: RAG-based Q&A over your documents (Instructed Retriever)
# MAGIC   * **Supervisor Agent**: Multi-agent orchestrator that routes to specialized subagents
# MAGIC   * **Document Intelligence**: Extract, classify, and process documents at scale
# MAGIC   * **Custom Agents**: Build your own agent with custom tools and logic
# MAGIC
# MAGIC * **Key differentiator**: No code required for setup. You specify the agent's purpose, configure it on your data, and the platform handles the rest.
# MAGIC
# MAGIC > Agent Bricks is built on Unity Catalog, Model Serving, and Agent Evaluation - all the governance and observability you need for production.

# COMMAND ----------

# DBTITLE 1,5.6 Notes - Agent Bricks Lifecycle
# MAGIC %md
# MAGIC ## 5.6 : Agent Bricks Development Lifecycle
# MAGIC
# MAGIC ### Concepts
# MAGIC The Agent Bricks lifecycle follows a specify -> configure -> improve loop:
# MAGIC
# MAGIC 1. **Specify**: Define the agent's purpose, knowledge sources, and tools
# MAGIC    - Upload documents or point to UC volumes/tables
# MAGIC    - Define what the agent should and should not do
# MAGIC
# MAGIC 2. **Configure & Evaluate**: Test the agent on your actual data
# MAGIC    - The platform automatically evaluates quality using built-in judges
# MAGIC    - Review accuracy, groundedness, and safety scores
# MAGIC    - Iterate on instructions and configuration
# MAGIC
# MAGIC 3. **Deploy**: Publish the agent as a serving endpoint
# MAGIC    - Automatically versioned and governed by Unity Catalog
# MAGIC    - Monitor with inference tables
# MAGIC
# MAGIC 4. **Continuously Improve**: Use evaluation results to improve
# MAGIC    - Add examples to improve quality
# MAGIC    - Update knowledge sources
# MAGIC    - Re-evaluate and redeploy
# MAGIC
# MAGIC > The entire lifecycle is governed: UC permissions, audit logs, and lineage tracking apply at every step.

# COMMAND ----------

# DBTITLE 1,5.7 Notes - Genie vs Genie Agent
# MAGIC %md
# MAGIC ## 5.7 : Genie Space vs. Genie Agent
# MAGIC
# MAGIC ### Concepts
# MAGIC Both provide natural-language access to data, but serve different use cases:
# MAGIC
# MAGIC | Aspect | Genie Space | Genie Agent |
# MAGIC |---|---|---|
# MAGIC | **Purpose** | Curated NL interface to governed data | AI agent that can use tools and reason |
# MAGIC | **Scope** | Answers questions about specific tables | Can call tools, search, and take actions |
# MAGIC | **Setup** | Point to tables, write instructions | Configure agent with tools and prompts |
# MAGIC | **Use case** | "What were Q3 sales by region?" | "Analyze Q3 sales, identify anomalies, and draft a report" |
# MAGIC | **Autonomy** | Low - answers data questions | High - can chain tools and make decisions |
# MAGIC
# MAGIC **When to use Genie Space**: When users need self-serve data exploration with NL.
# MAGIC **When to use Genie Agent**: When the task requires multi-step reasoning and tool use.
# MAGIC
# MAGIC > Genie Spaces can be used standalone or as a specialized worker in a multi-agent system.

# COMMAND ----------

# DBTITLE 1,5.8 Notes - Integrating Genie with Agents
# MAGIC %md
# MAGIC ## 5.8 : Integrating Genie with Agents
# MAGIC
# MAGIC ### Concepts
# MAGIC Genie can serve as a specialized data worker in a multi-agent system:
# MAGIC
# MAGIC * **Pattern 1: Genie as a subagent**: A supervisor agent routes data questions to a Genie Space. The Genie Space answers using its curated tables, and the supervisor includes the answer in its response.
# MAGIC * **Pattern 2: Genie for ad-hoc analysis**: When the user asks an unexpected data question, the supervisor delegates to Genie instead of failing.
# MAGIC * **Pattern 3: Standalone Genie**: No agent needed - users interact with Genie directly for data questions.
# MAGIC
# MAGIC **Benefits of integration**:
# MAGIC * The agent handles complex reasoning and tool use
# MAGIC * Genie handles data queries with its optimized NL-to-SQL pipeline
# MAGIC * Governance is unified through Unity Catalog
# MAGIC
# MAGIC > Integration pattern: Supervisor -> Genie Space (data questions) + Research Agent (knowledge) + General Agent (FAQs) -> Supervisor synthesizes response.

# COMMAND ----------

# DBTITLE 1,5.9 Notes - Code-first vs No-code
# MAGIC %md
# MAGIC ## 5.9 : Code-first vs. No-code Agent Platforms
# MAGIC
# MAGIC ### Concepts
# MAGIC Two approaches to building agents on Databricks:
# MAGIC
# MAGIC | Approach | Tools | Best for |
# MAGIC |---|---|---|
# MAGIC | **Code-first** | OpenAI Agents SDK, LangChain, LangGraph, DSPy | Custom logic, complex tool chains, research, prototyping |
# MAGIC | **No-code** | Agent Bricks, Genie | Production agents, governed deployments, rapid delivery |
# MAGIC
# MAGIC **Code-first advantages**:
# MAGIC * Full control over agent logic and flow
# MAGIC * Can integrate any library or API
# MAGIC * Ideal for research and experimentation
# MAGIC
# MAGIC **No-code advantages**:
# MAGIC * Faster time to production
# MAGIC * Built-in evaluation and monitoring
# MAGIC * Automatic governance and compliance
# MAGIC * No infrastructure to manage
# MAGIC
# MAGIC > Start with code-first to prototype and validate. Move to no-code (Agent Bricks) for production deployment.

# COMMAND ----------

# DBTITLE 1,Learning Conclusion
# MAGIC %md
# MAGIC ## Learning Conclusion
# MAGIC
# MAGIC ### What we demonstrated
# MAGIC
# MAGIC | Topic | What was demoed | Key Takeaway |
# MAGIC |---|---|---|
# MAGIC | 5.2 | Supervisor-worker ticket routing | Supervisor classifies -> delegates to specialist -> synthesizes response |
# MAGIC | 5.5 | Agent Bricks concepts | No-code platform: Knowledge Assistant, Supervisor Agent, Document Intelligence |
# MAGIC | 5.6 | Agent Bricks lifecycle | Specify -> Configure/Evaluate -> Deploy -> Continuously Improve |
# MAGIC | 5.7 | Genie Space vs. Genie Agent | Space: NL data queries; Agent: multi-step reasoning with tools |
# MAGIC | 5.8 | Integrating Genie with agents | Genie as a specialized data worker in a multi-agent system |
# MAGIC | 5.9 | Code-first vs. no-code | Code-first for prototyping; Agent Bricks for production |
# MAGIC
# MAGIC ### Key principles
# MAGIC * **Supervisor-worker is the default pattern**: Route, delegate, synthesize.
# MAGIC * **Agent Bricks for production**: No-code, governed, evaluated, monitored.

# COMMAND ----------

# DBTITLE 1,Cleanup - Decommission Demo Artifacts
# CLEANUP: Drop all resources created in this demo
print("Dropping UC functions...")
for fn in ['get_ticket', 'count_tickets_by_category', 'get_priority']:
    spark.sql(f"DROP FUNCTION IF EXISTS module5a_demo6.agent_systems.{fn}")
print("  Functions dropped")

print("Dropping schema and catalog...")
spark.sql("DROP SCHEMA IF EXISTS module5a_demo6.agent_systems CASCADE")
spark.sql("DROP CATALOG IF EXISTS module5a_demo6 CASCADE")
print("  Schema and catalog dropped")

print("\nCleanup complete!")