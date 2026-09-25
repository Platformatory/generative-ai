# Databricks notebook source
# DBTITLE 1,Demo Title
# MAGIC %md
# MAGIC # Demo 1 : Dataset-to-Genie Walkthrough
# MAGIC
# MAGIC **Module 5A - Databricks SQL + Genie (Topics 1.1-1.6)**
# MAGIC
# MAGIC This notebook walks through the Databricks AI/BI family - from designing dashboard datasets to using Genie for ad-hoc data exploration. We cover dashboard building blocks, dataset sources, Genie Code for AI-assisted authoring, and the decision framework for choosing dashboards vs. Genie.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Setup - Catalog + Sample Data
# MAGIC %sql
# MAGIC -- SETUP: Create catalog, schema, and sample data for dashboard + Genie demos
# MAGIC -- We create a retail sales dataset that's realistic for dashboards and Genie queries.
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS module5a_demo1;
# MAGIC CREATE SCHEMA IF NOT EXISTS module5a_demo1.sales;
# MAGIC
# MAGIC -- Customers table
# MAGIC CREATE OR REPLACE TABLE module5a_demo1.sales.customers (
# MAGIC   customer_id    STRING NOT NULL,
# MAGIC   customer_name  STRING NOT NULL,
# MAGIC   customer_tier  STRING,
# MAGIC   region         STRING,
# MAGIC   signup_date    DATE
# MAGIC );
# MAGIC
# MAGIC INSERT INTO module5a_demo1.sales.customers VALUES
# MAGIC ('CUST-001', 'Acme Corp',        'enterprise', 'North America', '2024-01-15'),
# MAGIC ('CUST-002', 'Globex Inc',       'premium',    'Europe',        '2024-03-20'),
# MAGIC ('CUST-003', 'Initech LLC',      'standard',   'North America', '2024-06-10'),
# MAGIC ('CUST-004', 'Umbrella SA',      'enterprise', 'Asia Pacific',  '2024-02-05'),
# MAGIC ('CUST-005', 'Stark Industries', 'premium',    'Europe',        '2024-08-18'),
# MAGIC ('CUST-006', 'Wayne Tech',       'standard',   'North America', '2024-04-22'),
# MAGIC ('CUST-007', 'Soylent Co',       'premium',    'Asia Pacific',  '2024-05-14'),
# MAGIC ('CUST-008', 'Cyberdyne Sys',    'enterprise', 'Europe',        '2024-07-01');
# MAGIC
# MAGIC -- Products table
# MAGIC CREATE OR REPLACE TABLE module5a_demo1.sales.products (
# MAGIC   product_id     STRING NOT NULL,
# MAGIC   product_name   STRING NOT NULL,
# MAGIC   category       STRING,
# MAGIC   unit_price     DECIMAL(10,2)
# MAGIC );
# MAGIC
# MAGIC INSERT INTO module5a_demo1.sales.products VALUES
# MAGIC ('PROD-001', 'Databricks Pro License',   'Software',  499.00),
# MAGIC ('PROD-002', 'Databricks Enterprise',    'Software', 1499.00),
# MAGIC ('PROD-003', 'Vector Search Add-on',    'Add-on',    299.00),
# MAGIC ('PROD-004', 'AI Gateway Starter',      'Add-on',    199.00),
# MAGIC ('PROD-005', 'Pro Support Package',     'Support',   999.00),
# MAGIC ('PROD-006', 'Premium Training Voucher', 'Training',  499.00);
# MAGIC
# MAGIC -- Orders table (fact table — the core of our dashboard + Genie queries)
# MAGIC CREATE OR REPLACE TABLE module5a_demo1.sales.orders (
# MAGIC   order_id       STRING NOT NULL,
# MAGIC   customer_id    STRING NOT NULL,
# MAGIC   product_id     STRING NOT NULL,
# MAGIC   order_date     DATE,
# MAGIC   quantity       INT,
# MAGIC   unit_price     DECIMAL(10,2),
# MAGIC   total_amount   DECIMAL(12,2),
# MAGIC   status         STRING,
# MAGIC   channel        STRING
# MAGIC );
# MAGIC
# MAGIC INSERT INTO module5a_demo1.sales.orders VALUES
# MAGIC ('ORD-001', 'CUST-001', 'PROD-002', '2025-01-15', 3, 1499.00, 4497.00, 'closed_won', 'direct'),
# MAGIC ('ORD-002', 'CUST-002', 'PROD-001', '2025-01-28', 5,  499.00, 2495.00, 'closed_won', 'partner'),
# MAGIC ('ORD-003', 'CUST-003', 'PROD-003', '2025-02-10', 2,  299.00,  598.00, 'closed_won', 'direct'),
# MAGIC ('ORD-004', 'CUST-004', 'PROD-002', '2025-02-22', 2, 1499.00, 2998.00, 'closed_won', 'direct'),
# MAGIC ('ORD-005', 'CUST-005', 'PROD-005', '2025-03-05', 1,  999.00,  999.00, 'closed_won', 'partner'),
# MAGIC ('ORD-006', 'CUST-001', 'PROD-001', '2025-04-12', 4,  499.00, 1996.00, 'closed_won', 'direct'),
# MAGIC ('ORD-007', 'CUST-006', 'PROD-004', '2025-04-25', 3,  199.00,  597.00, 'closed_won', 'direct'),
# MAGIC ('ORD-008', 'CUST-007', 'PROD-002', '2025-05-18', 2, 1499.00, 2998.00, 'closed_won', 'partner'),
# MAGIC ('ORD-009', 'CUST-008', 'PROD-003', '2025-06-08', 5,  299.00, 1495.00, 'closed_won', 'direct'),
# MAGIC ('ORD-010', 'CUST-002', 'PROD-006', '2025-06-20', 3,  499.00, 1497.00, 'closed_won', 'direct'),
# MAGIC ('ORD-011', 'CUST-003', 'PROD-001', '2025-07-15', 2,  499.00,  998.00, 'closed_won', 'direct'),
# MAGIC ('ORD-012', 'CUST-004', 'PROD-005', '2025-08-01', 1,  999.00,  999.00, 'closed_won', 'partner'),
# MAGIC ('ORD-013', 'CUST-005', 'PROD-002', '2025-08-22', 3, 1499.00, 4497.00, 'closed_won', 'direct'),
# MAGIC ('ORD-014', 'CUST-001', 'PROD-004', '2025-09-10', 4,  199.00,  796.00, 'closed_won', 'partner'),
# MAGIC ('ORD-015', 'CUST-006', 'PROD-002', '2025-09-25', 1, 1499.00, 1499.00, 'pending',    'direct');
# MAGIC
# MAGIC SELECT * FROM module5a_demo1.sales.orders ORDER BY order_date;

# COMMAND ----------

# DBTITLE 1,1.1 Notes - AI/BI Family
# MAGIC %md
# MAGIC ## 1.1 : The Databricks AI/BI Family
# MAGIC
# MAGIC ### Concepts
# MAGIC The Databricks AI/BI family provides four complementary ways to access and analyze data:
# MAGIC
# MAGIC | Component | What it is | Best for |
# MAGIC |---|---|---|
# MAGIC | **Dashboards** | Visual, structured reports on UC data | Known, repeated questions for many users |
# MAGIC | **Genie Agent** | NL interface to governed data | Ad-hoc, exploratory questions |
# MAGIC | **Genie Code** | AI-assisted authoring in notebooks/dashboards | Generating SQL, visualizations, full dashboards from prompts |
# MAGIC | **Genie One** | Unified search across dashboards + Genie | Finding the right answer regardless of source |
# MAGIC
# MAGIC **How they fit together**:
# MAGIC * **Dashboards** answer the questions you *know you have* (revenue by quarter, top customers).
# MAGIC * **Genie** answers the questions you *didn't know you'd have* ("which enterprise customers in Europe haven't ordered in Q3?").
# MAGIC * **Genie Code** helps you *build* dashboards and queries faster.
# MAGIC * **Genie One** ties it all together with a single search bar.
# MAGIC
# MAGIC > All components are governed by Unity Catalog — the same permissions, lineage, and audit logs apply everywhere.

# COMMAND ----------

# DBTITLE 1,1.1 Demo - Four Access Patterns
# MAGIC %sql
# MAGIC -- 1.1 Demo: The same data, accessed four ways
# MAGIC -- This single query shows the structured pattern that Dashboards use.
# MAGIC -- Genie would generate similar SQL from natural language.
# MAGIC -- Genie Code would write this SQL from a prompt like 'show monthly revenue'.
# MAGIC -- Genie One would find this result whether it lives in a dashboard or Genie.
# MAGIC
# MAGIC SELECT 
# MAGIC   date_format(order_date, 'yyyy-MM') AS month,
# MAGIC   count(*) AS order_count,
# MAGIC   sum(total_amount) AS revenue
# MAGIC FROM module5a_demo1.sales.orders
# MAGIC WHERE status = 'closed_won'
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1;

# COMMAND ----------

# DBTITLE 1,1.2 Notes - Dashboard Building Blocks
# MAGIC %md
# MAGIC ## 1.2 : AI/BI Dashboard Building Blocks
# MAGIC
# MAGIC ### Concepts
# MAGIC A Databricks AI/BI Dashboard is built from five components:
# MAGIC
# MAGIC 1. **Datasets**: SQL queries that define what data the dashboard shows. Each dataset is a SELECT statement against UC tables or views.
# MAGIC 2. **Canvas**: The layout surface where you place visualizations, text, and filters.
# MAGIC 3. **Visualizations**: Bar charts, line charts, tables, KPIs, counters — each bound to a dataset.
# MAGIC 4. **Filters**: Interactive controls (dropdowns, date ranges) that let users narrow the data.
# MAGIC 5. **Pages**: A dashboard can have multiple pages, each with its own canvas and visualizations.
# MAGIC
# MAGIC **Design principle**: Build one well-designed dataset per page, then create multiple visualizations from it. This avoids redundant queries and keeps the dashboard fast.
# MAGIC
# MAGIC > A good dataset pre-joins and pre-aggregates so visualizations don't recompute everything on each load.

# COMMAND ----------

# DBTITLE 1,1.2 Demo - Designing a Dashboard Dataset
# MAGIC %sql
# MAGIC -- 1.2 Demo: Designing a dataset for a dashboard
# MAGIC -- A well-designed dataset pre-joins orders + customers + products
# MAGIC -- so every visualization on the dashboard uses one clean source.
# MAGIC
# MAGIC CREATE OR REPLACE TABLE module5a_demo1.sales.dashboard_dataset AS
# MAGIC SELECT 
# MAGIC   o.order_id,
# MAGIC   o.order_date,
# MAGIC   date_format(o.order_date, 'yyyy-MM') AS order_month,
# MAGIC   concat(year(o.order_date), '-Q', quarter(o.order_date)) AS order_quarter,
# MAGIC   o.customer_id,
# MAGIC   c.customer_name,
# MAGIC   c.customer_tier,
# MAGIC   c.region,
# MAGIC   o.product_id,
# MAGIC   p.product_name,
# MAGIC   p.category AS product_category,
# MAGIC   o.quantity,
# MAGIC   o.unit_price,
# MAGIC   o.total_amount,
# MAGIC   o.status,
# MAGIC   o.channel
# MAGIC FROM module5a_demo1.sales.orders o
# MAGIC JOIN module5a_demo1.sales.customers c ON o.customer_id = c.customer_id
# MAGIC JOIN module5a_demo1.sales.products p ON o.product_id = p.product_id;
# MAGIC
# MAGIC -- This single dataset powers all dashboard visualizations:
# MAGIC -- KPIs, trends, breakdowns by region/category/tier
# MAGIC SELECT * FROM module5a_demo1.sales.dashboard_dataset ORDER BY order_date;

# COMMAND ----------

# DBTITLE 1,1.3 Notes - Dashboard Dataset Sources
# MAGIC %md
# MAGIC ## 1.3 : Dashboard Dataset Sources
# MAGIC
# MAGIC ### Concepts
# MAGIC A dashboard dataset can pull from three sources:
# MAGIC
# MAGIC | Source | What it is | When to use |
# MAGIC |---|---|---|
# MAGIC | **UC table** | Any Unity Catalog table or view queried directly | Simple datasets, single-table queries |
# MAGIC | **Reusable aggregation** | A pre-computed aggregation stored as a UC table | Shared KPIs used across multiple dashboards |
# MAGIC | **Dashboard-local view** | A query defined within the dashboard SQL editor | One-off calculations specific to one dashboard |
# MAGIC
# MAGIC **Key distinction**:
# MAGIC * UC tables and reusable aggregations are **governed objects** — they have their own permissions, lineage, and audit trail.
# MAGIC * Dashboard-local views are **embedded in the dashboard** — they don't exist as separate UC objects.
# MAGIC
# MAGIC > Prefer reusable UC aggregations for KPIs that multiple dashboards need. Use dashboard-local views for one-off calculations.

# COMMAND ----------

# DBTITLE 1,1.3 Demo - Dataset Sources
# MAGIC %sql
# MAGIC -- 1.3 Demo: Three dashboard dataset sources in action
# MAGIC
# MAGIC -- Source 1: Direct UC table query (simplest)
# MAGIC SELECT 
# MAGIC   order_quarter,
# MAGIC   sum(total_amount) AS revenue,
# MAGIC   count(*) AS orders
# MAGIC FROM module5a_demo1.sales.dashboard_dataset
# MAGIC WHERE status = 'closed_won'
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1;
# MAGIC
# MAGIC -- Source 2: Reusable UC aggregation (pre-computed KPIs for multiple dashboards)
# MAGIC CREATE OR REPLACE TABLE module5a_demo1.sales.quarterly_revenue AS
# MAGIC SELECT 
# MAGIC   order_quarter,
# MAGIC   region,
# MAGIC   customer_tier,
# MAGIC   sum(total_amount) AS total_revenue,
# MAGIC   count(*) AS order_count,
# MAGIC   avg(total_amount) AS avg_order_value
# MAGIC FROM module5a_demo1.sales.dashboard_dataset
# MAGIC WHERE status = 'closed_won'
# MAGIC GROUP BY 1, 2, 3;
# MAGIC
# MAGIC SELECT * FROM module5a_demo1.sales.quarterly_revenue ORDER BY order_quarter;
# MAGIC
# MAGIC -- Source 3: Dashboard-local view (inline in the dashboard SQL editor)
# MAGIC -- This query would be defined directly in the dashboard, not as a UC object
# MAGIC SELECT 
# MAGIC   order_quarter,
# MAGIC   product_category,
# MAGIC   sum(total_amount) AS revenue
# MAGIC FROM module5a_demo1.sales.dashboard_dataset
# MAGIC WHERE status = 'closed_won'
# MAGIC GROUP BY 1, 2
# MAGIC ORDER BY 1, 2;

# COMMAND ----------

# DBTITLE 1,1.4 Notes - Genie Code
# MAGIC %md
# MAGIC ## 1.4 : Genie Code
# MAGIC
# MAGIC ### Concepts
# MAGIC Genie Code is Databricks' AI-assisted authoring tool. It lives inside notebooks and dashboard editors and generates:
# MAGIC
# MAGIC * **SQL from natural language**: Type "show revenue by quarter for each region" and get a ready-to-run SELECT.
# MAGIC * **Auto-configured visualizations**: Genie Code picks the chart type, axes, and grouping based on the data.
# MAGIC * **Full dashboard planning**: Describe what you want ("build a sales performance dashboard") and Genie Code plans pages, datasets, and widgets.
# MAGIC * **Inline quick-fixes**: Ask Genie Code to fix or modify existing SQL ("filter to closed_won only", "add a YoY comparison").
# MAGIC
# MAGIC **How it differs from Genie Agent**:
# MAGIC * Genie Code helps you *build* artifacts (SQL, dashboards). The output is code you can edit and save.
# MAGIC * Genie Agent *answers questions* at runtime. The output is a natural language answer backed by auto-generated SQL.
# MAGIC
# MAGIC > Genie Code is available in the notebook editor and the AI/BI Dashboard editor. Look for the AI button in the toolbar.

# COMMAND ----------

# DBTITLE 1,1.4 Demo - Genie Code Simulation
# 1.4 Demo: Genie Code — AI-assisted dashboard authoring
# Genie Code generates SQL, datasets, and visualizations from natural language.
# Here we simulate what Genie Code would produce from a prompt.

print("=== Genie Code: Natural Language to SQL + Visualization ===")
print()
print("User prompt: 'Show me revenue by quarter for each region'")
print()
print("Genie Code generates:")
print("  1. A SQL dataset query")
print("  2. Auto-detected visualization type (grouped bar chart)")
print("  3. Suggested filters (region, quarter)")
print()
print("--- Generated SQL ---")
print("""
SELECT
  order_quarter,
  region,
  SUM(total_amount) AS revenue
FROM module5a_demo1.sales.dashboard_dataset
WHERE status = 'closed_won'
GROUP BY order_quarter, region
ORDER BY order_quarter, region
""")
print("--- Auto-configured visualization ---")
print("  Type: Grouped bar chart")
print("  X-axis: order_quarter")
print("  Y-axis: revenue")
print("  Group by: region")
print()

print("=== Genie Code: Full Dashboard Planning ===")
print("User prompt: 'Build a sales performance dashboard'")
print()
print("Genie Code plans the full dashboard:")
print("  Page 1: KPIs (total revenue, order count, avg order value)")
print("  Page 2: Trends (revenue by quarter, orders by month)")
print("  Page 3: Breakdown (revenue by region, by product category)")
print("  Each page: auto-generated dataset + suggested visualizations")
print()

print("=== Genie Code: Inline Quick-Fix ===")
print("User prompt: 'Add a year-over-year comparison to the revenue chart'")
print()
print("Genie Code modifies the existing query:")
print("  - Adds a previous-year revenue column using LAG()")
print("  - Updates the visualization to show both current and prior year")
print("  - Suggests a line chart with two series")

# COMMAND ----------

# DBTITLE 1,1.5 Notes - Dashboard vs Genie
# MAGIC %md
# MAGIC ## 1.5 : Dashboard vs. Genie Agent
# MAGIC
# MAGIC ### Concepts
# MAGIC Dashboards and Genie Agent solve different problems with the same data:
# MAGIC
# MAGIC | Aspect | Dashboard | Genie Agent |
# MAGIC |---|---|---|
# MAGIC | **Question type** | Known, repeated | Unknown, ad-hoc |
# MAGIC | **Output** | Visual (charts, KPIs, tables) | Natural language + table results |
# MAGIC | **Setup** | Build datasets + visualizations once | Point at tables, add instructions |
# MAGIC | **User** | Many users, broad audience | Individual exploring data |
# MAGIC | **Speed** | Instant (pre-built) | A few seconds (generates SQL on the fly) |
# MAGIC | **Flexibility** | Low (fixed visualizations) | High (any question within scope) |
# MAGIC | **Governance** | UC permissions on underlying tables | UC permissions + Genie-specific instructions |
# MAGIC
# MAGIC **Decision framework**:
# MAGIC * **Use a Dashboard** when: the question is asked repeatedly, many users need the answer, and a visual format is best.
# MAGIC * **Use Genie** when: the question is new or exploratory, one person needs the answer right now, and flexibility matters more than polish.
# MAGIC * **Use both**: dashboards for the 80% known questions, Genie for the 20% ad-hoc questions on the same data.

# COMMAND ----------

# DBTITLE 1,1.5 Demo - Dashboard vs Genie
# MAGIC %sql
# MAGIC -- 1.5 Demo: Dashboard vs. Genie Agent — same data, different access pattern
# MAGIC
# MAGIC -- DASHBOARD question (known, repeated): "Revenue by quarter"
# MAGIC -- Pre-built, visual, shared with many users
# MAGIC SELECT 
# MAGIC   order_quarter,
# MAGIC   sum(total_amount) AS revenue,
# MAGIC   count(*) AS orders
# MAGIC FROM module5a_demo1.sales.dashboard_dataset
# MAGIC WHERE status = 'closed_won'
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1;
# MAGIC
# MAGIC -- GENIE question (ad-hoc, one-time): 
# MAGIC -- "Which enterprise customers in Europe haven't placed an order in Q3?"
# MAGIC -- Genie generates this SQL from natural language
# MAGIC SELECT c.customer_name, c.region, max(o.order_date) AS last_order
# MAGIC FROM module5a_demo1.sales.customers c
# MAGIC LEFT JOIN module5a_demo1.sales.orders o 
# MAGIC   ON c.customer_id = o.customer_id AND o.status = 'closed_won'
# MAGIC WHERE c.customer_tier = 'enterprise' AND c.region = 'Europe'
# MAGIC GROUP BY c.customer_name, c.region
# MAGIC HAVING max(o.order_date) IS NULL 
# MAGIC     OR max(o.order_date) < '2025-07-01';

# COMMAND ----------

# DBTITLE 1,1.6 Notes - Well-Scoped Dashboard
# MAGIC %md
# MAGIC ## 1.6 : Three Ingredients of a Well-Scoped Dashboard
# MAGIC
# MAGIC ### Concepts
# MAGIC Every successful dashboard has three ingredients:
# MAGIC
# MAGIC 1. **Purpose**: What decision does this dashboard support?
# MAGIC    * A dashboard without a purpose is just a data dump.
# MAGIC    * Example: "Help regional managers track quarterly sales and identify at-risk regions."
# MAGIC
# MAGIC 2. **Audience**: Who consumes it and what do they need?
# MAGIC    * Executives need high-level KPIs. Analysts need drill-down capability.
# MAGIC    * Match the level of detail and interactivity to the audience.
# MAGIC
# MAGIC 3. **Data**: What data is needed, and is it clean and governed?
# MAGIC    * Pre-join and pre-aggregate in the dataset.
# MAGIC    * Use UC-governed tables for permissions and lineage.
# MAGIC    * Filter to the relevant subset (e.g., closed_won only for revenue).
# MAGIC
# MAGIC ### Do's and Don'ts
# MAGIC
# MAGIC | Do | Don't |
# MAGIC |---|---|
# MAGIC | One clear purpose per dashboard | 20+ widgets covering unrelated topics |
# MAGIC | Pre-aggregate in the dataset | Raw transaction-level data |
# MAGIC | Filter to relevant data (closed_won) | Show all statuses including pending/lost |
# MAGIC | 5-10 visualizations per page | 30+ charts on one page |
# MAGIC | Use UC-governed tables | Ad-hoc CSV uploads with no governance |

# COMMAND ----------

# DBTITLE 1,1.6 Demo - Well-Scoped Dashboard
# MAGIC %sql
# MAGIC -- 1.6 Demo: Three ingredients applied to our sales dashboard
# MAGIC -- PURPOSE: Track quarterly revenue performance and identify at-risk regions
# MAGIC -- AUDIENCE: Regional sales managers (need numbers, not raw data)
# MAGIC -- DATA: Closed-won orders, by quarter and region, with customer tier
# MAGIC
# MAGIC SELECT 
# MAGIC   order_quarter,
# MAGIC   region,
# MAGIC   customer_tier,
# MAGIC   sum(total_amount) AS revenue,
# MAGIC   count(*) AS order_count
# MAGIC FROM module5a_demo1.sales.dashboard_dataset
# MAGIC WHERE status = 'closed_won'
# MAGIC GROUP BY 1, 2, 3
# MAGIC ORDER BY 1, 2, 3;
# MAGIC
# MAGIC -- Best practices checklist:
# MAGIC -- ✓ One clear purpose: sales performance tracking
# MAGIC -- ✓ Audience-appropriate: pre-aggregated, no SQL needed
# MAGIC -- ✓ Governed data: UC table, filtered to closed_won
# MAGIC -- ✓ 5-10 visualizations: one dataset powers multiple charts
# MAGIC -- ✓ Scoped: covers sales only, not marketing or operations

# COMMAND ----------

# DBTITLE 1,Genie Space Prompt
# MAGIC %md
# MAGIC ## Genie Space Prompt
# MAGIC
# MAGIC ### Instructions for the Genie Space
# MAGIC
# MAGIC Copy this prompt into the Genie Space instructions when setting up a Genie space for the `module5a_demo1.sales` schema.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Genie Space Instructions:**
# MAGIC
# MAGIC You are a sales analytics assistant for a B2B software company. You help users answer questions about sales orders, customers, and products.
# MAGIC
# MAGIC **Available tables:**
# MAGIC - `module5a_demo1.sales.orders` — Sales orders with order_date, quantity, unit_price, total_amount, status, channel
# MAGIC - `module5a_demo1.sales.customers` — Customer info with customer_name, customer_tier (enterprise/premium/standard), region
# MAGIC - `module5a_demo1.sales.products` — Product catalog with product_name, category, unit_price
# MAGIC - `module5a_demo1.sales.dashboard_dataset` — Pre-joined view of orders + customers + products (use this for most queries)
# MAGIC
# MAGIC **Key definitions:**
# MAGIC - "Revenue" = SUM(total_amount) where status = 'closed_won'
# MAGIC - "Order count" = COUNT(*) where status = 'closed_won'
# MAGIC - "Average order value" = AVG(total_amount) where status = 'closed_won'
# MAGIC - Quarters are derived from order_date using concat(year(order_date), '-Q', quarter(order_date))
# MAGIC - Regions: North America, Europe, Asia Pacific
# MAGIC
# MAGIC **Example questions:**
# MAGIC 1. "What was our total revenue in Q1?"
# MAGIC 2. "Show revenue by region for each quarter"
# MAGIC 3. "Which product category generates the most revenue?"
# MAGIC 4. "Who are our top 5 customers by total revenue?"
# MAGIC 5. "What's the average order value for enterprise customers?"
# MAGIC
# MAGIC **Guidelines:**
# MAGIC - Always filter to status = 'closed_won' for revenue calculations
# MAGIC - Use the dashboard_dataset table for most queries (it has all joins pre-built)
# MAGIC - Format dates as yyyy-MM for months or use concat(year(date), '-Q', quarter(date)) for quarters
# MAGIC - Round currency to 2 decimal places
# MAGIC - Limit result sets to 100 rows unless the user asks for more

# COMMAND ----------

# DBTITLE 1,Learning Conclusion
# MAGIC %md
# MAGIC ## Learning Conclusion
# MAGIC
# MAGIC ### What we demonstrated
# MAGIC
# MAGIC | Topic | What was demoed | Key Takeaway |
# MAGIC |---|---|---|
# MAGIC | 1.1 | AI/BI family overview | Same data, four access patterns: Dashboard, Genie, Genie Code, Genie One |
# MAGIC | 1.2 | Designing a dashboard dataset | Pre-join and pre-aggregate so the dashboard doesn't recompute on load |
# MAGIC | 1.3 | Dashboard dataset sources | UC tables (direct), reusable aggregations (shared KPIs), dashboard-local views (one-off) |
# MAGIC | 1.4 | Genie Code | AI generates SQL, picks visualizations, plans full dashboards from NL prompts |
# MAGIC | 1.5 | Dashboard vs. Genie Agent | Dashboard = known/repeated questions; Genie = unknown/ad-hoc questions |
# MAGIC | 1.6 | Well-scoped dashboard | Three ingredients: purpose, audience, data |
# MAGIC
# MAGIC ### Key principles
# MAGIC * **Dashboards are for known questions**: Build them when the question is repeated and the answer should be visual.
# MAGIC * **Genie is for ad-hoc questions**: Use it when the question is new, exploratory, or one-time.
# MAGIC * **Genie Code accelerates authoring**: It generates SQL, visualizations, and full dashboard plans from natural language.
# MAGIC * **Dataset design matters**: Pre-join and pre-aggregate so dashboards load fast.
# MAGIC * **Scope your dashboard**: One purpose, one audience, clean governed data.

# COMMAND ----------

# DBTITLE 1,Cleanup - Decommission Demo Artifacts
# MAGIC %sql
# MAGIC -- CLEANUP: Decommission everything created in this demo
# MAGIC DROP TABLE IF EXISTS module5a_demo1.sales.dashboard_dataset;
# MAGIC DROP TABLE IF EXISTS module5a_demo1.sales.quarterly_revenue;
# MAGIC DROP TABLE IF EXISTS module5a_demo1.sales.orders;
# MAGIC DROP TABLE IF EXISTS module5a_demo1.sales.customers;
# MAGIC DROP TABLE IF EXISTS module5a_demo1.sales.products;
# MAGIC DROP SCHEMA IF EXISTS module5a_demo1.sales CASCADE;
# MAGIC DROP CATALOG IF EXISTS module5a_demo1 CASCADE;
# MAGIC
# MAGIC SHOW CATALOGS LIKE 'module5a_demo1';