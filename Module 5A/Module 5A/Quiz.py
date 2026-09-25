# Databricks notebook source
# DBTITLE 1,Quiz Title
# MAGIC %md
# MAGIC # Module 5A Quiz
# MAGIC
# MAGIC **Practice questions filtered to actual demo content in Module 5A**
# MAGIC
# MAGIC Each question maps to a topic **actually demonstrated** in one of the Module 5A demos (Demo 1-7). Questions about deployment infrastructure, MLflow, MCP, and other modules' topics have been excluded.
# MAGIC
# MAGIC **Format**: Questions are in markdown cells. **Answers are in expandable Python cells** below each question group — run the cell and click each answer summary to expand/collapse.
# MAGIC
# MAGIC **Total**: 32 questions covering Hours 2-6 topics that are actually demoed.

# COMMAND ----------

# DBTITLE 1,Hour 2 - Questions
# MAGIC %md
# MAGIC ## Hour 2: Gen AI Fundamentals & AI Functions (Demo 2)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q1.
# MAGIC
# MAGIC A Generative AI Engineer would like to build an application that can update a memo field that is about a paragraph long to just a single sentence gist that shows intent of the memo field, but fits into their application front end.
# MAGIC
# MAGIC With which Natural Language Processing task category should they evaluate potential LLMs for this application?
# MAGIC
# MAGIC **A.** text2text Generation  
# MAGIC **B.** Sentencizer  
# MAGIC **C.** Text Classification  
# MAGIC **D.** Summarization
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q2.
# MAGIC
# MAGIC A Generative AI Engineer is building an LLM to generate article summaries in the form of a type of poem, such as a haiku, given the article content. However, the initial output from the LLM does not match the desired tone or style. Which approach will NOT improve the LLM's response?
# MAGIC
# MAGIC **A.** Provide the LLM with a prompt that explicitly instructs it to generate text in the desired tone and style  
# MAGIC **B.** Use a neutralizer to normalize the tone and style of the underlying documents  
# MAGIC **C.** Include few-shot examples in the prompt to the LLM  
# MAGIC **D.** Fine-tune the LLM on a dataset of desired tone and style

# COMMAND ----------

# DBTITLE 1,Hour 2 - Answers
from IPython.display import display, HTML

display(HTML('''
<b>Answers - Hour 2</b><br><br>

<details>
<summary><b>Q1 - Click to reveal</b></summary>
<p><b>Answer: D</b> - Summarization</p>
<p>Condensing a paragraph into a single-sentence gist is a summarization task.</p>
</details>

<details>
<summary><b>Q2 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Normalizing the source documents tone does not help - the LLM generates its own text. Prompting, few-shot, and fine-tuning are valid approaches.</p>
</details>
'''))

# COMMAND ----------

# DBTITLE 1,Hour 3 RAG - Questions
# MAGIC %md
# MAGIC ## Hour 3: RAG Fundamentals & Document Processing (Demo 3)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q3.
# MAGIC
# MAGIC A Generative AI Engineer is assessing the responses from a customer-facing GenAI application that they are developing to assist in selling automotive parts. The application requires the customer to explicitly input account_id and transaction_id to answer questions. After initial launch, the customer feedback was that the application did well on answering order and billing details, but failed to accurately answer shipping and expected arrival date questions.
# MAGIC
# MAGIC Which of the following approaches would improve the application's ability to answer these questions?
# MAGIC
# MAGIC **A.** Create a vector store that includes the company shipping policies and payment terms for all automotive parts  
# MAGIC **B.** Create a feature store table with transaction_id as primary key that is populated with invoice data and expected delivery date  
# MAGIC **C.** Provide examples data for expected arrival dates as a tuning dataset, then periodically fine-tune the model so that it has updated shipping information  
# MAGIC **D.** Amend the chat prompt to input when the order was placed and instruct the model to add 14 days to that as no shipping method is expected to exceed 14 days
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q4.
# MAGIC
# MAGIC A Generative AI Engineer is building a RAG application that will rely on context retrieved from source documents that have been scanned and saved as image files in formats like .jpeg or .png. They want to develop a solution using the least amount of lines of code.
# MAGIC
# MAGIC Which Python package should be used to extract the text from the source documents?
# MAGIC
# MAGIC **A.** beautifulsoup  
# MAGIC **B.** scrapy  
# MAGIC **C.** pytesseract  
# MAGIC **D.** pyquery
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q5.
# MAGIC
# MAGIC You fine-tuned an instruction model for your internal knowledge base, but it often fabricates answers. You want to reduce hallucinations without another fine-tune. What's the best next step?
# MAGIC
# MAGIC **A.** Introduce retrieval-augmented generation using a curated Vector Search index and include citations in the prompt  
# MAGIC **B.** Increase temperature to encourage more diverse reasoning  
# MAGIC **C.** Switch to a larger model only  
# MAGIC **D.** Remove system prompts to avoid bias
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q6.
# MAGIC
# MAGIC A stakeholder wants the bot to answer only from approved sources and refuse otherwise, with a clear "I don't know" when evidence is weak. What's the best approach?
# MAGIC
# MAGIC **A.** Lower temperature and hope for the best  
# MAGIC **B.** Retrieval gating: require a minimum relevance threshold and include a refusal policy in the system prompt  
# MAGIC **C.** Increase top_p for diversity  
# MAGIC **D.** Use only embeddings without prompts
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q7.
# MAGIC
# MAGIC Your PDF-heavy corpus includes long tables that are poorly parsed into text, hurting retrieval quality. What's the best path?
# MAGIC
# MAGIC **A.** Ignore tables and rely on the model  
# MAGIC **B.** Add a table-aware extraction step that preserves structure; store both text and structured table data with metadata for retrieval  
# MAGIC **C.** Increase chunk size to include entire tables in each chunk  
# MAGIC **D.** Use only OCR text
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q8.
# MAGIC
# MAGIC A Generative AI Engineer is building a RAG application that answers questions about internal documents for the company SnoPen AI. The source documents may contain a significant amount of irrelevant content, such as advertisements, sports news, or entertainment news, or content about other companies.
# MAGIC
# MAGIC Which approach is advisable when building a RAG application to achieve this goal of filtering irrelevant information?
# MAGIC
# MAGIC **A.** Keep all articles because the RAG application needs to understand non-company content to avoid answering questions about them.  
# MAGIC **B.** Include in the system prompt that any information it sees will be about SnoPenAI, even if no data filtering is performed.  
# MAGIC **C.** Include in the system prompt that the application is not supposed to answer any questions unrelated to SnoPen AI.  
# MAGIC **D.** Consolidate all SnoPen AI related documents into a single chunk in the vector database.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q9.
# MAGIC
# MAGIC A company has a typical RAG-enabled, customer-facing chatbot. Select the correct sequence of components a user's question goes through before the final output is returned.
# MAGIC
# MAGIC **A.** 1.embedding model, 2.vector search, 3.context-augmented prompt, 4.response-generating LLM  
# MAGIC **B.** 1.context-augmented prompt, 2.vector search, 3.embedding model, 4.response-generating LLM  
# MAGIC **C.** 1.response-generating LLM, 2.vector search, 3.context-augmented prompt, 4.embedding model  
# MAGIC **D.** 1.response-generating LLM, 2.context-augmented prompt, 3.vector search, 4.embedding model
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q10.
# MAGIC
# MAGIC A Generative AI Engineer is building a RAG application that will rely on context retrieved from source documents that are currently in PDF format. These PDFs can contain both text and images. They want to develop a solution using the least amount of lines of code.
# MAGIC
# MAGIC Which Python package should be used to extract the text from the source documents?
# MAGIC
# MAGIC **A.** flask  
# MAGIC **B.** beautifulsoup  
# MAGIC **C.** unstructured  
# MAGIC **D.** numpy

# COMMAND ----------

# DBTITLE 1,Hour 3 RAG - Answers
from IPython.display import display, HTML

display(HTML('''
<b>Answers - Hour 3: RAG Fundamentals</b><br><br>

<details>
<summary><b>Q3 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>The application lacks delivery date data. A feature store table keyed by transaction_id provides structured delivery date data that the application can look up directly.</p>
</details>

<details>
<summary><b>Q4 - Click to reveal</b></summary>
<p><b>Answer: C</b> - pytesseract</p>
<p>pytesseract is an OCR wrapper for Tesseract that extracts text from images with minimal code.</p>
</details>

<details>
<summary><b>Q5 - Click to reveal</b></summary>
<p><b>Answer: A</b></p>
<p>RAG grounds responses in retrieved facts, reducing hallucinations. Citations add traceability.</p>
</details>

<details>
<summary><b>Q6 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Retrieval gating with a minimum relevance threshold ensures only high-confidence context is used. A refusal policy handles the "I don't know" case.</p>
</details>

<details>
<summary><b>Q7 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Table-aware extraction preserves structure that plain OCR loses. Storing both text and structured data with metadata enables accurate retrieval of tabular content.</p>
</details>

<details>
<summary><b>Q8 - Click to reveal</b></summary>
<p><b>Answer: C</b></p>
<p>A system prompt instruction to refuse unrelated questions is the simplest approach among the given options.</p>
</details>

<details>
<summary><b>Q9 - Click to reveal</b></summary>
<p><b>Answer: A</b></p>
<p>RAG flow: embed query, vector search for relevant chunks, augment prompt with retrieved context, generate response with LLM.</p>
</details>

<details>
<summary><b>Q10 - Click to reveal</b></summary>
<p><b>Answer: C</b> - unstructured</p>
<p>The unstructured library handles mixed-format PDFs (text + images) with minimal code.</p>
</details>
'''))

# COMMAND ----------

# DBTITLE 1,Hour 3 AI Search - Questions
# MAGIC %md
# MAGIC ## Hour 3: AI Search, Embeddings & Vector Search (Demo 3)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q11.
# MAGIC
# MAGIC A Generative AI Engineer is loading 150 million embeddings into a vector database that takes a maximum of 100 million. Which TWO actions can they take to reduce the record count?
# MAGIC
# MAGIC **A.** Increase the document chunk size  
# MAGIC **B.** Decrease the overlap between chunks  
# MAGIC **C.** Decrease the document chunk size  
# MAGIC **D.** Increase the overlap between chunks  
# MAGIC **E.** Use a smaller embedding model
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q12.
# MAGIC
# MAGIC A Generative AI Engineer is creating an LLM-based application. The documents for its retriever have been chunked to a maximum of 512 tokens each. The Generative AI Engineer knows that cost and latency are more important than quality for this application. They have several context length levels to choose from.
# MAGIC
# MAGIC Which will fulfill their need?
# MAGIC
# MAGIC **A.** context length 512: smallest model is 0.13GB and embedding dimension 384  
# MAGIC **B.** context length 514: smallest model is 0.44GB and embedding dimension 768  
# MAGIC **C.** context length 2048: smallest model is 11GB and embedding dimension 2560  
# MAGIC **D.** context length 32768: smallest model is 14GB and embedding dimension 4096
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q13.
# MAGIC
# MAGIC A Generative AI Engineer working for an online retailer is attempting to improve their search functionality with vector search plus metadata filtering. Searches can total as many as 80 per second and latency is their most critical metric. They don't mind up front development costs if it improves accuracy without harming latency. The inventory consists of 100 million items nationwide.
# MAGIC
# MAGIC How should the engineer set this up?
# MAGIC
# MAGIC **A.** Leverage GTE Large embedding model, use standard vector search with hybrid search and reranking turned on.  
# MAGIC **B.** Leverage GTE Large embedding model, use storage optimized vector search with hybrid search and reranking turned on.  
# MAGIC **C.** Fine tune a custom embedding model, use standard vector search, keep hybrid search and reranking off.  
# MAGIC **D.** Fine tune a custom embedding model, use storage optimized vector search, keep hybrid search and reranking off.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q14.
# MAGIC
# MAGIC You notice high retrieval latency from your Vector Search index. Chunks are 2,500 tokens and documents contain mixed topics. What is the most impactful remediation?
# MAGIC
# MAGIC **A.** Increase chunk size further to reduce index size  
# MAGIC **B.** Introduce semantic chunking with smaller, coherent chunks and add metadata filters for doc_type and product  
# MAGIC **C.** Remove metadata to simplify the index  
# MAGIC **D.** Use only keyword search because vector search is slower by default
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q15.
# MAGIC
# MAGIC Your RAG system sometimes returns irrelevant context due to ambiguous queries. You want to improve retrieval without changing the model. What should you try?
# MAGIC
# MAGIC **A.** Hybrid retrieval combining vector similarity with BM25 or metadata filters  
# MAGIC **B.** Increase temperature to encourage variety  
# MAGIC **C.** Remove metadata to reduce conflicts  
# MAGIC **D.** Use only embeddings trained on code
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q16.
# MAGIC
# MAGIC You notice retrieval quality degrades after frequent schema changes in source tables. Some embeddings don't match expected vector dimensions. What's the most robust fix?
# MAGIC
# MAGIC **A.** Re-embed only new rows  
# MAGIC **B.** Enforce a contract for embedding model + vector dimension; store model metadata alongside vectors; rebuild index when changing model  
# MAGIC **C.** Convert vectors to the new dimension by zero-padding  
# MAGIC **D.** Switch to keyword-only search
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q17.
# MAGIC
# MAGIC After adding richer context, the model sometimes exceeds token limits and truncates answers. What's the best immediate mitigation?
# MAGIC
# MAGIC **A.** Increase temperature  
# MAGIC **B.** Apply retrieval budget: limit number/size of chunks by dynamic relevance, compress or summarize context before generation  
# MAGIC **C.** Use a smaller model  
# MAGIC **D.** Remove system prompts
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q18.
# MAGIC
# MAGIC After changing the response generating LLM in a RAG pipeline from GPT-4 to a model with a shorter context length that the company self-hosts, the Generative AI Engineer is getting an error that the context exceeds the model's limit. What TWO solutions should the Generative AI Engineer implement without changing the response generating model? (Choose two.)
# MAGIC
# MAGIC **A.** Use a smaller embedding model to generate embeddings  
# MAGIC **B.** Reduce the maximum output tokens of the new model  
# MAGIC **C.** Decrease the chunk size of embedded documents  
# MAGIC **D.** Reduce the number of records retrieved from the vector database  
# MAGIC **E.** Retrain the response generating model using ALiBi
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q19.
# MAGIC
# MAGIC A Generative AI Engineer has successfully ingested unstructured documents and chunked them by document sections. They would like to store the chunks in a Vector Search index. The current format of the dataframe has two columns: (i) original document file name (ii) an array of text chunks for each document.
# MAGIC
# MAGIC What is the most performant way to store this dataframe?
# MAGIC
# MAGIC **A.** Split the data into train and test set, create a unique identifier for each document, then save to a Delta table  
# MAGIC **B.** Flatten the dataframe to one chunk per row, create a unique identifier for each row, and save to a Delta table  
# MAGIC **C.** First create a unique identifier for each document, then save to a Delta table  
# MAGIC **D.** Store each chunk as an independent JSON file in Unity Catalog Volume

# COMMAND ----------

# DBTITLE 1,Hour 3 AI Search - Answers
from IPython.display import display, HTML

display(HTML('''
<b>Answers - Hour 3: AI Search & Embeddings</b><br><br>

<details>
<summary><b>Q11 - Click to reveal</b></summary>
<p><b>Answer: A, B</b></p>
<p>Larger chunks = fewer records. Less overlap = fewer chunks. A smaller embedding model reduces vector dimensions, not record count.</p>
</details>

<details>
<summary><b>Q12 - Click to reveal</b></summary>
<p><b>Answer: A</b></p>
<p>Smallest model (0.13GB) with context length 512 matches the chunk size. Prioritizes cost/latency over quality.</p>
</details>

<details>
<summary><b>Q13 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Storage-optimized for 100M items. GTE Large + hybrid search + reranking improves accuracy without harming latency.</p>
</details>

<details>
<summary><b>Q14 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Smaller, coherent chunks reduce noise. Metadata filters narrow the search space, reducing latency.</p>
</details>

<details>
<summary><b>Q15 - Click to reveal</b></summary>
<p><b>Answer: A</b></p>
<p>Hybrid retrieval combines semantic (vector) and lexical (BM25) search, improving recall for ambiguous queries.</p>
</details>

<details>
<summary><b>Q16 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Enforce model+dimension contract; rebuild index when model changes. Zero-padding produces garbage vectors.</p>
</details>

<details>
<summary><b>Q17 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Retrieval budget limits chunks fed to the model. Compressing context before generation reduces token usage.</p>
</details>

<details>
<summary><b>Q18 - Click to reveal</b></summary>
<p><b>Answer: C, D</b></p>
<p>Decrease chunk size + reduce retrieved records. Both cut context length without changing the model.</p>
</details>

<details>
<summary><b>Q19 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Vector Search requires one row per chunk with a unique ID. Flattening ensures each chunk is individually retrievable.</p>
</details>
'''))

# COMMAND ----------

# DBTITLE 1,Hours 4-5 - Questions
# MAGIC %md
# MAGIC ## Hours 4-5: Agents, Tools & Multi-Agent (Demo 4, 5, 6)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q20.
# MAGIC
# MAGIC You added tool/function-calling to let the model query an internal REST API for order status. Sometimes the LLM hallucinates tool parameters. How can you improve reliability?
# MAGIC
# MAGIC **A.** Allow free-form text for tool arguments  
# MAGIC **B.** Provide JSON schemas for tool inputs and validate before execution; include few-shot examples of correct tool usage  
# MAGIC **C.** Increase temperature  
# MAGIC **D.** Remove function calling and hardcode API calls
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q21.
# MAGIC
# MAGIC Your RAG answers include stale prices for SKUs that change daily. You already re-embed content nightly. What else should you do?
# MAGIC
# MAGIC **A.** Add a tool/function call to fetch live pricing for cited SKUs and instruct the model to prioritize tool data over retrieved chunks  
# MAGIC **B.** Increase vector dimension  
# MAGIC **C.** Disable retrieval and use the tool only  
# MAGIC **D.** Reduce k in retrieval
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q22.
# MAGIC
# MAGIC A Generative AI Engineer is creating an LLM-powered application that will need access to up-to-date news articles and stock prices. The design requires the use of stock prices which are stored in Delta tables and finding the latest relevant news articles by searching the internet.
# MAGIC
# MAGIC How should the Generative AI Engineer architect their LLM system?
# MAGIC
# MAGIC **A.** Use an LLM to summarize the latest news articles and lookup stock tickers from the summaries to find stock prices.  
# MAGIC **B.** Query the Delta table for volatile stock prices and use an LLM to generate a search query to investigate potential causes of the stock volatility.  
# MAGIC **C.** Download and store news articles and stock price information in a vector store. Use a RAG architecture to retrieve and generate at runtime.  
# MAGIC **D.** Create an agent with tools for SQL querying of Delta tables and web searching, provide retrieved values to an LLM for generation of response.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q23.
# MAGIC
# MAGIC A Generative AI Engineer received the following business requirements for an external chatbot. The chatbot needs to know what types of questions the user asks and routes to appropriate models to answer the questions. For example, the user might ask about upcoming event details. Another user might ask about purchasing tickets for a particular event.
# MAGIC
# MAGIC What is an ideal workflow for such a chatbot?
# MAGIC
# MAGIC **A.** The chatbot should only look at previous event information  
# MAGIC **B.** There should be two different chatbots handling different types of user queries.  
# MAGIC **C.** The chatbot should be implemented as a multi-step LLM workflow. First, identify the type of question asked, then route the question to the appropriate model. If it's an upcoming event question, send the query to a text-to-SQL model. If it's about ticket purchasing, the customer should be redirected to a payment platform.  
# MAGIC **D.** The chatbot should only process payments

# COMMAND ----------

# DBTITLE 1,Hours 4-5 - Answers
from IPython.display import display, HTML

display(HTML('''
<b>Answers - Hours 4-5: Agents & Multi-Agent</b><br><br>

<details>
<summary><b>Q20 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>JSON schemas constrain the model output to valid parameters. Validation catches errors before execution. Few-shot examples guide correct tool usage.</p>
</details>

<details>
<summary><b>Q21 - Click to reveal</b></summary>
<p><b>Answer: A</b></p>
<p>For rapidly changing data, a tool call to the live source is more reliable than nightly re-embedding. The system prompt should instruct the model to use tool data for current pricing.</p>
</details>

<details>
<summary><b>Q22 - Click to reveal</b></summary>
<p><b>Answer: D</b></p>
<p>An agent with tools (SQL for Delta tables, web search for news) can access real-time data on demand. RAG alone cannot handle live stock prices or fresh news.</p>
</details>

<details>
<summary><b>Q23 - Click to reveal</b></summary>
<p><b>Answer: C</b></p>
<p>A multi-step LLM workflow with question-type identification and routing is the most flexible approach. It handles different question types with appropriate models and actions.</p>
</details>
'''))

# COMMAND ----------

# DBTITLE 1,Hour 6 - Questions
# MAGIC %md
# MAGIC ## Hour 6: Security, Governance & Evaluation (Demo 4, 7)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q24.
# MAGIC
# MAGIC You're onboarding a multi-tenant RAG app across business units with strict data separation. What's the primary control to enforce isolation of embeddings and source documents?
# MAGIC
# MAGIC **A.** Model endpoint tokens  
# MAGIC **B.** Unity Catalog permissions on source tables and vector indexes, + service principals scoped per tenant  
# MAGIC **C.** Notebook ACLs only  
# MAGIC **D.** Row-level security in notebooks via Python conditionals
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q25.
# MAGIC
# MAGIC Your legal team requires removal of sensitive PII in prompts and model outputs. You want minimal developer friction. What should you deploy?
# MAGIC
# MAGIC **A.** Manual developer checklist  
# MAGIC **B.** Pre/post-processing policies that redact PII at the gateway or serving layer, with audit logs  
# MAGIC **C.** Remove logs entirely  
# MAGIC **D.** Ask users not to enter PII
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q26.
# MAGIC
# MAGIC The security team wants visibility into who called which model, with what data classes, and when. What should you enable?
# MAGIC
# MAGIC **A.** Random sampling of prompts in notebooks  
# MAGIC **B.** Centralized access logs and lineage across data sources and serving endpoints, with Unity Catalog tags for sensitive data  
# MAGIC **C.** Save logs to a local file  
# MAGIC **D.** Disable access to reduce risk
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q27.
# MAGIC
# MAGIC You're asked to launch an A/B test of two prompts for the same endpoint to reduce hallucinations. You need traffic splitting and win-rate measurement. What's the best plan?
# MAGIC
# MAGIC **A.** Manually alternate between prompts in the notebook  
# MAGIC **B.** Use a gateway or routing layer to split traffic between prompt variants, and log outcomes for statistical comparison  
# MAGIC **C.** Launch both in separate workspaces and compare logs by hand  
# MAGIC **D.** Switch models instead of prompts
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q28.
# MAGIC
# MAGIC You need to roll out a new model version but want a safe migration with minimal risk to production users. What should you do?
# MAGIC
# MAGIC **A.** Replace the model immediately  
# MAGIC **B.** Run shadow or canary traffic for the new version, monitor metrics and feedback, then gradually increase traffic  
# MAGIC **C.** Force all users to test in dev first  
# MAGIC **D.** Disable logging during rollout
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q29.
# MAGIC
# MAGIC A Generative AI Engineer is responsible for evaluating a customer-support RAG assistant. Four domain experts review sampled answers each week using dimensions such as factual accuracy, completeness, and usefulness. After several rounds, the engineer notices that expert ratings vary widely for the same responses, making the evaluation data unreliable.
# MAGIC
# MAGIC What should the engineer do?
# MAGIC
# MAGIC **A.** Use an LLM-as-a-judge to rescore past and future responses, and treat the model-generated ratings as the primary source of truth instead of reconciling expert disagreement.  
# MAGIC **B.** Define clear rubrics, calibrate SMEs on the criteria, and use the aligned judgments for consistent agent evaluation.  
# MAGIC **C.** Average the scores from all domain experts for each response and use the blended score directly as the definitive benchmark for model tuning.  
# MAGIC **D.** Build the benchmark only from responses where all experts already agree, and exclude disputed cases from the evaluation set to improve consistency.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q30.
# MAGIC
# MAGIC The team wants to evaluate end-to-end task success (not just BLEU/ROUGE) for a claims-processing agent that calls multiple tools. What should you implement?
# MAGIC
# MAGIC **A.** Only measure token counts  
# MAGIC **B.** Task-level success metrics with golden tasks, plus step-level traces for tool calls and failures  
# MAGIC **C.** Per-token probabilities  
# MAGIC **D.** Context window utilization
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q31.
# MAGIC
# MAGIC A Generative AI Engineer is building a system which will answer questions on latest stock news articles. Which will NOT help with ensuring the outputs are relevant to financial news?
# MAGIC
# MAGIC **A.** Implement a comprehensive guardrail framework that includes policies for content filters tailored to the finance sector.  
# MAGIC **B.** Increase the compute to improve processing speed of questions to allow greater relevancy analysis  
# MAGIC **C.** Implement a profanity filter to screen out offensive language.  
# MAGIC **D.** Incorporate manual reviews to correct any problematic outputs prior to sending to the users.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Q32.
# MAGIC
# MAGIC A Generative AI Engineer has created a RAG application which can help employees retrieve answers from an internal knowledge base. The prototype is working with positive feedback. Now the engineer wants to formally evaluate the system's performance and understand where to focus improvement efforts.
# MAGIC
# MAGIC How should the Generative AI Engineer evaluate the system?
# MAGIC
# MAGIC **A.** Use cosine similarity score to comprehensively evaluate the quality of the final generated answers.  
# MAGIC **B.** Curate a dataset that can test the retrieval and generation components of the system separately. Use built-in evaluation metrics to perform the evaluation on the retrieval and generation components.  
# MAGIC **C.** Benchmark multiple LLMs with the same data and pick the best LLM for the job.  
# MAGIC **D.** Use an LLM-as-a-judge to evaluate the quality of the final answers generated.

# COMMAND ----------

# DBTITLE 1,Hour 6 - Answers
from IPython.display import display, HTML

display(HTML('''
<b>Answers - Hour 6: Security, Governance & Evaluation</b><br><br>

<details>
<summary><b>Q24 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Unity Catalog provides table-level and column-level permissions. Scoped service principals per tenant ensure each tenant only accesses their own data and indexes.</p>
</details>

<details>
<summary><b>Q25 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Gateway/serving-layer PII policies automate redaction without requiring developers to modify each prompt. Audit logs provide compliance traceability.</p>
</details>

<details>
<summary><b>Q26 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Centralized access logs and lineage provide full visibility. UC tags for sensitive data enable governance policies.</p>
</details>

<details>
<summary><b>Q27 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>AI Gateway or a routing layer enables programmatic traffic splitting between prompt variants. Logging outcomes allows statistical comparison of win rates.</p>
</details>

<details>
<summary><b>Q28 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Shadow/canary deployment routes a small percentage of traffic to the new model. Monitoring metrics before gradual increase minimizes risk to production users.</p>
</details>

<details>
<summary><b>Q29 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Clear rubrics and SME calibration reduce rater inconsistency. Aligned judgments produce reliable evaluation data for tracking improvements over time.</p>
</details>

<details>
<summary><b>Q30 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Task-level success metrics (golden tasks) measure end-to-end outcomes. Step-level traces diagnose tool call failures. BLEU/ROUGE only measure text similarity.</p>
</details>

<details>
<summary><b>Q31 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Increasing compute does not improve relevance - it only speeds up processing. Guardrails, profanity filters, and manual reviews all directly address content quality.</p>
</details>

<details>
<summary><b>Q32 - Click to reveal</b></summary>
<p><b>Answer: B</b></p>
<p>Separating retrieval and generation evaluation pinpoints which component needs improvement. A curated dataset with dedicated metrics for each is the most systematic approach.</p>
</details>
'''))

# COMMAND ----------

# DBTITLE 1,Removed Questions
# MAGIC %md
# MAGIC ## Questions Removed (Not Covered in Module 5A Demos)
# MAGIC
# MAGIC The following questions from the Sample Questions notebook were **excluded** because their topics are NOT demonstrated in any Module 5A demo. Most belong to Module 5B (MLflow/ML lifecycle) or Module 6B (deployment/governance) of the official curriculum.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Removed: Model Deployment & Serving Infrastructure (7 questions)
# MAGIC
# MAGIC | Question | Why Removed | Official Module |
# MAGIC |---|---|---|
# MAGIC | Prompt optimization (templates, caching) | Not discussed in any demo — Demo 2 covers temperature, not prompt compression | 6B |
# MAGIC | Batch inference for cost reduction | Not discussed — Demo 2 mentions “Deploy” as a stage name only | 6B |
# MAGIC | Batch inference for structured output | Same — no batch inference demo | 6B |
# MAGIC | Model Serving autoscaling | Not discussed in any demo | 6B |
# MAGIC | Warm instances / scale-to-zero | Not discussed in any demo | 6B |
# MAGIC | Model deployment workflow (log, register, serve) | Not discussed — this is MLflow-specific (removed from Module 5A) | 5B |
# MAGIC | Provisioned throughput vs pay-per-token | Not discussed in any demo | 6B |
# MAGIC
# MAGIC ### Removed: Model Knowledge (2 questions)
# MAGIC
# MAGIC | Question | Why Removed |
# MAGIC |---|---|
# MAGIC | CodeLlama for code generation | Demo 2 compares Llama models generically, but CodeLlama specifically is never mentioned |
# MAGIC | DBRX for large context window | Not mentioned in any demo |
# MAGIC
# MAGIC ### Removed: Other Topics Not in Demos (5 questions)
# MAGIC
# MAGIC | Question | Why Removed |
# MAGIC |---|---|
# MAGIC | Chatbot engagement metric (diversity) | Not discussed in any demo |
# MAGIC | Streaming RAG architecture (Bronze → Silver) | Streaming is a Module 4 topic; Demo 3 covers AI Search index creation, not streaming pipelines |
# MAGIC | Multilingual RAG | Demo 2 covers ai_translate and Demo 3 covers embeddings, but multilingual embedding strategies aren't discussed |
# MAGIC | Distributed embedding generation with Spark | Demo 3 covers embeddings but not Spark-based distributed generation specifically |
# MAGIC | Production monitoring with MRR/Recall@k | Demo 7 covers LLM-as-judge conceptually, but specific retrieval metrics (MRR, Recall@k) aren't discussed |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Syllabus Headings NOT Backed by Demo Content
# MAGIC
# MAGIC These topics appear in the syllabus but are only mentioned as headings, not actually demonstrated:
# MAGIC
# MAGIC | Syllabus Topic | Demo Coverage | Status |
# MAGIC |---|---|---|
# MAGIC | 2.10 - Deploy stage | Only mentioned as a name in a table ("Deploy — Model Serving, Agent Bricks, Genie Spaces") | Heading only |
# MAGIC | 2.9 - Compound AI Systems | Demo 2 covers chaining AI functions, but the full compound system concept (model + retrieval + tools + classical ML + memory + guardrails) is a notes-only section | Partially |
# MAGIC | 4.7 - AI Gateway | Demo 4 covers it conceptually (print statements), not hands-on | Concept only |
# MAGIC | 6.1 - Platform Map | Demo 7 has a comparison table but no hands-on demo | Concept only |
# MAGIC | 6.2 - Adoption Roadmap | Demo 7 has notes only, no demo | Notes only |