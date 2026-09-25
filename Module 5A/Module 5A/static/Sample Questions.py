# Databricks notebook source
Question 1
Objective: Apply a chunking strategy for a given document structure and model constraints
A Generative AI Engineer is loading 150 million embeddings into a vector database that takes a
maximum of 100 million.
Which TWO actions can they take to reduce the record count?
A. Increase the document chunk size
B. Decrease the overlap between chunks
C. Decrease the document chunk size
D. Increase the overlap between chunks
E. Use a smaller embedding model
Question 2
Objective: Identify needed source documents that provide necessary knowledge and quality for a
given RAG application.
A Generative AI Engineer is assessing the responses from a customer-facing GenAI application
that they are developing to assist in selling automotive parts. The application requires the
customer to explicitly input account_id and transaction_id to answer questions. After initial
launch, the customer feedback was that the application did well on answering order and billing
details, but failed to accurately answer shipping and expected arrival date questions.
Which of the following approaches would improve the application's ability to answer these
questions?
A. Create a vector store that includes the company shipping policies and payment terms for
all automotive parts
B. Create a feature store table with transaction_id as primary key that is populated with
invoice data and expected delivery date
C. Provide examples data for expected arrival dates as a tuning dataset, then periodically
fine-tune the model so that it has updated shipping information
D. Amend the chat prompt to input when the order was placed and instruct the model to add
14 days to that as no shipping method is expected to exceed 14 days
Question 3
Objective: Choose the appropriate Python package to extract document content from provided
source data and format.
A Generative AI Engineer is building a RAG application that will rely on context retrieved from
source documents that have been scanned and saved as image files in formats like .jpeg or .png.
They want to develop a solution using the least amount of lines of code.
Which Python package should be used to extract the text from the source documents?
A. beautifulsoup
B. scrapy
C. pytesseract
D. pyquery
Question 4
Objective: Select an embedding model context length based on source documents, expected
queries, and optimization strategy
A Generative AI Engineer is creating a LLM-based application. The documents for its retriever have
been chunked to a maximum of 512 tokens each. The Generative AI Engineer knows that cost and
latency are more important than quality for this application. They have several context length
levels to choose from.
Which will fulfill their need?
A. context length 512: smallest model is 0.13GB and embedding dimension 384
B. context length 514: smallest model is 0.44GB and embedding dimension 768
C. context length 2048: smallest model is 11GB and embedding dimension 2560
D. context length 32768: smallest model is 14GB and embedding dimension 4096
Question 5
Objective: Select the best LLM based on the attributes of the application to be developed
A Generative AI Engineer would like to build an application that can update a memo field that is
about a paragraph long to just a single sentence gist that shows intent of the memo field, but fits
into their application front end.
With which Natural Language Processing task category should they evaluate potential LLMs for this
application?
A. text2text Generation
B. Sentencizer
C. Text Classification
D. Summarization
Question 6
Objective: Configure vector search for a particular solution based on number of embeddings,
update frequency, latency, and cost requirements.
A Generative AI Engineer working for an online retailer is attempting to improve their search
functionality with vector search plus metadata filtering. Searches can total as many as 80 per
second and latency is their most critical metric. They don't mind up front development costs if it
improves accuracy without harming latency. The inventory consists of 100 million items
nationwide.
How should the engineer set this up?
A. Leverage GTE Large embedding model, use standard vector search with hybrid search and
reranking turned on.
B. Leverage GTE Large embedding model, use storage optimized vector search with hybrid
search and reranking turned on.
C. Fine tune a custom embedding model, use standard vector search, keep hybrid search and
reranking off.
D. Fine tune a custom embedding model, use storage optimized vector search, keep hybrid
search and reranking off.
Question 7
Objective: Apply CI/CD best practices such as updating a Vector Search index, promoting
prompts across environments, and testing individual components of an agent.
A Generative AI Engineer needs to manage prompt templates for an agent across development,
staging, and production. The team requires a gated release process: prompts are updated in
development, validated in staging with automated tests, and promoted to production only after
approval. The solution must preserve version history and allow rollback to a previous prompt
version if needed.
Which approach supports this promotion workflow?
A. Store prompt templates in the application repository and promote them by merging the
staging branch into the production branch after tests pass.
B. Track prompts as MLflow versions and promote the prompts using aliases after they pass.
C. Save prompts in a JSON file on the CI runner and overwrite the prod prompt on each run.
D. Put prompts in Delta tables and overwrite the table in prod on every deployment to ensure
consistency.
Question 8
Objective: Develop an appropriate interactive user facing interface for an agent usage scenario
(Apps, Slack, Teams, etc.).
Generative AI Engineer is building a Databricks App that lets customer support agents ask
questions and receive answers grounded in internal PDFs. Requirements: users must authenticate
with their corporate identity, the app must call a Mosaic AI Agent endpoint without exposing
long-lived tokens in the browser, and access to answers must respect each user’s permissions.
Which approach meets these requirements?
A. Use a Databricks App backend to call the Agent endpoint with the app’s credentials and
enforce user identity/permissions via the app’s authenticated context.
B. Store a Databricks personal access token (PAT) in the app’s JavaScript and call the Agent
endpoint directly from the browser.
C. Publish the Agent endpoint publicly and protect it with an API key embedded in the app
frontend.
D. Export the PDFs to a public bucket so the Agent can read them without identity checks.
Question 9
Objective: Integrate managed, external, and custom MCP servers based on a given application
requirements.
A Generative AI Engineer is building a research assistant agent that needs to access factual
information from an internet data source and perform web searches using an external API.
Databricks provides a managed MCP server for this internet data source, and an external MCP
server is available for the external API that requires a key. The application must minimize
maintenance overhead while ensuring reliable access to both data sources in production.
Which two actions should the engineer take to integrate these data sources into the agent?
A. Build a custom MCP server that wraps both the internet resource and the external APIs into
a single unified interface for the agent to call.
B. Use the managed web browser MCP server to programmatically navigate to the internet
resources for retrieving information.
C. Configure Unity Catalog external tables to cache the internet resources's content and also
the search results for offline access by the agent.
D. Configure the managed MCP server through the agent's MCP server configuration by
specifying the server type as "managed" and providing the internet resource's server
identifier.
E. Deploy the external MCP server by providing its connection details, storing the API key in
Databricks Secrets, and referencing it in the MCP server configuration.
Question 10
Objective: Incorporate SME feedback to improve agent performance.
A Generative AI Engineer is responsible for evaluating a customer-support RAG assistant used
internally by operations teams. Four domain experts review sampled answers each week in MLflow
using dimensions such as factual accuracy, completeness, and usefulness. After several rounds,
the engineer notices that expert ratings vary widely for the same responses, making the evaluation
data unreliable for tracking model improvements over time. The engineer needs to create a
dependable evaluation process that reduces rater inconsistency while supporting iterative quality
improvements.
What should the engineer do?
A. Use an LLM-as-a-judge to rescore past and future responses, and treat the
model-generated ratings as the primary source of truth instead of reconciling expert
disagreement.
B. Define clear rubrics, calibrate SMEs on the criteria, and use the aligned judgments in
mlflow.genai.evaluate() for consistent agent evaluation.
C. Average the scores from all domain experts for each response and use the blended score
directly as the definitive benchmark for model tuning.
D. Build the benchmark only from responses where all experts already agree, and exclude
disputed cases from the evaluation set to improve consistency

# COMMAND ----------


Hi, Here are some example questions for you to practise on, I used a LLM to generate them and could use a similar approach to generate some more. I'd also recommend looking at some of the exam prep content on udemy, there are a few courses on there with many example questions. That's what I did to prepare for my exam, it helped me to nail down the concepts I needed to understand better.


1) You’re building a customer-support RAG chatbot on Databricks. New PDFs arrive hourly to a Bronze Delta table. You need low-latency retrieval with up-to-the-minute content. What’s the best architecture? A) Nightly batch embed PDFs to a Delta table and query with LIKE filters B) Stream Bronze → Silver, chunk + embed in a streaming job, sync to a Vector Search index, and query via the index C) Write all chunks to Parquet and use approximate nearest neighbor in a Python UDF D) Use MLflow to store embeddings and query with Delta ZORDER

2) Your LLM endpoint experiences sudden 5–10x traffic spikes during product launches, causing timeouts. You want to keep costs modest during normal traffic and scale up automatically during spikes. What should you prioritize? A) Disable autoscaling to avoid scale-up delays B) Use serverless Model Serving with autoscaling and warm pool settings tuned to expected bursts C) Run a single large GPU node 24/7 to avoid cold starts D) Move to batch inference for all traffic

3) You fine-tuned an instruction model for your internal knowledge base, but it often fabricates answers. You want to reduce hallucinations without another fine-tune. What’s the best next step? A) Introduce retrieval-augmented generation using a curated Vector Search index and include citations in the prompt B) Increase temperature to encourage more diverse reasoning C) Switch to a larger model only D) Remove system prompts to avoid bias

4) You’re onboarding a multi-tenant RAG app across business units with strict data separation. What’s the primary control to enforce isolation of embeddings and source documents? A) Model endpoint tokens B) Unity Catalog permissions on source tables and vector indexes, + service principals scoped per tenant C) Notebook ACLs only D) Row-level security in notebooks via Python conditionals

5) You notice high retrieval latency from your Vector Search index. Chunks are 2,500 tokens and documents contain mixed topics. What is the most impactful remediation? A) Increase chunk size further to reduce index size B) Introduce semantic chunking with smaller, coherent chunks and add metadata filters for doc_type and product C) Remove metadata to simplify the index D) Use only keyword search because vector search is slower by default

6) A product manager requests “Why did the model choose this answer?” for every chat response in production. You also need to compare retriever performance over time. What should you implement first? A) Only prompt logs in MLflow B) Model Serving request/response logging with retrieved contexts, and offline evaluations on retrieval quality (e.g., MRR/Recall@k) with versioned datasets C) A/B test two model sizes without logging D) Disable logging due to PII concerns

7) Your RAG system sometimes returns irrelevant context due to ambiguous queries. You want to improve retrieval without changing the model. What should you try? A) Hybrid retrieval combining vector similarity with BM25 or metadata filters B) Increase temperature to encourage variety C) Remove metadata to reduce conflicts D) Use only embeddings trained on code

😎 You’re migrating from a prototype to production. The team wants reproducible experiments, prompt versioning, and offline evaluations over a fixed test set. Which combination fits best? A) Store prompts and results as notebook markdown B) Track prompts, parameters, and metrics with MLflow, and run eval notebooks regularly against a Unity Catalog curated dataset C) Save everything in CSVs on DBFS D) Add comments to the model endpoint

9) A finance team needs a scheduled job to generate structured summaries (JSON) from new transactions daily. The priority is consistent schema and downstream parsing, not chat. What is the best approach? A) Use batch inference with a structured output schema and store results in a Delta table B) Use interactive chat endpoints with human supervision C) Log raw model text to a JSON column and parse downstream D) Use notebooks only, without any model serving

10) Your endpoint costs have doubled, and CPU utilization is low while GPU is medium. Most prompts are long, with repeated instructions. What should you optimize first? A) Compress or template the system prompt; leverage prompt templates and caching where feasible B) Scale up GPUs to reduce latency C) Increase context window size D) Disable autoscaling and run a fixed large cluster

11) You added tool/function-calling to let the model query an internal REST API for order status. Sometimes the LLM hallucinates tool parameters. How can you improve reliability? A) Allow free-form text for tool arguments B) Provide JSON schemas for tool inputs and validate before execution; include few-shot examples of correct tool usage C) Increase temperature D) Remove function calling and hardcode API calls

12) Your legal team requires removal of sensitive PII in prompts and model outputs. You want minimal developer friction. What should you deploy? A) Manual developer checklist B) Pre/post-processing policies that redact PII at the gateway or serving layer, with audit logs C) Remove logs entirely D) Ask users not to enter PII

13) You notice retrieval quality degrades after frequent schema changes in source tables. Some embeddings don’t match expected vector dimensions. What’s the most robust fix? A) Re-embed only new rows B) Enforce a contract for embedding model + vector dimension; store model metadata alongside vectors; rebuild index when changing model C) Convert vectors to the new dimension by zero-padding D) Switch to keyword-only search

14) You’re asked to launch an A/B test of two prompts for the same endpoint to reduce hallucinations. You need traffic splitting and win-rate measurement. What’s the best plan? A) Manually alternate between prompts in the notebook B) Use a gateway or routing layer to split traffic between prompt variants, and log outcomes for statistical comparison C) Launch both in separate workspaces and compare logs by hand D) Switch models instead of prompts

15) Your chatbot’s first-token latency is high after periods of inactivity. You cannot afford constant overprovisioning. What should you try? A) Increase autoscaling cooldown to scale down faster B) Configure a minimum number of warm instances and adjust scale-to-zero behavior for expected idle windows C) Use larger models to produce tokens faster D) Disable logging

16) A stakeholder wants the bot to answer only from approved sources and refuse otherwise, with a clear “I don’t know” when evidence is weak. What’s the best approach? A) Lower temperature and hope for the best B) Retrieval gating: require a minimum relevance threshold and include a refusal policy in the system prompt C) Increase top_p for diversity D) Use only embeddings without prompts

17) You must support multilingual queries on an English corpus and return English answers with citations. What is the safest approach? A) Translate corpus to all possible languages B) Use multilingual embeddings for retrieval, translate the query to English if needed, and instruct the model to answer in English with citations C) Force user to ask in English D) Use monolingual embeddings and increase k

18) You’re backfilling embeddings for 50M documents. Index build speed is too slow and blocking launch. What will help most? A) Single-threaded local job B) Distributed embedding generation using Spark, write to Delta in batches, and build the vector index incrementally with parallelism C) Embed on the serving endpoint D) Switch to a larger model first

19) The team wants to evaluate end-to-end task success (not just BLEU/ROUGE) for a claims-processing agent that calls multiple tools. What should you implement? A) Only measure token counts B) Task-level success metrics with golden tasks, plus step-level traces for tool calls and failures C) Per-token probabilities D) Context window utilization

20) After adding richer context, the model sometimes exceeds token limits and truncates answers. What’s the best immediate mitigation? A) Increase temperature B) Apply retrieval budget: limit number/size of chunks by dynamic relevance, compress or summarize context before generation C) Use a smaller model D) Remove system prompts

21) You’ve deployed a content-generation job that runs nightly with stable load, and latency is not critical. How can you reduce costs? A) Switch to batch inference on scheduled jobs and right-size compute to cheaper instances B) Force serverless real-time endpoints C) Always keep two warm GPUs D) Add more replicas to reduce duration

22) Your RAG answers include stale prices for SKUs that change daily. You already re-embed content nightly. What else should you do? A) Add a tool/function call to fetch live pricing for cited SKUs and instruct the model to prioritize tool data over retrieved chunks B) Increase vector dimension C) Disable retrieval and use the tool only D) Reduce k in retrieval

23) The security team wants visibility into who called which model, with what data classes, and when. What should you enable? A) Random sampling of prompts in notebooks B) Centralized access logs and lineage across data sources and serving endpoints, with Unity Catalog tags for sensitive data C) Save logs to a local file D) Disable access to reduce risk

24) Your PDF-heavy corpus includes long tables that are poorly parsed into text, hurting retrieval quality. What’s the best path? A) Ignore tables and rely on the model B) Add a table-aware extraction step that preserves structure; store both text and structured table data with metadata for retrieval C) Increase chunk size to include entire tables in each chunk D) Use only OCR text

25) You need to roll out a new model version but want a safe migration with minimal risk to production users. What should you do? A) Replace the model immediately B) Run shadow or canary traffic for the new version, monitor metrics and feedback, then gradually increase traffic C) Force all users to test in dev first D) Disable logging during rollout



# COMMAND ----------


Exam List
Login
Register
Contact
Blog
Certified Generative AI Engineer Associate v1.0
Page:    1 / 8   
Exam contains 109 questions
Question 1
After changing the response generating LLM in a RAG pipeline from GPT-4 to a model with a shorter context length that the company self-hosts, the Generative AI Engineer is getting the following error:
image8
What TWO solutions should the Generative AI Engineer implement without changing the response generating model? (Choose two.)

A. Use a smaller embedding model to generate embeddings
B. Reduce the maximum output tokens of the new model
C. Decrease the chunk size of embedded documents
D. Reduce the number of records retrieved from the vector database
E. Retrain the response generating model using ALiBi


Answer : CD

Question 2
A Generative Al Engineer is building a system which will answer questions on latest stock news articles.
Which will NOT help with ensuring the outputs are relevant to financial news?

A. Implement a comprehensive guardrail framework that includes policies for content filters tailored to the finance sector.
B. Increase the compute to improve processing speed of questions to allow greater relevancy analysis
C. Implement a profanity filter to screen out offensive language.
D. Incorporate manual reviews to correct any problematic outputs prior to sending to the users


Answer : B

Question 3
A Generative Al Engineer is building a RAG application that answers questions about internal documents for the company SnoPen AI.
The source documents may contain a significant amount of irrelevant content, such as advertisements, sports news, or entertainment news, or content about other companies.
Which approach is advisable when building a RAG application to achieve this goal of filtering irrelevant information?

A. Keep all articles because the RAG application needs to understand non-company content to avoid answering questions about them.
B. Include in the system prompt that any information it sees will be about SnoPenAI, even if no data filtering is performed.
C. Include in the system prompt that the application is not supposed to answer any questions unrelated to SnoPen AI.
D. Consolidate all SnoPen AI related documents into a single chunk in the vector database.


Answer : C

Question 4
A Generative Al Engineer has successfully ingested unstructured documents and chunked them by document sections. They would like to store the chunks in a Vector Search index. The current format of the dataframe has two columns: (i) original document file name (ii) an array of text chunks for each document.
What is the most performant way to store this dataframe?

A. Split the data into train and test set, create a unique identifier for each document, then save to a Delta table
B. Flatten the dataframe to one chunk per row, create a unique identifier for each row, and save to a Delta table
C. First create a unique identifier for each document, then save to a Delta table
D. Store each chunk as an independent JSON file in Unity Catalog Volume. For each JSON file, the key is the document section name and the value is the array of text chunks for that section


Answer : B

Question 5
A Generative AI Engineer has created a RAG application which can help employees retrieve answers from an internal knowledge base, such as Confluence pages or Google Drive. The prototype application is now working with some positive feedback from internal company testers. Now the Generative Al Engineer wants to formally evaluate the system’s performance and understand where to focus their efforts to further improve the system.
How should the Generative AI Engineer evaluate the system?

A. Use cosine similarity score to comprehensively evaluate the quality of the final generated answers.
B. Curate a dataset that can test the retrieval and generation components of the system separately. Use MLflow’s built in evaluation metrics to perform the evaluation on the retrieval and generation components.
C. Benchmark multiple LLMs with the same data and pick the best LLM for the job.
D. Use an LLM-as-a-judge to evaluate the quality of the final answers generated.


Answer : B

Question 6
A Generative Al Engineer has already trained an LLM on Databricks and it is now ready to be deployed.
Which of the following steps correctly outlines the easiest process for deploying a model on Databricks?

A. Log the model as a pickle object, upload the object to Unity Catalog Volume, register it to Unity Catalog using MLflow, and start a serving endpoint
B. Log the model using MLflow during training, directly register the model to Unity Catalog using the MLflow API, and start a serving endpoint
C. Save the model along with its dependencies in a local directory, build the Docker image, and run the Docker container
D. Wrap the LLM’s prediction function into a Flask application and serve using Gunicorn


Answer : B

Question 7
A Generative AI Engineer developed an LLM application using the provisioned throughput Foundation Model API. Now that the application is ready to be deployed, they realize their volume of requests are not sufficiently high enough to create their own provisioned throughput endpoint. They want to choose a strategy that ensures the best cost-effectiveness for their application.
What strategy should the Generative AI Engineer use?

A. Switch to using External Models instead
B. Deploy the model using pay-per-token throughput as it comes with cost guarantees
C. Change to a model with a fewer number of parameters in order to reduce hardware constraint issues
D. Throttle the incoming batch of requests manually to avoid rate limiting issues


Answer : B

Question 8
A Generative AI Engineer is building an LLM to generate article summaries in the form of a type of poem, such as a haiku, given the article content. However, the initial output from the LLM does not match the desired tone or style.
Which approach will NOT improve the LLM’s response to achieve the desired response?

A. Provide the LLM with a prompt that explicitly instructs it to generate text in the desired tone and style
B. Use a neutralizer to normalize the tone and style of the underlying documents
C. Include few-shot examples in the prompt to the LLM
D. Fine-tune the LLM on a dataset of desired tone and style


Answer : B

Question 9
A Generative AI Engineer is creating an LLM-powered application that will need access to up-to-date news articles and stock prices.
The design requires the use of stock prices which are stored in Delta tables and finding the latest relevant news articles by searching the internet.
How should the Generative AI Engineer architect their LLM system?

A. Use an LLM to summarize the latest news articles and lookup stock tickers from the summaries to find stock prices.
B. Query the Delta table for volatile stock prices and use an LLM to generate a search query to investigate potential causes of the stock volatility.
C. Download and store news articles and stock price information in a vector store. Use a RAG architecture to retrieve and generate at runtime.
D. Create an agent with tools for SQL querying of Delta tables and web searching, provide retrieved values to an LLM for generation of response.


Answer : D

Question 10
A Generative AI Engineer is designing a chatbot for a gaming company that aims to engage users on its platform while its users play online video games.
Which metric would help them increase user engagement and retention for their platform?

A. Randomness
B. Diversity of responses
C. Lack of relevance
D. Repetition of responses


Answer : B

Question 11
A company has a typical RAG-enabled, customer-facing chatbot on its website.
image9
Select the correct sequence of components a user's questions will go through before the final output is returned. Use the diagram above for reference.

A. 1.embedding model, 2.vector search, 3.context-augmented prompt, 4.response-generating LLM
B. 1.context-augmented prompt, 2.vector search, 3.embedding model, 4.response-generating LLM
C. 1.response-generating LLM, 2.vector search, 3.context-augmented prompt, 4.embedding model
D. 1.response-generating LLM, 2.context-augmented prompt, 3.vector search, 4.embedding model


Answer : A

Question 12
A team wants to serve a code generation model as an assistant for their software developers. It should support multiple programming languages. Quality is the primary objective.
Which of the Databricks Foundation Model APIs, or models available in the Marketplace, would be the best fit?

A. Llama2-70b
B. BGE-large
C. MPT-7b
D. CodeLlama-34B


Answer : D

Question 13
A Generative AI Engineer is building a RAG application that will rely on context retrieved from source documents that are currently in PDF format. These PDFs can contain both text and images. They want to develop a solution using the least amount of lines of code.
Which Python package should be used to extract the text from the source documents?

A. flask
B. beautifulsoup
C. unstructured
D. numpy


Answer : C

Question 14
A Generative AI Engineer received the following business requirements for an external chatbot.
The chatbot needs to know what types of questions the user asks and routes to appropriate models to answer the questions. For example, the user might ask about upcoming event details. Another user might ask about purchasing tickets for a particular event.
What is an ideal workflow for such a chatbot?

A. The chatbot should only look at previous event information
B. There should be two different chatbots handling different types of user queries.
C. The chatbot should be implemented as a multi-step LLM workflow. First, identify the type of question asked, then route the question to the appropriate model. If it’s an upcoming event question, send the query to a text-to-SQL model. If it’s about ticket purchasing, the customer should be redirected to a payment platform.
D. The chatbot should only process payments


Answer : C

Question 15
A Generative Al Engineer is tasked with developing an application that is based on an open source large language model (LLM). They need a foundation LLM with a large context window.
Which model fits this need?

A. DistilBERT
B. MPT-30B
C. Llama2-70B
D. DBRX


Answer : D

Page:    1 / 8   
Exam contains 109 questions
Talk to us!
Have any questions or issues ? Please dont hesitate to contact us

support@certlibrary.com

Certlibrary.com is owned by MBS Tech Limited: Room 1905 Nam Wo Hong Building, 148 Wing Lok Street, Sheung Wan, Hong Kong. Company registration number: 2310926
Certlibrary doesn't offer Real Microsoft Exam Questions. Certlibrary Materials do not contain actual questions and answers from Cisco's Certification Exams.
CFA Institute does not endorse, promote or warrant the accuracy or quality of Certlibrary. CFA® and Chartered Financial Analyst® are registered trademarks owned by CFA Institute.
Terms & Conditions | Privacy Policy | Amazon Exams | Cisco Exams | CompTIA Exams | Databricks Exams | Fortinet Exams | Google Exams | Microsoft Exams | VMware Exams