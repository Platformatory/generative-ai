"""SkyJet passenger-support agent: production agent logic (env-injected config)."""
import os, asyncio, logging
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors

# Route the OpenAI Agents SDK through Databricks model serving (same wiring
# as the lab notebook's Section D): without this the SDK calls api.openai.com
# directly and every agent run fails. The CLIENT itself is created per run
# in run() below - see the comment there for why.
set_default_openai_api("chat_completions")
set_trace_processors([])   # disable OpenAI trace processor; MLflow owns tracing
logging.getLogger("openai").setLevel(logging.WARNING)

UC_TOOL_CATALOG        = os.environ["UC_TOOL_CATALOG"]
UC_TOOL_SCHEMA         = os.environ["UC_TOOL_SCHEMA"]
SERVING_ENDPOINT       = os.environ["SERVING_ENDPOINT"]

ws = WorkspaceClient()

# The EXACT instructions the acceptance tests (E2/T2) were validated against.
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

# Agent wiring - three constructor options, three different jobs:
#   tools=[...]        agent-LOCAL glue (@function_tool): lives only inside
#                      this process, ungoverned and undiscoverable. We pass
#                      NONE because every shared SkyJet tool is a UC function
#                      (the Notebook 1 Part A rule: shared means UC, glue
#                      stays local - tools= is for helpers like formatting a
#                      chat-UI answer, it is not prototype-only, just local).
#   mcp_servers=[...]  governed tools, discovered at RUNTIME from the managed
#                      MCP server over the UC schema. Left EMPTY here on
#                      purpose: an MCP server is an async context manager
#                      that must live inside the request's event loop, so
#                      run() below opens it per request and attaches it.
#                      Empty-at-construction is a placeholder, NOT "no tools".
#   handoffs=[...]     supervisor-worker routing: the supervisor itself has
#                      NO tools and answers nothing, it only delegates to
#                      the specialists (the 5A Demo 6 pattern).
policy_agent = Agent(name="PolicyAgent", model=SERVING_ENDPOINT, instructions=POLICY_AGENT_INSTRUCTIONS, mcp_servers=[])
flight_agent = Agent(name="FlightAgent", model=SERVING_ENDPOINT, instructions=FLIGHT_AGENT_INSTRUCTIONS, mcp_servers=[])
supervisor = Agent(name="SkyJetSupervisor", model=SERVING_ENDPOINT, instructions=SUPERVISOR_INSTRUCTIONS, handoffs=[policy_agent, flight_agent])


async def run(user_message: str) -> str:
    # A fresh OpenAI client per run: httpx binds its connection pool to the
    # running event loop, and every /chat executes asyncio.run() (a brand-new
    # loop). A module-level client would try to reuse pooled connections
    # from a CLOSED loop and crash with "Event loop is closed" (the live
    # app taught us this on its second request).
    set_default_openai_client(AsyncDatabricksOpenAI())
    # ONE MCP server, opened per request, serves BOTH specialists.
    # from_uc_function() is the UC-functions MCP - the same managed server
    # Notebook 1 hit at /api/2.0/mcp/functions/<catalog>/<schema>. It exposes
    # every UC function in the schema: the SQL flight tools, the Python
    # fee/masking tools, AND search_policies, the RAG retrieval tool (a SQL
    # function wrapping the vector_search() built-in over the policies index).
    # Note there is deliberately NO separate index MCP here: retrieval is
    # already a governed UC function, so one namespace, one server, and one
    # set of grants cover RAG and the operational tools alike, and the app
    # never needs direct vector-index credentials.
    # Why attach per request instead of at construction? Two reasons:
    #   1. lifecycle: the server is an async context manager, and each /chat
    #      runs in a fresh event loop (see the client comment above)
    #   2. runtime discovery: a UC function registered today appears in the
    #      next request's tools/list - new tools with zero redeploy
    # How each specialist knows which tools to call: both workers see all 5
    # tools (same server attached to both), and their INSTRUCTIONS name which
    # ones to use (PolicyAgent: search_policies, mask_contact_info; FlightAgent:
    # the three flight tools). For hard isolation you would instead attach
    # DIFFERENT per-schema servers to different agents.
    async with McpServer.from_uc_function(
        catalog=UC_TOOL_CATALOG, schema=UC_TOOL_SCHEMA,
        workspace_client=ws, name="skyjet-uc-tools", timeout=120,
    ) as uc_server:
        policy_agent.mcp_servers = [uc_server]
        flight_agent.mcp_servers = [uc_server]
        result = await Runner.run(supervisor, user_message)
        return result.final_output


def handle_request(user_message: str, max_attempts: int = 2) -> str:
    # Same bounded retry as the notebook entrypoint (Section D): served Llama
    # models occasionally emit a malformed tool-call, which the endpoint
    # rejects with a 400 ("did not respect the required format"); Databricks'
    # guidance is to retry once.
    from openai import BadRequestError
    import mlflow
    with mlflow.start_span(name="skyjet_chat"):
        for attempt in range(1, max_attempts + 1):
            try:
                return asyncio.run(run(user_message))
            except BadRequestError as e:
                if attempt == max_attempts or "required format" not in str(e):
                    raise
