"""FastAPI server wrapping the SkyJet agent (Databricks Apps entrypoint)."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import mlflow
from agent import handle_request


@asynccontextmanager
async def lifespan(app):
    # Startup: bind traces to the injected experiment. Observability must
    # never take the app down: if MLflow cannot reach the experiment
    # (credentials, permissions) we log the failure and keep serving.
    try:
        os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
        mlflow.openai.autolog()   # the official 5A tracing pattern
    except Exception as e:
        print(f"MLflow setup failed (traces will not upload): {e}")
    yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    # SYNC on purpose: FastAPI runs sync endpoints in its threadpool, which
    # lets handle_request call asyncio.run() for the agent loop. An async
    # endpoint would already be inside the event loop and crash with
    # "asyncio.run() cannot be called from a running event loop".
    return {"response": handle_request(request.message)}


# The SkyJet Assist chat UI: browser users authenticate via workspace SSO, so
# this is both the human smoke test and the demo surface for the deployed
# agent. The fetch to /chat carries the SSO session cookie automatically.
# Deliberately ONE self-contained HTML string - no build step, no framework.
# A production app would serve a real frontend bundle instead.
CHAT_UI = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SkyJet Assist</title>
<style>
  :root{
    --bg:#170a2b; --pink:#ff5ec7; --violet:#a78bfa; --cyan:#5eead4;
    --text:#e9e4f5; --dim:#a99fc4;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;
       background:radial-gradient(1200px 600px at 80% -10%,#2b1650 0%,var(--bg) 55%);
       color:var(--text);height:100vh;overflow:hidden}
  .app{display:flex;height:100vh}

  /* ---- sidebar ---- */
  aside{width:230px;min-width:230px;background:linear-gradient(180deg,#16092b,#0f061d);
        border-right:1px solid #3b2260;padding:26px 18px;display:flex;flex-direction:column}
  .side-title{color:var(--pink);font-weight:600;font-size:15px;margin:12px 0 10px;
              text-shadow:0 0 12px rgba(255,94,199,.45)}
  #history,#policies{list-style:none;display:flex;flex-direction:column;gap:8px;
                     max-height:24vh;overflow-y:auto}
  #history li{font-size:12.5px;color:var(--dim);padding:7px 10px;border:1px solid #35205c;
              border-radius:8px;cursor:pointer;white-space:nowrap;overflow:hidden;
              text-overflow:ellipsis}
  #history li:hover{border-color:var(--violet);color:var(--text)}
  #policies li{font-size:12.5px;color:var(--cyan);padding:4px 2px;
               text-shadow:0 0 10px rgba(94,234,212,.4)}
  .empty{color:#6b5a92 !important;font-style:italic}
  .plane-big{margin-top:auto;align-self:center;
             filter:drop-shadow(0 0 14px rgba(94,234,212,.55))
                    drop-shadow(0 0 22px rgba(255,94,199,.35))}

  /* ---- main ---- */
  main{flex:1;display:flex;flex-direction:column;padding:30px 44px 26px;
       max-width:960px;margin:0 auto;width:100%}
  h1{font-size:44px;font-weight:800;letter-spacing:.5px;
     background:linear-gradient(90deg,var(--pink),var(--violet) 55%,#7dd3fc);
     -webkit-background-clip:text;background-clip:text;color:transparent;
     filter:drop-shadow(0 0 18px rgba(167,139,250,.35))}
  .sub{color:var(--dim);margin:8px 0 22px;font-size:14.5px;max-width:520px;line-height:1.5}
  #chat{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:18px;
        padding:6px 4px 18px}

  .row{display:flex;align-items:flex-end;gap:10px}
  .row.user{justify-content:flex-end}
  .avatar{width:30px;height:30px;border-radius:50%;flex:none;display:flex;
          align-items:center;justify-content:center;font-size:15px;
          background:linear-gradient(135deg,var(--cyan),var(--violet));
          box-shadow:0 0 12px rgba(94,234,212,.5)}
  .bubble{max-width:72%;padding:13px 16px;border-radius:14px;font-size:14.5px;
          line-height:1.55;white-space:pre-wrap;word-wrap:break-word}
  .bubble.user{background:#2e2b36;border-bottom-right-radius:4px;color:#f2eefc}
  .bubble.bot{position:relative;background:#150a28;border:2px solid #7c6ff0;
      border-radius:4px 16px 16px 16px;
      box-shadow:0 0 16px rgba(167,139,250,.3), inset 0 0 14px rgba(124,111,240,.12)}
  .bot-icon{position:absolute;left:-36px;bottom:4px;
            filter:drop-shadow(0 0 8px rgba(94,234,212,.6))}
  .dots span{display:inline-block;width:7px;height:7px;margin-right:4px;border-radius:50%;
             background:var(--violet);animation:blink 1.2s infinite}
  .dots span:nth-child(2){animation-delay:.2s}
  .dots span:nth-child(3){animation-delay:.4s}
  @keyframes blink{0%,80%,100%{opacity:.25}40%{opacity:1}}

  /* ---- composer ---- */
  form{display:flex;gap:10px;align-items:center;background:#1c0f36;
       border:1px solid #6d3fd4;border-radius:999px;padding:8px 8px 8px 20px;
       box-shadow:0 0 18px rgba(140,82,255,.25)}
  form:focus-within{border-color:var(--pink);box-shadow:0 0 26px rgba(255,94,199,.35)}
  input{flex:1;background:transparent;border:0;outline:0;color:var(--text);font-size:15px}
  input::placeholder{color:#7c6aa8}
  button{background:linear-gradient(90deg,#d946ef,#a855f7);border:0;color:#fff;
         font-weight:600;font-size:14.5px;padding:9px 22px;border-radius:999px;
         cursor:pointer;box-shadow:0 0 14px rgba(217,70,239,.45)}
  button:hover{filter:brightness(1.15)}
  button:disabled{opacity:.55;cursor:wait}
  @media(max-width:700px){aside{display:none}h1{font-size:32px}}
</style>
</head>
<body>
<div class="app">
  <aside>
    <div class="side-title">Chat History</div>
    <ul id="history"></ul>
    <div class="side-title">Recent Policies</div>
    <ul id="policies"><li class="empty">cited docs appear here</li></ul>
    <svg class="plane-big" width="120" height="120" viewBox="0 0 24 24" fill="none">
      <defs><linearGradient id="pg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#5eead4"/><stop offset="1" stop-color="#ff5ec7"/>
      </linearGradient></defs>
      <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"
            stroke="url(#pg)" stroke-width="1.1" stroke-linejoin="round"/>
    </svg>
  </aside>
  <main>
    <h1>SkyJet Assist</h1>
    <p class="sub">Governed passenger support: flight status and policy questions, grounded in the official policy docs. The first answer can take a couple of minutes while the tools warm up.</p>
    <div id="chat"></div>
    <form onsubmit="return ask(event)">
      <input id="q" placeholder="Ask SkyJet Assist..." autocomplete="off">
      <button id="b">Ask</button>
    </form>
  </main>
</div>
<script>
const chat=document.getElementById('chat'), q=document.getElementById('q'), b=document.getElementById('b');
const planeSVG='<svg class="bot-icon" width="24" height="24" viewBox="0 0 24 24" fill="none">'
  +'<path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z" '
  +'stroke="#5eead4" stroke-width="1.1" stroke-linejoin="round"/></svg>';
function bubble(cls){const row=document.createElement('div');row.className='row '+cls;
  row.innerHTML = cls==='user'
    ? '<div class="bubble user"></div><div class="avatar">👤</div>'
    : '<div class="bubble bot">'+planeSVG+'<span></span></div>';
  chat.appendChild(row); chat.scrollTop=chat.scrollHeight;
  return row.querySelector('span') || row.querySelector('.bubble');}
function citeDocs(text){const docs=[...new Set(text.match(/doc-\\d+/g)||[])];
  if(!docs.length)return;
  const ul=document.getElementById('policies');
  const e=ul.querySelector('.empty'); if(e)e.remove();
  docs.forEach(d=>{if(![...ul.children].some(li=>li.textContent===d)){
    const li=document.createElement('li'); li.textContent=d; ul.appendChild(li);}});}
async function ask(ev){ev&&ev.preventDefault(); const text=q.value.trim(); if(!text)return false;
  q.value=''; b.disabled=true;
  bubble('user').textContent=text;
  const h=document.createElement('li'); h.textContent=text; h.title='click to ask again';
  h.onclick=()=>{q.value=text; q.focus()};
  document.getElementById('history').prepend(h);
  const t=bubble('bot');
  t.innerHTML='<span class="dots"><span></span><span></span><span></span></span> '
    +'<i style="color:#a99fc4;font-size:12.5px">thinking... (first answer can take a couple of minutes)</i>';
  try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
       body:JSON.stringify({message:text})});
    const d=await r.json(); const ans=d.response||('error: '+r.status);
    t.textContent=ans; citeDocs(ans);
  }catch(e){t.textContent='network error: '+e}
  b.disabled=false; q.focus(); return false;}
bubble('bot').textContent='Hi! I am SkyJet Assist. Ask me about flight status, baggage fees, or anything in the official passenger policies.';
</script>
</body></html>"""


@app.get("/")
async def home():
    return HTMLResponse(CHAT_UI)


@app.get("/health")
async def health():
    return {"status": "ok"}
