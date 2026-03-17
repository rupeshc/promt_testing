
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import requests
import time
import json

app = FastAPI()

MODEL="phi3:mini"

PROMPTS={
"Simple":"Explain {query} in simple terms.",
"Technical":"Provide a technical explanation of {query}.",
"Example":"Explain {query} with real world examples.",
"Step":"Explain {query} step by step.",
"Comparison":"Explain {query} and compare it with other technologies."
}

def clamp(v):
    return round(max(0,min(v,1)),2)

def generate(prompt):

    start=time.time()

    r=requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":MODEL,
            "prompt":prompt,
            "stream":False
        }
    )

    answer=r.json()["response"]

    latency=time.time()-start

    return answer,latency

def evaluate(query,answer,latency):

    query_words=query.lower().split()
    ans_words=answer.lower().split()

    query_set=set(query_words)
    ans_set=set(ans_words)

    overlap=len(query_set.intersection(ans_set))

    sentences=answer.count(".")+answer.count("?")+answer.count("!")

    unique_ratio=len(ans_set)/max(len(ans_words),1)

    avg_sentence=len(ans_words)/max(sentences,1)

    relevance=clamp(overlap/max(len(query_set),1))
    completeness=clamp(len(ans_words)/150)
    coherence=clamp(sentences/10)
    depth=clamp(unique_ratio*2)
    fluency=clamp(avg_sentence/18)
    latency_score=clamp(1-latency/60)

    metrics={
    "Relevance":relevance,
    "Completeness":completeness,
    "Coherence":coherence,
    "Depth":depth,
    "Fluency":fluency,
    "Latency":latency_score
    }

    calculations={
    "query_words":len(query_words),
    "answer_words":len(ans_words),
    "sentences":sentences,
    "keyword_overlap":overlap,
    "unique_words":len(ans_set),
    "unique_ratio":round(unique_ratio,2),
    "avg_sentence_length":round(avg_sentence,2),
    "latency":round(latency,2)
    }

    return metrics,calculations

def avg_score(m):
    return round(sum(m.values())/len(m),2)

def judge(query,a,b):

    prompt=f"""
Compare two answers.

Question:
{query}

Answer A:
{a}

Answer B:
{b}

Explain clearly in 6 points which is better and why.
"""

    r=requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":MODEL,
            "prompt":prompt,
            "stream":False
        }
    )

    return r.json()["response"]

@app.get("/",response_class=HTMLResponse)
def home():

    options=""

    for p in PROMPTS:
        options+=f"<option value='{p}'>{p}</option>"

    return f"""

<html>

<head>

<style>

body{{background:#0f172a;color:white;font-family:Arial;padding:40px}}

.card{{background:#1e293b;padding:20px;border-radius:10px;margin-bottom:20px}}

input,textarea,select{{width:70%;padding:8px;background:#020617;color:white;border:1px solid #475569}}

button{{background:#22c55e;padding:10px;border:none;color:white}}

</style>

<script>

function switchMode(){{

mode=document.getElementById("mode").value

if(mode=="manual"){{

document.getElementById("manual").style.display="block"
document.getElementById("auto").style.display="none"

}}

else{{

document.getElementById("manual").style.display="none"
document.getElementById("auto").style.display="block"

}}

}}

</script>

</head>

<body>

<h1>LLM Prompt A/B Testing Framework</h1>

<form method="post" action="/run">

<div class="card">

Query<br>
<input name="query">

<br><br>

Mode<br>

<select id="mode" name="mode" onchange="switchMode()">

<option value="manual">Manual Prompt</option>
<option value="auto">Automatic Prompt</option>

</select>

</div>

<div id="manual" class="card">

Prompt A<br>
<textarea name="promptA"></textarea>

<br><br>

Prompt B<br>
<textarea name="promptB"></textarea>

</div>

<div id="auto" class="card" style="display:none">

Prompt Strategy A<br>
<select name="autoA">
{options}
</select>

<br><br>

Prompt Strategy B<br>
<select name="autoB">
{options}
</select>

</div>

<button type="submit">Run Experiment</button>

</form>

</body>
</html>
"""

@app.post("/run",response_class=HTMLResponse)
def run(query:str=Form(...),
mode:str=Form(...),
promptA:str=Form(""),
promptB:str=Form(""),
autoA:str=Form("Simple"),
autoB:str=Form("Technical")):

    auto_prompts=""

    if mode=="auto":

        promptA=PROMPTS[autoA].format(query=query)
        promptB=PROMPTS[autoB].format(query=query)

        auto_prompts=f"""
<h2>Auto Generated Prompts</h2>

Prompt A Strategy: {autoA}<br>
<b>{promptA}</b>

<br><br>

Prompt B Strategy: {autoB}<br>
<b>{promptB}</b>
"""

    ansA,latA=generate(promptA)
    ansB,latB=generate(promptB)

    mA,cA=evaluate(query,ansA,latA)
    mB,cB=evaluate(query,ansB,latB)

    avgA=avg_score(mA)
    avgB=avg_score(mB)

    winner="Prompt A" if avgA>avgB else "Prompt B"

    judge_text=judge(query,ansA,ansB)

    labels=json.dumps(list(mA.keys()))
    dataA=json.dumps(list(mA.values()))
    dataB=json.dumps(list(mB.values()))

    return f"""

<html>

<head>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

body{{background:#0f172a;color:white;font-family:Arial;padding:40px}}

.chartbox{{width:320px;height:260px}}

</style>

</head>

<body>

{auto_prompts}

<h2>Answer A</h2>

<p>{ansA}</p>

Runtime: {round(latA,2)} seconds

<h2>Answer B</h2>

<p>{ansB}</p>

Runtime: {round(latB,2)} seconds

<h2>Evaluation Metrics</h2>

<h3>Relevance</h3>

Definition: Measures how closely the answer matches the user query keywords.

Formula  
Relevance = keyword_overlap / total_query_keywords

Calculation  
Prompt A = {cA["keyword_overlap"]} / {cA["query_words"]} = {mA["Relevance"]}  
Prompt B = {cB["keyword_overlap"]} / {cB["query_words"]} = {mB["Relevance"]}

Explanation  
The system checks how many keywords from the query appear in the answer.

<hr>

<h3>Completeness</h3>

Definition: Measures how detailed the response is.

Formula  
Completeness = answer_words / 150

Calculation  
Prompt A = {cA["answer_words"]} / 150 = {mA["Completeness"]}  
Prompt B = {cB["answer_words"]} / 150 = {mB["Completeness"]}

Explanation  
Longer answers usually cover more concepts, increasing completeness.

<hr>

<h3>Coherence</h3>

Definition: Measures structural clarity.

Formula  
Coherence = sentences / 10

Calculation  
Prompt A = {cA["sentences"]} / 10 = {mA["Coherence"]}  
Prompt B = {cB["sentences"]} / 10 = {mB["Coherence"]}

Explanation  
Well structured answers typically contain multiple clear sentences.

<hr>

<h3>Depth</h3>

Definition: Measures vocabulary diversity.

Formula  
Depth = unique_ratio × 2

Calculation  
Prompt A = {cA["unique_ratio"]} × 2 = {mA["Depth"]}  
Prompt B = {cB["unique_ratio"]} × 2 = {mB["Depth"]}

Explanation  
Higher vocabulary diversity indicates deeper explanations.

<hr>

<h3>Fluency</h3>

Definition: Measures readability.

Formula  
Fluency = avg_sentence_length / 18

Calculation  
Prompt A = {cA["avg_sentence_length"]} / 18 = {mA["Fluency"]}  
Prompt B = {cB["avg_sentence_length"]} / 18 = {mB["Fluency"]}

Explanation  
Balanced sentence length improves readability.

<hr>

<h3>Latency</h3>

Definition: Measures response speed.

Formula  
Latency Score = 1 − latency/60

Calculation  
Prompt A = 1 − {cA["latency"]}/60 = {mA["Latency"]}  
Prompt B = 1 − {cB["latency"]}/60 = {mB["Latency"]}

Explanation  
Faster responses produce higher scores.

<hr>

<h2>Chart Analysis</h2>

<select id="chartMode" onchange="updateChart()">

<option value="both">Both</option>
<option value="A">Prompt A</option>
<option value="B">Prompt B</option>

</select>

<br><br>

<div class="chartbox">
<canvas id="radar"></canvas>
</div>

<br>

<div class="chartbox">
<canvas id="line"></canvas>
</div>

<script>

const labels={labels}

const dataA={dataA}
const dataB={dataB}

const radar=new Chart(document.getElementById('radar'),{{
type:'radar',
data:{{labels:labels,datasets:[
{{label:'Prompt A',data:dataA,borderColor:'cyan'}},
{{label:'Prompt B',data:dataB,borderColor:'orange'}}
]}}
}})

const line=new Chart(document.getElementById('line'),{{
type:'line',
data:{{labels:labels,datasets:[
{{label:'Prompt A',data:dataA,borderColor:'cyan'}},
{{label:'Prompt B',data:dataB,borderColor:'orange'}}
]}}
}})

function updateChart(){{
mode=document.getElementById("chartMode").value

radar.data.datasets[0].hidden=(mode=="B")
radar.data.datasets[1].hidden=(mode=="A")

line.data.datasets[0].hidden=(mode=="B")
line.data.datasets[1].hidden=(mode=="A")

radar.update()
line.update()
}}

</script>

<h2>Final Result</h2>

Average A: {avgA}<br>
Average B: {avgB}<br>

<b>Winner: {winner}</b>

<h2>LLM Judge Explanation</h2>

<pre>{judge_text}</pre>

</body>
</html>
"""

