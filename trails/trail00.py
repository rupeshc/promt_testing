from flask import Flask, request, render_template_string
import requests
import time
import pandas as pd

app = Flask(__name__)

MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"


PROMPT_STRATEGIES = {

"Beginner":{
"desc":"Explains topic in simple beginner friendly language.",
"template":"Explain {q} in simple beginner friendly language."
},

"Technical":{
"desc":"Provides deeper technical explanation.",
"template":"Provide a detailed technical explanation of {q}."
},

"Step-by-Step":{
"desc":"Explains concept in sequential steps.",
"template":"Explain {q} step by step."
},

"Example-Based":{
"desc":"Uses real world examples.",
"template":"Explain {q} with real world examples."
},

"Analytical":{
"desc":"Focuses on conceptual reasoning.",
"template":"Provide an analytical explanation of {q}."
},

"Research":{
"desc":"Academic style explanation.",
"template":"Explain {q} in research style."
},

"Comparison":{
"desc":"Explains concept through comparison.",
"template":"Explain {q} by comparing related ideas."
}

}


METRIC_INFO = {

"Relevance":"Measures how closely the generated answer matches the user query intent. Higher score means the response directly answers the question.",

"Completeness":"Measures how much of the topic is covered in the response. Higher score indicates broader and more detailed coverage.",

"Coherence":"Evaluates logical flow and readability of the explanation. Higher score means the answer is easier to follow.",

"Depth":"Measures conceptual richness and diversity of ideas. Higher score indicates deeper explanation.",

"Fluency":"Evaluates language quality and readability. Higher score indicates natural language usage.",

"Latency":"Measures response generation speed. Faster responses receive higher scores."

}



def ollama_generate(prompt):

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict":200}
    }

    try:

        r = requests.post(OLLAMA_URL,json=payload)
        data = r.json()

        if "response" not in data:
            return "⚠ Model did not return response."

        return data["response"]

    except:
        return "⚠ Ollama server not running."



def evaluate(query,answer,latency):

    q=set(query.lower().split())
    a=set(answer.lower().split())

    relevance=len(q.intersection(a))/max(len(q),1)

    completeness=min(len(answer)/400,1)

    sentences=answer.count(".")+1
    coherence=min(sentences/8,1)

    vocab=len(a)/max(len(answer.split()),1)
    depth=min(vocab*2,1)

    fluency=min(len(answer.split())/150,1)

    latency_score=max(0,1-(latency/10))

    return{
        "Relevance":round(relevance,2),
        "Completeness":round(completeness,2),
        "Coherence":round(coherence,2),
        "Depth":round(depth,2),
        "Fluency":round(fluency,2),
        "Latency":round(latency_score,2)
    }



def metric_explanations(scoreA,scoreB):

    explanations=[]

    for metric in scoreA:

        a=scoreA[metric]
        b=scoreB[metric]

        if a>b:
            better="Prompt A performs better"
        elif b>a:
            better="Prompt B performs better"
        else:
            better="Both prompts perform equally"

        text=f"""
<h3>{metric}</h3>

<b>Definition</b><br>
{METRIC_INFO[metric]}<br><br>

<b>Score Analysis</b><br>
Prompt A Score : {a}<br>
Prompt B Score : {b}<br><br>

{better} for this metric because the generated response better satisfies the evaluation criteria.
<br><br>
"""

        explanations.append(text)

    return explanations



def judge_answers(query,ansA,ansB):

    judge_prompt=f"""
You are evaluating two AI responses.

User Question:
{query}

Response A:
{ansA}

Response B:
{ansB}

Compare both responses in terms of:

1 relevance
2 clarity
3 explanation depth
4 completeness

Then determine which response is overall better.

Give:

Winner Summary

Why the winning response is better

3 improvements for the weaker response

Use bullet points.
"""

    return ollama_generate(judge_prompt)



@app.route("/",methods=["GET","POST"])
def home():

    result=None

    if request.method=="POST":

        mode=request.form.get("mode")
        query=request.form.get("query")

        if mode=="manual":

            promptA=request.form.get("promptA")
            promptB=request.form.get("promptB")

        else:

            styleA=request.form.get("styleA")
            styleB=request.form.get("styleB")

            promptA=PROMPT_STRATEGIES[styleA]["template"].format(q=query)
            promptB=PROMPT_STRATEGIES[styleB]["template"].format(q=query)



        start=time.time()
        ansA=ollama_generate(promptA)
        timeA=round(time.time()-start,2)

        start=time.time()
        ansB=ollama_generate(promptB)
        timeB=round(time.time()-start,2)



        scoreA=evaluate(query,ansA,timeA)
        scoreB=evaluate(query,ansB,timeB)



        avgA=sum(scoreA.values())/len(scoreA)
        avgB=sum(scoreB.values())/len(scoreB)

        winner="Prompt A" if avgA>avgB else "Prompt B"



        explanations=metric_explanations(scoreA,scoreB)

        judge=judge_answers(query,ansA,ansB)



        df=pd.DataFrame({
            "Metric":scoreA.keys(),
            "Prompt A":scoreA.values(),
            "Prompt B":scoreB.values()
        })



        result={
            "promptA":promptA,
            "promptB":promptB,
            "ansA":ansA,
            "ansB":ansB,
            "timeA":timeA,
            "timeB":timeB,
            "table":df.to_html(index=False),
            "avgA":round(avgA,2),
            "avgB":round(avgB,2),
            "winner":winner,
            "explanations":explanations,
            "judge":judge
        }



    html="""

<html>

<head>

<style>

body{
background:#121212;
color:white;
font-family:Arial;
margin:40px;
}

h2{
margin-top:30px;
}

h3{
margin-top:20px;
color:#4CAF50;
}

input,textarea,select{
width:100%;
padding:10px;
margin-top:10px;
margin-bottom:20px;
background:#1e1e1e;
color:white;
border:1px solid #444;
}

button{
padding:12px;
background:#4CAF50;
border:none;
color:white;
cursor:pointer;
}

.box{
background:#1e1e1e;
padding:25px;
margin-top:25px;
border-radius:10px;
line-height:1.6;
}

table{
width:100%;
border-collapse:collapse;
}

td,th{
border:1px solid #444;
padding:8px;
}

</style>

<script>

function toggleMode(){

var mode=document.getElementById("mode").value

if(mode=="manual"){
document.getElementById("manual").style.display="block"
document.getElementById("auto").style.display="none"
}
else{
document.getElementById("manual").style.display="none"
document.getElementById("auto").style.display="block"
}

}

</script>

</head>

<body>

<h1>LLM Prompt A/B Testing Framework</h1>

<form method="post">

Mode
<select id="mode" name="mode" onchange="toggleMode()">

<option value="manual">Manual Prompt Mode</option>

<option value="auto">Prompt Strategy Mode</option>

</select>


Query
<input name="query" required>


<div id="manual">

Prompt A
<textarea name="promptA"></textarea>

Prompt B
<textarea name="promptB"></textarea>

</div>



<div id="auto" style="display:none">

Strategy A
<select name="styleA">
"""

    for k in PROMPT_STRATEGIES:
        html+=f"<option>{k}</option>"

    html+="""
</select>

Strategy B
<select name="styleB">
"""

    for k in PROMPT_STRATEGIES:
        html+=f"<option>{k}</option>"

    html+="""

</select>

</div>


<button type="submit">Run Experiment</button>

</form>

"""



    if result:

        html+=f"""

<div class="box">

<h2>Prompt A</h2>

{result["promptA"]}

</div>



<div class="box">

<h2>Prompt B</h2>

{result["promptB"]}

</div>



<div class="box">

<h2>Answer A</h2>

{result["ansA"]}

<p>Latency : {result["timeA"]} sec</p>

</div>



<div class="box">

<h2>Answer B</h2>

{result["ansB"]}

<p>Latency : {result["timeB"]} sec</p>

</div>



<div class="box">

<h2>Metric Analysis</h2>

"""

        for e in result["explanations"]:
            html+=e



        html+=f"""

</div>



<div class="box">

<h2>Evaluation Metrics Table</h2>

{result["table"]}

</div>



<div class="box">

<h2>Average Score</h2>

Prompt A : {result["avgA"]}<br>

Prompt B : {result["avgB"]}

</div>



<div class="box">

<h2>Winner</h2>

{result["winner"]}

</div>



<div class="box">

<h2>LLM Evaluation Summary</h2>

{result["judge"]}

</div>

"""



    html+="</body></html>"



    return render_template_string(html)



if __name__=="__main__":

    app.run(debug=True)