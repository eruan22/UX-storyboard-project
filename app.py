from flask import Flask, render_template, request, session, redirect, url_for
import json

from models.schemas import StoryboardInput, StoryboardOutput, CriticOutput, PanelCritique, DesignOutput, DesignRecommendation
from agents.journey_agent import run_journey_agent
from agents.ux_critic_agent import run_critic_agent
from agents.design_agent import run_design_agent
from utils.chroma_setup import basic_retrieve, get_vectorstore

# ── LLM setup (adjust use_remote / model to match your environment) ──────────
import sys
sys.path.insert(0, '../inclass')
from llm_utils import get_chat_model

USE_REMOTE = True
chat_model = get_chat_model(use_remote=USE_REMOTE, model="qwen3.5:4b")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "ux-storyboard-secret"   # needed for session storage


# ── helpers ───────────────────────────────────────────────────────────────────
def panels_to_dicts(panels):
    return [p.model_dump() for p in panels]

def critiques_to_dicts(critiques):
    return [c.model_dump() for c in critiques]

def recs_to_dicts(recs):
    return [r.model_dump() for r in recs]


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Landing page — input form."""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Run Journey Agent → store panels in session → show storyboard."""
    persona  = request.form["persona"]
    goal     = request.form["goal"]
    product  = request.form["product"]
    scenario = request.form["scenario"]

    user_input = StoryboardInput(persona=persona, goal=goal,
                                  product=product, scenario=scenario)

    storyboard: StoryboardOutput = run_journey_agent(user_input, chat_model)

    # Persist for next steps
    session["panels"]   = panels_to_dicts(storyboard.panels)
    session["persona"]  = persona
    session["goal"]     = goal
    session["product"]  = product
    session["scenario"] = scenario

    return render_template("storyboard.html", panels=storyboard.panels,
                           persona=persona, goal=goal,
                           product=product, scenario=scenario)


@app.route("/critique", methods=["POST"])
def critique():
    """Run UX Critic Agent on stored panels → show critiques."""
    panels_data = session.get("panels", [])

    # Reconstruct Panel objects
    from models.schemas import Panel
    panels = [Panel(**p) for p in panels_data]

    # RAG retrieval
    vector_store   = get_vectorstore()
    retrieved_docs = basic_retrieve(panels, vector_store, top_k=5)

    critic_output: CriticOutput = run_critic_agent(panels, retrieved_docs, chat_model)

    session["critiques"] = critiques_to_dicts(critic_output.critiques)

    return render_template("critique.html",
                           critiques=critic_output.critiques,
                           panels=panels)


@app.route("/recommend", methods=["POST"])
def recommend():
    """Run Design Agent on stored panels + critiques → show recommendations."""
    from models.schemas import Panel, PanelCritique, CriticOutput

    panels_data   = session.get("panels", [])
    critiques_data = session.get("critiques", [])

    panels   = [Panel(**p) for p in panels_data]
    critiques = [PanelCritique(**c) for c in critiques_data]
    critic_output = CriticOutput(critiques=critiques)

    design_output: DesignOutput = run_design_agent(panels, critic_output, chat_model)

    return render_template("recommendations.html",
                           recommendations=design_output.recommendations,
                           panels=panels)


if __name__ == "__main__":
    app.run(debug=True)

#### OLD ####
# from flask import Flask, render_template, request, jsonify, Response, stream_with_context
# import sys
# from langchain_ollama import ChatOllama
# from models.schemas import StoryboardInput
# from agents.journey_agent import run_journey_agent
# from agents.ux_critic_agent import run_critic_agent
# from agents.design_agent import run_design_agent
# from utils.chroma_setup import basic_retrieve, get_vectorstore

# app = Flask(__name__)

# # LLM SET UP
# USE_REMOTE = False
# sys.path.insert(0, '../inclass')
# from llm_utils import get_llm, get_chat_model

# model = get_llm(use_remote=USE_REMOTE, model="qwen3.5:4b")
# chat_model = get_chat_model(use_remote=USE_REMOTE, model="qwen3.5:4b")

# # initialize chat model once at startup
# # chat_model = ChatOllama(model="qwen3:4b", temperature=0.7)
# # initialize LLM
# llm = ChatOllama(
#     model="qwen3.5:4b",
#     temperature=0.7,
#     #base_url=OLLAMA_BASE_URL
# )

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/generate", methods=["POST"])
# def generate():
#     data = request.json

#     # step 1: collect input
#     user_input = StoryboardInput(
#         persona=data["persona"],
#         goal=data["goal"],
#         product=data["product"],
#         scenario=data["scenario"]
#     )

#     # step 2: journey agent
#     storyboard_output = run_journey_agent(user_input, chat_model)

#     # step 3: retrieve docs
#     vector_store = get_vectorstore()
#     retrieved_docs = basic_retrieve(storyboard_output.panels, vector_store, top_k=5)

#     # step 4: critic agent
#     critic_output = run_critic_agent(storyboard_output.panels, retrieved_docs, chat_model)

#     # step 5: design agent
#     design_output = run_design_agent(storyboard_output.panels, critic_output, chat_model)

#     # return all results as JSON
#     return jsonify({
#         "panels": [p.dict() for p in storyboard_output.panels],
#         "critiques": [c.dict() for c in critic_output.critiques],
#         "recommendations": [r.dict() for r in design_output.recommendations]
#     })

# if __name__ == "__main__":
#     app.run(debug=False)