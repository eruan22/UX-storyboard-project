from flask import Flask, render_template, request, session, redirect, url_for
import json

from models.schemas import StoryboardInput, StoryboardOutput, CriticOutput, PanelCritique, DesignOutput, DesignRecommendation
from agents.journey_agent import run_journey_agent
from agents.ux_critic_agent import run_critic_agent
from agents.design_agent import run_design_agent
from utils.chroma_setup import basic_retrieve, get_vectorstore

# ── LLM setup ─────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, '../inclass')
from llm_utils import get_chat_model

USE_REMOTE = True
chat_model = get_chat_model(use_remote=USE_REMOTE, model="qwen3.5:4b")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "ux-storyboard-secret"


# ── helpers ───────────────────────────────────────────────────────────────────
def panels_to_dicts(panels):
    return [p.model_dump() for p in panels]

def critiques_to_dicts(critiques):
    return [c.model_dump() for c in critiques]

def recs_to_dicts(recs):
    return [r.model_dump() for r in recs]

def session_flags():
    """Track which steps have completed data — passed to every template."""
    return {
        "has_storyboard":      bool(session.get("panels")),
        "has_critique":        bool(session.get("critiques")),
        "has_recommendations": bool(session.get("recommendations")),
    }


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **session_flags())


# ── storyboard ────────────────────────────────────────────────────────────────

@app.route("/storyboard", methods=["GET"])
def storyboard_get():
    """Pipeline button: revisit storyboard stored in session."""
    panels_data = session.get("panels")
    if not panels_data:
        return redirect(url_for("index"))
    from models.schemas import Panel
    panels = [Panel(**p) for p in panels_data]
    return render_template("storyboard.html",
                           panels=panels,
                           persona=session.get("persona", ""),
                           goal=session.get("goal", ""),
                           product=session.get("product", ""),
                           scenario=session.get("scenario", ""),
                           **session_flags())

@app.route("/generate", methods=["POST"])
def generate():
    """Run Journey Agent → store panels → show storyboard."""
    persona  = request.form["persona"]
    goal     = request.form["goal"]
    product  = request.form["product"]
    scenario = request.form["scenario"]

    user_input = StoryboardInput(persona=persona, goal=goal,
                                  product=product, scenario=scenario)
    storyboard: StoryboardOutput = run_journey_agent(user_input, chat_model)

    session["panels"]   = panels_to_dicts(storyboard.panels)
    session["persona"]  = persona
    session["goal"]     = goal
    session["product"]  = product
    session["scenario"] = scenario
    # clear downstream when regenerating
    session.pop("critiques", None)
    session.pop("recommendations", None)

    return render_template("storyboard.html",
                           panels=storyboard.panels,
                           persona=persona, goal=goal,
                           product=product, scenario=scenario,
                           **session_flags())


# ── critique ──────────────────────────────────────────────────────────────────

@app.route("/critique", methods=["GET"])
def critique_get():
    """Pipeline button: revisit critique stored in session."""
    critiques_data = session.get("critiques")
    panels_data    = session.get("panels")
    if not critiques_data or not panels_data:
        return redirect(url_for("index"))
    from models.schemas import Panel, PanelCritique
    panels    = [Panel(**p) for p in panels_data]
    critiques = [PanelCritique(**c) for c in critiques_data]
    return render_template("critique.html",
                           critiques=critiques,
                           panels=panels,
                           **session_flags())

@app.route("/critique", methods=["POST"])
def critique_post():
    """Run UX Critic Agent on stored panels."""
    panels_data = session.get("panels", [])
    from models.schemas import Panel
    panels = [Panel(**p) for p in panels_data]

    vector_store   = get_vectorstore()
    retrieved_docs = basic_retrieve(panels, vector_store, top_k=5)
    critic_output: CriticOutput = run_critic_agent(panels, retrieved_docs, chat_model)

    session["critiques"] = critiques_to_dicts(critic_output.critiques)
    session.pop("recommendations", None)

    return render_template("critique.html",
                           critiques=critic_output.critiques,
                           panels=panels,
                           **session_flags())


# ── recommendations ───────────────────────────────────────────────────────────

@app.route("/recommendations", methods=["GET"])
def recommendations_get():
    """Pipeline button: revisit recommendations stored in session."""
    recs_data   = session.get("recommendations")
    panels_data = session.get("panels")
    if not recs_data or not panels_data:
        return redirect(url_for("index"))
    from models.schemas import Panel, DesignRecommendation
    panels          = [Panel(**p) for p in panels_data]
    recommendations = [DesignRecommendation(**r) for r in recs_data]
    return render_template("recommendations.html",
                           recommendations=recommendations,
                           panels=panels,
                           **session_flags())

@app.route("/recommend", methods=["POST"])
def recommend():
    """Run Design Agent on stored panels + critiques."""
    from models.schemas import Panel, PanelCritique, CriticOutput

    panels_data    = session.get("panels", [])
    critiques_data = session.get("critiques", [])

    panels        = [Panel(**p) for p in panels_data]
    critiques     = [PanelCritique(**c) for c in critiques_data]
    critic_output = CriticOutput(critiques=critiques)

    design_output: DesignOutput = run_design_agent(panels, critic_output, chat_model)

    session["recommendations"] = recs_to_dicts(design_output.recommendations)

    return render_template("recommendations.html",
                           recommendations=design_output.recommendations,
                           panels=panels,
                           **session_flags())


if __name__ == "__main__":
    app.run(debug=True)