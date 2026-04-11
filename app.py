from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import sys
from langchain_ollama import ChatOllama
from models.schemas import StoryboardInput
from agents.journey_agent import run_journey_agent
from agents.ux_critic_agent import run_critic_agent
from agents.design_agent import run_design_agent
from utils.chroma_setup import basic_retrieve

app = Flask(__name__)

# LLM SET UP
USE_REMOTE = False
sys.path.insert(0, '../inclass')
from llm_utils import get_llm, get_chat_model

model = get_llm(use_remote=USE_REMOTE, model="qwen3.5:4b")
chat_model = get_chat_model(use_remote=USE_REMOTE, model="qwen3.5:4b")

# initialize chat model once at startup
# chat_model = ChatOllama(model="qwen3:4b", temperature=0.7)
# initialize LLM
llm = ChatOllama(
    model="qwen3.5:4b",
    temperature=0.7,
    #base_url=OLLAMA_BASE_URL
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json

    # step 1: collect input
    user_input = StoryboardInput(
        persona=data["persona"],
        goal=data["goal"],
        product=data["product"],
        scenario=data["scenario"]
    )

    # step 2: journey agent
    storyboard_output = run_journey_agent(user_input, chat_model)

    # step 3: retrieve docs
    retrieved_docs = basic_retrieve(storyboard_output.panels, top_k=5)

    # step 4: critic agent
    critic_output = run_critic_agent(storyboard_output.panels, retrieved_docs, chat_model)

    # step 5: design agent
    design_output = run_design_agent(storyboard_output.panels, critic_output, chat_model)

    # return all results as JSON
    return jsonify({
        "panels": [p.dict() for p in storyboard_output.panels],
        "critiques": [c.dict() for c in critic_output.critiques],
        "recommendations": [r.dict() for r in design_output.recommendations]
    })

if __name__ == "__main__":
    app.run(debug=False)