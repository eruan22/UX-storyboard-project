import sys
import json

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser


from models.schemas import StoryboardInput, StoryboardOutput, Panel, CriticOutput, PanelCritique, DesignOutput, DesignRecommendation
from agents.journey_agent import run_journey_agent

from agents.ux_critic_agent import run_critic_agent
from agents.design_agent import run_design_agent
from utils.chroma_setup import basic_retrieve

# LLM SET UP
USE_REMOTE = False
sys.path.insert(0, '../inclass')
from llm_utils import get_llm, get_chat_model

model = get_llm(use_remote=USE_REMOTE, model="qwen3:4b")
chat_model = get_chat_model(use_remote=USE_REMOTE, model="qwen3:4b")

# initialize LLM
llm = ChatOllama(
    model="qwen-3:4b",
    temperature=0.7,
    #base_url=OLLAMA_BASE_URL
)

# INPUT COLLECTION
# function to collect user input for the Journey Agent
def collect_input() -> StoryboardInput:
    """Collect user input for the storyboard agent."""
    persona = input("Enter the user persona (Who is the user? ex: Student): ").strip()
    goal = input("Enter the user goal (What are they trying to do? ex: book a room): ").strip()
    product = input("Enter the product (What app are they using? ex: room booker): ").strip()
    scenario = input("Enter the scenario (What's the situation/context? ex: Needs to study for an exam): ").strip()
    return StoryboardInput(persona=persona, goal=goal, product=product, scenario=scenario)

# DISPLAY
# function to display the output
def display_storyboard(output: StoryboardOutput):
    print("Generated output:")
    for p in output.panels:
        print(f"Panel Number: {p.panel_number}")
        print(f"Action: {p.action}")
        print(f"Context: {p.context}")
        print(f"Emotion: {p.emotion}")
        print("-" * 40)

# function to display the critiques output
def display_critiques(critic_output: CriticOutput):
    print("RAG Chain Response:")
    #parsed = json.loads(critic_output)
    for c in critic_output.critiques:
        print(f"Panel Number: {c.panel}")
        print(f"Pain Point: {c.pain_point}")
        print(f"Reason: {c.reason}")
        print(f"Severity: {c.severity}")
        print("-" * 40)

# function to display the recommendations output
def display_recommendations(design_output: DesignOutput):
    print("Design Recommendations:")
    for r in design_output.recommendations:
        print(f"Panel Number: {r.panel}")
        print(f"Pain Point: {r.pain_point}")
        print(f"Recommnedation: {r.recommendation}")
        print("-" * 40)

# ENTRY POINT
if __name__ == "__main__":
    chat_model = get_chat_model()
    user_input = collect_input()
 
    # Generating Storyboard
    print("==" *60)
    print("\nGenerating storyboard...\n")
    print("==" *60)
    output = run_journey_agent(user_input, chat_model)
    display_storyboard(output)

    # Pulling documents
    print("\nRetrieving relevant documentation...\n")
    retrieved_docs = basic_retrieve(output.panels, top_k=5)

    # Critiquing storyboard output
    print("==" *60)
    print("\nRunning UX Critic Agent...\n")
    print("==" *60)
    critic_output = run_critic_agent(output.panels, retrieved_docs, chat_model)
    display_critiques(critic_output)

    # Generating design recommendations
    print("\nGenerating design recommendations...\n")
    print("==" *60)
    design_output = run_design_agent(output, critic_output, chat_model)
    display_recommendations(design_output)
