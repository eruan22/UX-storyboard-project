import sys
import json

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from utils.chroma_setup import basic_retrieve, get_vectorstore
from utils.rag_triad import run_rag_triad
from agents.journey_agent import run_journey_agent
from agents.ux_critic_agent import run_critic_agent
from agents.design_agent import run_design_agent
from models.schemas import StoryboardInput, StoryboardOutput, Panel, CriticOutput, PanelCritique, DesignOutput, DesignRecommendation

# LLM SET UP
USE_REMOTE = True
sys.path.insert(0, '../inclass')
from llm_utils import get_llm, get_chat_model

chat_model = get_chat_model(use_remote=USE_REMOTE, model="qwen3.5:4b")

# LOAD SAMPLE DATA
with open("rag_triad_sample_inputs.json", "r") as f:
    SAMPLE_INPUTS = json.load(f)

# run RAG triad evaluation
triad_results = []
length_rec = []
for i, sample in enumerate(SAMPLE_INPUTS):
    print(f"--Running RAG triad evaluation for sample {i+1}--")

    # journey agent
    print("Running Journey Agent...")
    user_input = StoryboardInput(**sample)
    storyboard_output = run_journey_agent(user_input, chat_model)

    # retrieve docs
    vector_store = get_vectorstore()
    retrieved_docs = basic_retrieve(storyboard_output.panels, vector_store, top_k=5)

    # critic agent
    print("\nRunning Critic Agent...")
    critic_output = run_critic_agent(storyboard_output.panels, retrieved_docs, chat_model)

    # run RAG triad evaluation
    triad_result = run_rag_triad(storyboard_output.panels, retrieved_docs, critic_output, chat_model)
    triad_results.append(triad_result)

    # print results
    print(f"Context Relevance Score: {triad_result['context_relevance']}")
    print(f"Faithfulness Score: {triad_result['faithfulness']}")
    print(f"Answer Relevance Score: {triad_result['answer_relevance']}")

    # design agent
    print("\nRunning Design Agent...")
    design_output = run_design_agent(storyboard_output.panels, critic_output, chat_model)
    recommendations = design_output.recommendations
    for r in recommendations:
        length = len(r.recommendation.split())
        length_rec.append(length)
        print(length)

# print average scores across all samples
avg_context_relevance = sum(r["context_relevance"] for r in triad_results) / len(triad_results)
avg_faithfulness = sum(r["faithfulness"] for r in triad_results) / len(triad_results)
avg_answer_relevance = sum(r["answer_relevance"] for r in triad_results) / len(triad_results)
print(f"\nAverage Context Relevance Score: {avg_context_relevance}")
print(f"Average Faithfulness Score: {avg_faithfulness}")
print(f"Average Answer Relevance Score: {avg_answer_relevance}")

avg_length = sum(length_rec) / len(length_rec)
print(f"\nAverage number of words in Recommendation: {avg_length}")