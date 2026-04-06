import sys
import json

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from utils.chroma_setup import get_vectorstore, basic_retrieve
from utils.rag_triad import run_rag_triad
from agents.journey_agent import run_journey_agent
from agents.ux_critic_agent import run_critic_agent
from agents.design_agent import run_design_agent
from models.schemas import StoryboardInput, StoryboardOutput, Panel, CriticOutput, PanelCritique, DesignOutput, DesignRecommendation

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

# LOAD SAMPLE DATA
with open("rag_triad_sample_inputs.json", "r") as f:
    SAMPLE_INPUTS = json.load(f)

#SET UP
vectorstore = get_vectorstore()

# run RAG triad evaluation
triad_results = []
for i, sample in enumerate(SAMPLE_INPUTS):
    print(f"--Running RAG triad evaluation for sample {i+1}--")

    # journey agent
    user_input = StoryboardInput(**sample["user_input"])
    storyboard_output = run_journey_agent(user_input, chat_model)

    # retrieve docs
    retrieved_docs = basic_retrieve(vectorstore, user_input)

    # critic agent
    critic_output = run_critic_agent(storyboard_output.panels, retrieved_docs, chat_model)

    # design agent
    design_output = run_design_agent(critic_output, chat_model)

    # run RAG triad evaluation
    triad_result = run_rag_triad(storyboard_output.panels, retrieved_docs, critic_output, chat_model)
    triad_results.append(triad_result)

    # print results
    print(f"Context Relevance Score: {triad_result['context_relevance']}")
    print(f"Faithfulness Score: {triad_result['faithfulness']}")
    print(f"Answer Relevance Score: {triad_result['answer_relevance']}")

# print average scores across all samples
avg_context_relevance = sum(r["context_relevance"] for r in triad_results) / len(triad_results)
avg_faithfulness = sum(r["faithfulness"] for r in triad_results) / len(triad_results)
avg_answer_relevance = sum(r["answer_relevance"] for r in triad_results) / len(triad_results)
print(f"\nAverage Context Relevance Score: {avg_context_relevance}")
print(f"Average Faithfulness Score: {avg_faithfulness}")
print(f"Average Answer Relevance Score: {avg_answer_relevance}")
