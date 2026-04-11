import json
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from models.schemas import Panel, CriticOutput, PanelCritique

# BUILD CRITIC CHAIN
def build_critic_chain(chat_model):
    parser = StrOutputParser()
    critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a UX Critic Agent. Your job is to analyze the provided storyboard panels and
identify potential UX pain points based on the retrieved documentation.
For each panel, determine if there are any UX issues that could arise based on the user's action and context.
Use the retrieved documentation and context to support your analysis and provide a severity rating for each identified pain point.
Return only valid JSON, no extra text."""),
    ("human", """Here are the storyboard panels:
{panels}

Here is the relevant UX research:
{retrieved_docs}

Identify the UX pain points in these panels backed by the research above.
Format your response as JSON with this structure:
{{
    "critiques": [
        {{
        "panel": <index of the panel being critiqued>,
        "pain_point": <description of the identified pain point>,
        "reason": <explanation of why this is a pain point based on the retrieved documentation>,
        "severity": <severity of the pain point based on the documentation and potential user impact (Low, Medium, High)>
        }}
     ]
}}""")
])
    critic_chain = critic_prompt | chat_model | parser
    return critic_chain

# FORMAT CRITIC CHAIN INPUTS
# format panels for RAG chain
def format_panels(panels: List[Panel]) -> str:
    return "\n".join([
        f"Panel {p.panel_number}: Action='{p.action}', Context='{p.context}', Emotion='{p.emotion}'"
        for p in panels
    ])

# format retrieved docs for RAG chain
def format_retrieved_docs(retrieved_docs: List[str]) -> str:
    return "\n\n".join([f"[Doc {i+1}]: {doc}" for i, doc in enumerate(retrieved_docs)])

# RUN CRITIC AGENT CHAIN
def run_critic_agent(panels: List[Panel], retrieved_docs: List[str], chat_model) -> CriticOutput:
    chain = build_critic_chain(chat_model)
    response = chain.invoke({
        "panels": format_panels(panels),
        "retrieved_docs": format_retrieved_docs(retrieved_docs)
    })
    parsed = json.loads(response)
    critiques = [PanelCritique(**c) for c in parsed["critiques"]]
    print("ux critic agent finished!")
    return CriticOutput(critiques=critiques)
