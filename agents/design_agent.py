from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models.schemas import DesignOutput, StoryboardInput, StoryboardOutput, Panel, CriticOutput, DesignRecommendation
from typing import List

# BUILD DESIGN CHAIN
def build_design_chain(chat_model):
    design_parser = PydanticOutputParser(pydantic_object=DesignOutput)
    # create design agent prompt
    design_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a user experience (UX) designer tasked with improving a storyboard
            that illustrates a user's journey towards achieving a specific goal. 
            The storyboard consists of multiple panels, each depicting a step in the user's journey. 
            Your task is to analyze each panel, take the identified pain points, 
            and provide actionable design recommendations to enhance the overall user experience."""),
            ("human", """Here is the storyboard you need to suggest improvements for:
            {panels}
            These are the pain points identified in each panel:
            {pain_points}
            For each pain point, provide a specific, actionable design recommendation.
            Format your response as ONLY JSON with this structure:
            {{
                "recommendations":[
                    {{
                        "panel": panel number that this recommendation applies to,
                        "pain_point": the specific pain point identified in the panel,
                        "recommendation": the design recommendation to address the pain point
                    }},
                    ...
                ]
            }}""")
        ])

    design_chain = design_prompt | chat_model | design_parser
    return design_chain

# FORMAT DESIGN AGENT INPUTS
# format panels
def format_panels(panels: List[Panel]) -> str:
    return "\n".join([
        f"Panel {p.panel_number}: Action='{p.action}', Context='{p.context}', Emotion='{p.emotion}'"
        for p in panels
    ])

# format pain points
def format_pain_points(critic_output:CriticOutput) -> str:
    return "\n".join([
        f"Panel {c.panel} ({c.severity} severity): {c.pain_point} — {c.reason}"
        for c in critic_output.critiques
    ])

# RUN DESIGN AGENT CHAIN
def run_design_agent(storyboard_output: StoryboardOutput, critic_output: CriticOutput, chat_model) -> DesignOutput:
    chain = build_design_chain(chat_model)
    response = chain.invoke({
        "panels": format_panels(storyboard_output.panels),
        "pain_points": format_pain_points(critic_output)
    })
    return response