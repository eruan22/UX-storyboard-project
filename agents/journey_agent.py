from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models.schemas import StoryboardInput, StoryboardOutput

# BUILD JOURNEY CHAIN
def build_journey_chain(chat_model):
    journey_parser = PydanticOutputParser(pydantic_object=StoryboardOutput)
    # create prompt template for Journey Agent
    journey_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a user experience designer creating a storyboard to illustrate a user's journey with a product."),
        ("human", """Given the following user information, create a storyboard consisting of 6 panels \
    that describes the user's action, context, and emotion. Be concise and focus on the key interaction \
    in context of the product and the user's goals.

    User Persona: {persona}
    User Goal: {goal}
    Product: {product}
    Scenario: {scenario}

    Format your response as JSON with this structure:
    {{
    "panels": [
        {{
        "panel_number": The sequential number of this panel in the storyboard,
        "action": "concise description of user action",
        "context": "details about user state or environment",
        "emotion": "one or two word emotion"
        }}
    ]
    }}

    Return only valid JSON, no extra text.""")
    ])

    journey_chain = journey_prompt | chat_model | journey_parser
    return journey_chain

# RUN JOURNEY AGENT CHAIN
def run_journey_agent(user_input: StoryboardInput, chat_model) -> StoryboardOutput:
    chain = build_journey_chain(chat_model)
    result = chain.invoke({"persona": user_input.persona, 
                                "goal": user_input.goal, 
                                "product": user_input.product, 
                                "scenario": user_input.scenario})
    print("journey agent finished!")
    return result