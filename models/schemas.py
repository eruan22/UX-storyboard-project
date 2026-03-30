from pydantic import BaseModel, Field
from typing import List, Literal

# create input schema for Journey Agent
class StoryboardInput(BaseModel):
    """Input schema for the StoryboardAgent."""
    persona: str = Field(..., description = "The user's persona or role, " \
    "which may influence their goals and constraints.")
    goal: str = Field(..., description = "The user's high-level goal or task.")
    product: str = Field(..., description = "The product or service the user is " \
    "interacting with, which may have specific features or limitations.")
    scenario: str = Field(..., description = "The specific context or situation "
    "in which the user is trying to achieve their goal, which may include " \
    "environmental factors or constraints.")

# create output panel for Journey Agent
class Panel(BaseModel):
    """Output panel for the StoryboardAgent."""
    panel_number: int = Field(..., description = "The sequential number of the " \
    "panel in the storyboard.")
    action: str = Field(..., description = "A concise description of the user's " \
    "action or interaction in this panel.")
    context: str = Field(..., description = "Additional context or details about " \
    "the user's state or environment during this panel.")
    emotion: str = Field(..., description = "A one or two word description of the " \
    "user's emotional state during this panel (e.g., 'frustrated', 'satisfied', 'confused').")

# class output schema for Journey Agent
class StoryboardOutput(BaseModel):
    """Output schema for the StoryboardAgent."""
    panels: List[Panel] = Field(..., description = "A list of panels that make up the storyboard, " \
    "each describing a step in the user's journey towards their goal.")

# create critic input for UX Critic Agent
# class CriticInput(BaseModel):
#     """Input to the Critic Agent"""
#     panels: List[Panel] = Field(..., description="List of UI panels to critique")
#     retrieved_docs: List[str] = Field(..., description="Relevant documentation snippets retrieved from the vector store")

# create critic output panel for UX Critic Agent
class PanelCritique(BaseModel):
    """Panel output from the Critic Agent"""
    panel: int = Field(..., description="Index of the panel being critiqued")
    pain_point: str = Field(..., description="Description of the identified pain point")
    reason: str = Field(..., description="Explanation of why this is a pain point based on the retrieved documentation")
    severity: Literal["Low", "Medium", "High"] = Field(..., description="Severity of the pain point based on the documentation and potential user impact")

# create critic output for UX Critic Agent
class CriticOutput(BaseModel):
    """Output from the Critic Agent"""
    critiques: List[PanelCritique] = Field(..., description="List of critiques for each panel")