from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class EmailSummary(BaseModel):
    id: str = Field(description="Unique identifier for the email")
    sender: str = Field(description="Sender email address")
    subject: str = Field(description="Email subject")
    is_read: bool = Field(description="Whether the email has been read")
    timestamp: str = Field(description="Time the email was received")
    priority: str = Field(description="Priority of the email (low, normal, high, urgent)")
    sentiment: str = Field(description="Detected sentiment (e.g., angry, neutral, happy)")
    customer_tier: str = Field(description="Customer tier (e.g., standard, premium, enterprise, none)")

class MyAction(Action):
    """Action for the Email Triage environment."""
    action_type: Literal["start_task", "read_email", "reply", "forward", "archive", "search_knowledge_base", "escalate_to_human", "submit"] = Field(
        ..., description="Action to perform"
    )
    email_id: Optional[str] = Field(None, description="ID of the email to act upon")
    message: Optional[str] = Field(None, description="Message to reply or forward with")
    forward_to: Optional[str] = Field(None, description="Email address to forward to")
    task_id: Optional[str] = Field(None, description="Task to start: 'easy', 'medium', or 'hard'")
    query: Optional[str] = Field(None, description="Search query for the Knowledge Base")
    reason: Optional[str] = Field(None, description="Reason for escalating this ticket to a human")

class MyObservation(Observation):
    """Observation from the Email Triage environment."""
    task_description: str = Field(default="", description="The current task description for the agent to follow")
    inbox: List[EmailSummary] = Field(default_factory=list, description="List of emails in current inbox")
    current_email_body: Optional[str] = Field(None, description="Body of the currently read email")
    kb_result: Optional[str] = Field(None, description="Result of the Knowledge Base search")
    feedback: str = Field(default="", description="Feedback from the last action")
    score: float = Field(default=0.0, description="Current progress/score of task (0.0 to 1.0)")
