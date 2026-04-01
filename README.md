---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---
# Customer Support Email Triage Environment

## Environment Description
The Customer Support Email Triage Environment models a real-world email inbox management task. 
An AI agent must process an inbox filled with varied emails including customer refund requests, spam, system outage logs, and partner emails. 

## Action Space
`MyAction` defines the actions the agent can take:
- `action_type`: Literal["start_task", "read_email", "reply", "forward", "archive", "submit"]
- `email_id`: ID of the email to interact with.
- `message`: The content to reply or forward with.
- `forward_to`: Target email address for forwarding.
- `task_id`: "easy", "medium", or "hard" (starts the respective task).

## Observation Space
`MyObservation` gives the current state of the environment:
- `task_description`: The required task for the current run.
- `inbox`: A list of summaries for emails currently in the inbox.
- `current_email_body`: The full content of the email currently read.
- `feedback`: String feedback from the environment based on the last action.
- `score`: The agent's current progress in completing the task smoothly (0.0 to 1.0).

## Setup Instructions

1. Install requirements using `uv lock` followed by `uv sync`, or `pip install openenv-core openai`.
2. Set API variables: `OPENAI_API_KEY`, `API_BASE_URL` (optional), and `MODEL_NAME` (e.g., `gpt-4o-mini`).
3. (Optional) Start the API server for HTTP access via `python -m server.app`.

## Baseline Scores
Running `python inference.py` on `gpt-4o-mini` produces identical optimal action sequences and maximum scores:
- Easy Task: 1.00
- Medium Task: 1.00
- Hard Task: 1.00
