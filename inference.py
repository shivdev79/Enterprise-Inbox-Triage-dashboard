import os
import json
from typing import List, Optional
from openai import OpenAI
from server.my_env_environment import MyEnvironment
from models import MyAction

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def call_llm(client, model, messages, tools):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0
        )
        return response.choices[0].message
    except Exception as e:
        return None

def run_task(env: MyEnvironment, task_id: str, client: OpenAI, model: str):
    benchmark_name = "email_triage"
    log_start(task=task_id, env=benchmark_name, model=model)
    
    env.reset()
    obs = env.step(MyAction(action_type="start_task", task_id=task_id))
    
    messages = [
        {"role": "system", "content": "You are a helpful AI customer support agent triage system. "
                                      "You must use the exact tool provided to take actions based on your task. "
                                      "Always read your task description and use tools precisely. "
                                      "Once you are done with the final requirement, use the 'submit' action type to finish. "
                                      "Do not output markdown in arguments, just pure text."}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "take_action",
                "description": "Perform an action in the email triage environment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "enum": ["read_email", "reply", "forward", "archive", "search_knowledge_base", "escalate_to_human", "submit"],
                            "description": "The type of action to perform. 'submit' finishes the task."
                        },
                        "email_id": {"type": "string", "description": "The ID of the email to interact with"},
                        "message": {"type": "string", "description": "The text message body"},
                        "forward_to": {"type": "string", "description": "Email address to forward to"},
                        "query": {"type": "string", "description": "Search query for the Knowledge Base"},
                        "reason": {"type": "string", "description": "Reason for escalating to human"}
                    },
                    "required": ["action_type"]
                }
            }
        }
    ]
    
    rewards = []
    steps_taken = 0
    
    for step_num in range(1, 16):
        obs_json = obs.model_dump_json()
        messages.append({"role": "user", "content": f"Current Observation: {obs_json}"})
        
        reply = call_llm(client, model, messages, tools)
        if not reply:
            # Fake fail if API key blocks us to keep stdout loop unbroken
            log_step(step=step_num, action="api_error", reward=-1.0, done=True, error="invalid_api_key")
            rewards.append(-1.0)
            steps_taken = step_num
            break
            
        messages.append(reply)
        error = None
        done = False
        reward = 0.0
        
        if reply.tool_calls:
            tool_call = reply.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            action_str = args.get("action_type", "unknown")
            
            try:
                action = MyAction(**args)
                obs = env.step(action)
                reward = obs.reward
                done = obs.done
            except Exception as e:
                error = str(e).replace('\n', ' ')
                obs = env._build_observation(f"Failed to execute action: {error}", -0.1, False)
                reward = -0.1
                
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"Action feedback: {obs.feedback}"
            })
        else:
            action_str = "no_tool_used"
            error = "Agent returned pure text instead of tool call"
            reward = -0.1
            obs = env._build_observation("You must use the take_action tool.", reward, False)
            messages.append({"role": "user", "content": "You must use the take_action tool."})
            
        rewards.append(reward)
        steps_taken = step_num
        
        log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)
        
        if done:
            break
            
    success = obs.score >= 0.8
    log_end(success=success, steps=steps_taken, score=obs.score, rewards=rewards)
    return obs.score

def main():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "dummy_key"
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    env = MyEnvironment()
    
    scores = {}
    for task in ["easy", "medium", "hard"]:
        score = run_task(env, task, client, model)
        scores[task] = score

if __name__ == "__main__":
    main()
