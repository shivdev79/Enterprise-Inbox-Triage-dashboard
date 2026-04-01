import streamlit as st
import os
import json
import time
import pandas as pd
from openai import OpenAI
from server.my_env_environment import MyEnvironment
from models import MyAction

st.set_page_config(layout="wide", page_title="AI Agent Telemetry", page_icon="🤖")

st.title("🤖 Live AI Agent Telemetry Dashboard")
st.markdown("Watch the agent triage customer support emails in real-time. This dashboard visualizes the exact state of the OpenEnv instance.")

# Setup sidebar
st.sidebar.header("Configuration")
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "dummy"
base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
model = os.getenv("MODEL_NAME", "gpt-4o-mini")

task_choice = st.sidebar.selectbox("Select Task Difficulty", ["easy", "medium", "hard"])
start_button = st.sidebar.button("🚀 Start Agent Inference")

# UI Columns
col_inbox, col_feed, col_stats = st.columns([1, 1.5, 1])

with col_inbox:
    st.subheader("📬 Current Inbox State")
    inbox_placeholder = st.empty()

with col_feed:
    st.subheader("🧠 Agent Thought & Action Feed")
    feed_placeholder = st.empty()

with col_stats:
    st.subheader("📈 Telemetry & Score")
    score_metric = st.empty()
    token_metric = st.empty()
    chart_placeholder = st.empty()

def mock_llm_response(step, task, obs):
    """Fallback if no API key is set, to visualize the 'wow' factor instantly."""
    time.sleep(1.5)
    inbox_ids = [e.id for e in obs.inbox]
    if task == "easy" and step == 0 and "e2" in inbox_ids:
        return {"action_type": "archive", "email_id": "e2"}
    if task == "easy" and step == 1:
        return {"action_type": "submit"}
    if task == "medium":
        if step == 0 and "m2" in inbox_ids:
            return {"action_type": "read_email", "email_id": "m2"}
        if step == 1 and "m2" in inbox_ids:
            return {"action_type": "search_knowledge_base", "query": "refund policy"}
        if step == 2 and "m2" in inbox_ids:
            return {"action_type": "reply", "email_id": "m2", "message": "Refund denied."}
        if step == 3 and "m2" in inbox_ids:
            return {"action_type": "archive", "email_id": "m2"}
        if step == 4:
            return {"action_type": "submit"}
    if task == "hard":
        if step == 0 and "h1" in inbox_ids:
            return {"action_type": "forward", "email_id": "h1", "forward_to": "it@company.com"}
        if step == 1 and "h3" in inbox_ids:
            return {"action_type": "reply", "email_id": "h3", "message": "Approved"}
        if step == 2 and "h5" in inbox_ids:
            return {"action_type": "read_email", "email_id": "h5"}
        if step == 3 and "h5" in inbox_ids:
            return {"action_type": "escalate_to_human", "email_id": "h5", "reason": "VIP client extreme anger and lawsuit threat"}
        if step == 4 and ("h2" in inbox_ids or "h4" in inbox_ids):
            target = "h2" if "h2" in inbox_ids else "h4"
            return {"action_type": "archive", "email_id": target}
        if step == 5 and ("h2" in inbox_ids or "h4" in inbox_ids):
            target = "h2" if "h2" in inbox_ids else "h4"
            return {"action_type": "archive", "email_id": target}
        if step == 6:
            return {"action_type": "submit"}
    return {"action_type": "submit"}

def call_llm(client, messages, tools):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0
        )
        msg = response.choices[0].message
        tokens = response.usage.total_tokens if response.usage else 0
        return msg, tokens
    except Exception as e:
        return str(e), 0

if start_button:
    env = MyEnvironment()
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    st.session_state.feed_lines = []
    
    env.reset()
    obs = env.step(MyAction(action_type="start_task", task_id=task_choice))
    
    score_history = [0.0]
    token_usage_total = [0]
    
    messages = [
        {"role": "system", "content": "You are a customer support agent. Use the exact tool provided to interact with emails. Read your task description. Output pure tool calls. Once done, use 'submit' action."}
    ]
    
    tools = [{
        "type": "function",
        "function": {
            "name": "take_action",
            "description": "Perform an action in the email triage environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "enum": ["read_email", "reply", "forward", "archive", "search_knowledge_base", "escalate_to_human", "submit"]},
                    "email_id": {"type": "string"},
                    "message": {"type": "string"},
                    "forward_to": {"type": "string"},
                    "query": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["action_type"]
            }
        }
    }]
    
    st.session_state.feed_lines.append(f"✅ **Started Task:** {task_choice.upper()}")
    st.session_state.feed_lines.append(f"📝 **Objective:** {obs.task_description}")
    
    for step in range(15):
        # Update Inbox HTML
        inbox_html = ""
        for email in obs.inbox:
            badge_color = "red" if email.priority in ["urgent", "high"] else ("blue" if email.priority == "normal" else "gray")
            inbox_html += f"""
            <div style="border:1px solid #ddd; padding:10px; border-radius:5px; margin-bottom:10px; background-color:#f9f9f9;">
                <b>{email.sender}</b> <span style="color:white; background-color:{badge_color}; padding:2px 5px; border-radius:3px; font-size:10px;">{email.priority.upper()}</span><br/>
                <i>{email.subject}</i><br/>
                <small style="color:#666;">Tier: {email.customer_tier} | Sent: {email.timestamp} | Sentiment: {email.sentiment.title()}</small>
            </div>
            """
        if not obs.inbox:
            inbox_html = "<i>Inbox Empty!</i>"
        inbox_placeholder.markdown(inbox_html, unsafe_allow_html=True)
        
        # Update Telemetry
        score_metric.metric("Evaluating Environment Score", f"{obs.score:.2f} / 1.00")
        token_metric.metric("Total Tokens Processed", f"{token_usage_total[-1]:,}")
        df = pd.DataFrame({"Score": score_history})
        chart_placeholder.line_chart(df)
        
        obs_json = obs.model_dump_json()
        messages.append({"role": "user", "content": f"Observation: {obs_json}"})
        
        if api_key == "dummy" or "Error" in api_key:
            # Fallback Mock Mode Visualizer
            st.session_state.feed_lines.append(f"*(Mock Mode LLM step {step+1})*")
            feed_placeholder.markdown("<br/>".join(st.session_state.feed_lines), unsafe_allow_html=True)
            args = mock_llm_response(step, task_choice, obs)
            tokens = 120
            reply_msg = {
                "role": "assistant", 
                "tool_calls": [{"id": "mock_call", "function": {"name": "take_action", "arguments": json.dumps(args)}}]
            }
            messages.append(reply_msg)
        else:
            with feed_placeholder.container():
                st.markdown("<br/>".join(st.session_state.feed_lines), unsafe_allow_html=True)
                with st.spinner(f"Agent executing step {step+1}..."):
                    reply, tokens = call_llm(client, messages, tools)
            
            if isinstance(reply, str):
                st.session_state.feed_lines.append(f"❌ **API Error:** {reply}")
                feed_placeholder.markdown("<br/>".join(st.session_state.feed_lines), unsafe_allow_html=True)
                st.error(f"LLM API Error: {reply}")
                break
            
            messages.append(reply)
            if not reply.tool_calls:
                st.session_state.feed_lines.append(f"⚠️ Agent forgot to use tool: {reply.content}")
                messages.append({"role": "user", "content": "You must use the exact take_action tool."})
                continue
                
            tool_call = reply.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
        
        # Apply Action
        action_type = args.get("action_type", "unknown")
        st.session_state.feed_lines.append(f"🤖 **Action Chosen:** `{action_type.upper()}`")
        if "email_id" in args:
            st.session_state.feed_lines.append(f"└ Target: `{args['email_id']}`")
        
        action = MyAction(**args)
        obs = env.step(action)
        
        st.session_state.feed_lines.append(f"⚙️ **Environment Feedback:** {obs.feedback}")
        st.session_state.feed_lines.append("<hr style='margin:10px 0px'/>")
        feed_placeholder.markdown("<br/>".join(st.session_state.feed_lines), unsafe_allow_html=True)
        
        score_history.append(obs.score)
        token_usage_total.append(token_usage_total[-1] + tokens)
        
        if obs.done:
            st.session_state.feed_lines.append(f"🏆 **Task Completed! Final Score: {obs.score:.2f}**")
            feed_placeholder.markdown("<br/>".join(st.session_state.feed_lines), unsafe_allow_html=True)
            
            # Final UI refresh
            score_metric.metric("Evaluating Environment Score", f"{obs.score:.2f} / 1.00")
            df = pd.DataFrame({"Score": score_history})
            chart_placeholder.line_chart(df)
            st.success("Simulation done!")
            break
            
    st.sidebar.success("Inference loop finished.")
