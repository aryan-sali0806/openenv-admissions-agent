"""
Inference Script for Admissions Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local image to use for the environment if you are using from_docker_image()
                   method
"""
import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

# Import your environment classes
from client import AdmissionsEnv
from models import AdmissionsAction

IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("ADMISSIONS_BENCHMARK", "admissions_env")

MAX_STEPS = 10
TEMPERATURE = 0.1
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

# Define 3 distinct curriculum tasks
TASKS = ["admissions_easy", "admissions_medium", "admissions_hard"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert postgraduate admissions officer.
    Evaluate candidates using available tools, ending with a final decision (admit, reject, waitlist).
    
    CRITICAL: You must reply with EXACTLY ONE valid JSON object and NOTHING ELSE.
    Do not use markdown formatting or code blocks.
    
    JSON Schema:
    {
      "action_type": "tool_name",
      "action_args": {"arg_name": "arg_value"}
    }
    """
).strip()

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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, last_obs: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last Observation: {last_obs}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        
        Send your next action as a strict JSON object.
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, last_obs: str, last_reward: float, history: List[str]) -> AdmissionsAction:
    user_prompt = build_user_prompt(step, last_obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        clean_text = text.replace("```json", "").replace("```", "").strip()
        action_data = json.loads(clean_text)
        
        return AdmissionsAction(**action_data)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return AdmissionsAction(action_type="analyze_resume", action_args={})

async def main() -> None:
    if not API_KEY:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await AdmissionsEnv.from_docker_image(IMAGE_NAME)
    else:
        env = AdmissionsEnv(base_url="http://127.0.0.1:8000")
        await env.connect()

    try:
        # Execute the 3 distinct tasks sequentially
        for task_name in TASKS:
            history: List[str] = []
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False

            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            
            result = await env.reset() 
            last_obs = result.observation.model_dump_json(indent=2)
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action = get_model_action(client, step, last_obs, last_reward, history)

                try:
                    result = await env.step(action)
                    obs = result.observation
                    reward = result.reward or 0.0
                    done = result.done
                    error = None
                    last_obs = obs.model_dump_json(indent=2)
                except Exception as e:
                    reward = 0.0
                    done = True
                    error = str(e)
                    last_obs = "{}"

                rewards.append(reward)
                steps_taken = step
                last_reward = reward
                
                action_str = f"{action.action_type}({json.dumps(action.action_args)})".replace(" ", "")
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

                if done:
                    break

                if not IMAGE_NAME:
                    await asyncio.sleep(0.1)

            # Calculate raw score bounded between [0.0, 1.0]
            score = sum(rewards)
            score = max(0.0, min(1.0, score)) 
            success = score >= SUCCESS_SCORE_THRESHOLD

            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())