"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

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

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("ADMISSIONS_TASK", "admissions_evaluation")
BENCHMARK = os.getenv("ADMISSIONS_BENCHMARK", "admissions_env")

MAX_STEPS = 10
TEMPERATURE = 0.1
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

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
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


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
        
        # Parse JSON and clean any markdown
        clean_text = text.replace("```json", "").replace("```", "").strip()
        action_data = json.loads(clean_text)
        return AdmissionsAction(**action_data)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback to prevent crashing the evaluation
        return AdmissionsAction(action_type="analyze_resume", action_args={})


async def main() -> None:
    if not API_KEY:
        raise ValueError("API_KEY or HF_TOKEN is missing!")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Automatically handle the Judges' Docker environment vs Your Local Environment
    if IMAGE_NAME:
        env = await AdmissionsEnv.from_docker_image(IMAGE_NAME)
    else:
        env = AdmissionsEnv(base_url="http://127.0.0.1:8000")
        await env.connect()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
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

            # Format action string cleanly without spaces to comply with STDOUT rules
            action_str = f"{action.action_type}({json.dumps(action.action_args)})".replace(" ", "")

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

            # Add a small delay if interacting with local to prevent race conditions
            if not IMAGE_NAME:
                await asyncio.sleep(0.1)

        # Calculate final score (clamped strictly to [0, 1])
        score = sum(rewards)
        score = min(max(score, 0.0), 1.0)  
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        
        # This will ALWAYS emit, fulfilling the final rule
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())