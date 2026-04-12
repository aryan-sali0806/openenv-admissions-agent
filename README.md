---
title: Admissions Env Environment Server
emoji: 🎙️
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Admissions Env Environment

An autonomous, LLM-powered postgraduate admissions officer environment. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

## Quick Start

The simplest way to use the Admissions Env environment is through the `AdmissionsEnv` class:

```python
from admissions_env import AdmissionsAction, AdmissionsEnv
try:
    # Create environment from Docker image
    admissions_envenv = AdmissionsEnv.from_docker_image("admissions_env-env:latest")
    # Reset
    result = admissions_envenv.reset()
    print("New Candidate Profile Generated!")
    # Send multiple actions
    actions = [
        AdmissionsAction(action_type="check_eligibility", action_args={"cgpa_threshold": 6.5}),
        AdmissionsAction(action_type="analyze_resume", action_args={}),
        AdmissionsAction(action_type="admit", action_args={})
    ]
    for action in actions:
        result = admissions_envenv.step(action)
        print(f"Executed: '{action.action_type}'")
        print(f"  → Done: {result.done}")
        print(f"  → Reward: {result.reward}")
finally:
    # Always clean up
    admissions_envenv.close()
That's it! The AdmissionsEnv.from_docker_image() method handles:
Starting the Docker container
Waiting for the server to be ready
Connecting to the environment
Container cleanup when you call close()
Building the Docker Image
Before using the environment, you need to build the Docker image:
Bash
# From project root
docker build -t admissions_env-env:latest -f server/Dockerfile .
Deploying to Hugging Face Spaces
You can easily deploy your OpenEnv environment to Hugging Face Spaces using the openenv push command:
Bash
# From the environment directory (where openenv.yaml is located)
openenv push
# Or specify options
openenv push --namespace my-org --private
The openenv push command will:
Validate that the directory is an OpenEnv environment (checks for openenv.yaml)
Prepare a custom build for Hugging Face Docker space (enables web interface)
Upload to Hugging Face (ensuring you're logged in)
Prerequisites
Authenticate with Hugging Face: The command will prompt for login if not already authenticated
Options
--directory, -d: Directory containing the OpenEnv environment (defaults to current directory)
--repo-id, -r: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
--base-image, -b: Base Docker image to use (overrides Dockerfile FROM)
--private: Deploy the space as private (default: public)
Examples
Bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push
# Push to a specific repository
openenv push --repo-id my-org/my-env
# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest
# Push as a private space
openenv push --private
# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
After deployment, your space will be available at:
https://huggingface.co/spaces/<repo-id>
The deployed space includes:
Web Interface at /web - Interactive UI for exploring the environment
API Documentation at /docs - Full OpenAPI/Swagger interface
Health Check at /health - Container health monitoring
WebSocket at /ws - Persistent session endpoint for low-latency interactions
Environment Details
Action
AdmissionsAction: Expects a strict JSON payload for tool execution
action_type (str) - The name of the tool to use (e.g., check_eligibility, admit, reject).
action_args (dict) - The arguments required for that specific tool.
Observation
AdmissionsObservation: Contains the candidate state and tool responses
candidate_data (dict) - Current known information about the applicant.
tool_result (str) - The feedback or data returned from the last action.
reward (float) - The reward earned for the last action.
done (bool) - True if a terminal action (admit, reject) was taken.
metadata (dict) - Additional info like step count
Reward
The environment issues dynamic Reinforcement Learning rewards:
Partial Rewards (e.g., 0.05) for correctly using investigative tools like analyze_resume.
Terminal Rewards (e.g., 1.0) for making the correct final admission decision based on the candidate profile.
Advanced Usage
Connecting to an Existing Server
If you already have a Admissions Env environment server running, you can connect directly:
Python
from admissions_env import AdmissionsEnv, AdmissionsAction
# Connect to existing server
admissions_envenv = AdmissionsEnv(base_url="<ENV_HTTP_URL_HERE>")
# Use as normal
result = admissions_envenv.reset()
result = admissions_envenv.step(AdmissionsAction(action_type="check_eligibility", action_args={}))
Note: When connecting to an existing server, admissions_envenv.close() will NOT stop the server.
Using the Context Manager
The client supports context manager usage for automatic connection management:
Python
from admissions_env import AdmissionsAction, AdmissionsEnv
# Connect with context manager (auto-connects and closes)
with AdmissionsEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print("Environment Reset. Evaluating...")
    
    # Multiple steps with low latency
    actions = ["analyze_resume", "score_profile", "admit"]
    for act in actions:
        result = env.step(AdmissionsAction(action_type=act, action_args={}))
        print(f"Action: {act} -> Reward: {result.reward}")
The client uses WebSocket connections for:
Lower latency: No HTTP connection overhead per request
Persistent session: Server maintains your environment state
Efficient for episodes: Better for many sequential steps
Concurrent WebSocket Sessions
The server supports multiple concurrent WebSocket connections. To enable this,
modify server/app.py to use factory mode:
Python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    AdmissionsEnvironment,  # Pass class, not instance
    AdmissionsAction,
    AdmissionsObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
Then multiple clients can connect simultaneously:
Python
from admissions_env import AdmissionsAction, AdmissionsEnv
from concurrent.futures import ThreadPoolExecutor
def run_episode(client_id: int):
    with AdmissionsEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(AdmissionsAction(action_type="admit", action_args={}))
        return client_id, result.reward
# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
Development & Testing
Direct Environment Testing
Test the environment logic directly without starting the HTTP server:
Bash
# From the server directory
python3 server/admissions_env_environment.py
This verifies that:
Environment resets correctly
Step executes actions properly
State tracking works
Rewards are calculated correctly
Running Locally
Run the server locally for development:
Bash
uvicorn server.app:app --reload
Project Structure
admissions_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py           # Module exports
├── README.md             # This file
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Locked dependencies (generated)
├── client.py             # AdmissionsEnv client
├── models.py             # Action and Observation models
└── server/
    ├── __init__.py       # Server module exports
    ├── admissions_env_environment.py  # Core environment logic
    ├── app.py            # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile        # Container image definition
    └── Dockerfile         # Container image definition
```
