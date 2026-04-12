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
  - rl-environment
  - llm-agent
  - curriculum-learning
---

# Admissions Env for OpenEnv

An autonomous, LLM-powered postgraduate admissions officer environment. Built for evaluating an agent's ability to navigate incomplete information, utilize investigative tools, and make bounded, deterministic decisions.

## 🚀 Built on the OpenEnv framework

- **Stateful Curriculum Learning**  
  Deterministically cycles through Easy, Medium, and Hard applicant profiles.

- **Scaled Dense Rewards**  
  Rewards stay strictly within `(0, 1)` bounds to satisfy Phase 2 validators.

- **Information Asymmetry**  
  Candidate data is masked and must be discovered via tools.

- **Circuit Breakers**  
  Prevents infinite loops (max 10 steps).

---

## 🔍 What Makes This Different

| Feature | Description |
|--------|------------|
| Actions have costs | Tool usage incurs penalty (`-0.01`) |
| Information is gated | Must call tools like `analyze_resume` |
| Missing Data Handling | Use `request_more_info` when needed |

---

## ⚡ Quick Start

```python
from admissions_env import AdmissionsEnv, AdmissionsAction

with AdmissionsEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.task)

    result = env.step(AdmissionsAction(action_type="analyze_resume", action_args={}))
    print(result.observation.applicant.resume_summary)

    result = env.step(AdmissionsAction(action_type="admit", action_args={"reason": "Strong profile"}))
    print(result.reward)
```

---

## 📚 Scenarios & Curriculum

| Level | Candidate | Challenge | Expected Behavior |
|------|----------|----------|------------------|
| Easy | Priya Sharma (CGPA 9.2) | Direct Evaluation | Analyze resume → Admit (~0.79) |
| Medium | Alex Johnson (CGPA 7.5) | Missing Data | Use `request_more_info` |
| Hard | Vikram Singh (CGPA 6.1) | Below Threshold | Reject immediately |

---

## 🎯 Reward Architecture

| Action Type | Reward | Description |
|------------|--------|------------|
| Step Penalty | -0.01 | Prevents stalling |
| Exploration | +0.01 to +0.03 | Encourages tool usage |
| Admit / Reject | 0.75 | Correct decision |
| Failure | 0.10 | Incorrect decision |

---

## ⚙️ Execution Model

- Turn-based system
- State updates after each action
- History tracking for context

---

## 🚢 Deployment

### Hugging Face Spaces
```bash
openenv push --repo-id username/admissions_env
```

### Local Docker
```bash
docker build -t admissions_env:latest -f server/Dockerfile .
docker run -p 8000:8000 admissions_env:latest
```

---

## 📊 Specifications

| Attribute | Value |
|----------|------|
| Action Space | 7 tools |
| Max Steps | 10 |
| API | HTTP/WebSocket |
| Compatibility | Qwen2.5-72B-Instruct |

---

## 🧠 Technical Notes

- Masking logic reveals data via tools
- Handles API failures gracefully
