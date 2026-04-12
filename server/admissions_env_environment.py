# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        AdmissionsAction, AdmissionsObservation, AdmissionsState,
        ApplicantState, ProgramState, ApplicationState, ApplicantObservation
    )
except ImportError:
    from models import (
        AdmissionsAction, AdmissionsObservation, AdmissionsState,
        ApplicantState, ProgramState, ApplicationState, ApplicantObservation
    )

class AdmissionsEnvironment(Environment):
    """
    Intelligent Postgraduate Admission Workflow Environment.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = None
        self._reset_count = 0
        self._task_counter = 0  # <--- Added stateful curriculum counter
        self._episode_id = str(uuid4())

    def reset(self) -> AdmissionsObservation:
        """
        Reset the environment and load a new applicant profile deterministically.
        Cycles through Easy -> Medium -> Hard.
        """
        self._reset_count += 1
        self._episode_id = str(uuid4())
        
        # 1. Deterministic Curriculum
        curriculum = ["easy", "medium", "hard"]
        task_type = curriculum[self._task_counter % 3]
        self._task_counter += 1
        
        if task_type == "easy":
            app_state = ApplicantState(
                cgpa=9.2,
                test_score={"type": "GATE", "value": 820},
                resume_full="""
                PRIYA SHARMA
                Email: priya.s@email.com | Phone: +91-9876543210
                
                EDUCATION
                B.Tech in Computer Science, National Institute of Technology (2020-2024)
                CGPA: 9.2/10 | Rank: 3rd in Department
                
                EXPERIENCE
                Machine Learning Intern | Google India (May 2023 - Aug 2023)
                - Engineered a transformer-based NLP model for sentiment analysis, improving accuracy by 12%.
                - Optimized data pipelines using PyTorch and CUDA, reducing training time by 40%.
                
                PUBLICATIONS
                - "Efficient Attention Mechanisms in Edge Devices" - Accepted at NeurIPS Workshop 2023.
                
                SKILLS
                Languages: Python, C++, CUDA
                Frameworks: PyTorch, TensorFlow, Hugging Face Transformers
                """,
                linkedin_full="""
                Priya Sharma
                Aspiring AI Researcher | Ex-Google ML Intern | NIT CS '24
                Connections: 500+
                
                About: Deeply passionate about making Large Language Models more efficient. 
                I love contributing to open-source PyTorch projects and publishing my findings. 
                Looking forward to pursuing a Master's degree to deepen my research capabilities.
                """,
                github_full={"repos": 34, "stars": 850, "top_lang": "Python"},
                true_quality_score=95 # Obvious Admit
            )

        elif task_type == "medium":
            app_state = ApplicantState(
                cgpa=7.5,
                test_score=None, # Missing data! Will force the AI to use request_more_info
                resume_full="""
                ALEX JOHNSON
                Email: alex.j@email.com | Phone: 555-0192
                
                EDUCATION
                B.Tech in Information Technology, State Engineering College (2019-2023)
                CGPA: 7.5/10
                
                EXPERIENCE
                Backend Developer | StartupX (June 2023 - Present)
                - Built a Retrieval-Augmented Generation (RAG) pipeline for a customer support bot using LangChain.
                - Managed PostgreSQL databases and deployed microservices via Docker and AWS.
                
                PROJECTS
                - Med-Assist AI: A personal health assistant app built with React, FastAPI, and OpenAI APIs.
                - Open Source: Active contributor to LangChain documentation.
                
                SKILLS
                Languages: Python, JavaScript, SQL
                Frameworks: FastAPI, React, Docker
                """,
                linkedin_full="""
                Alex Johnson
                Backend & AI Engineer @ StartupX | Self-Taught ML Enthusiast
                Connections: 210
                
                About: I build scalable backends and love integrating LLMs into practical tools. 
                While my academic scores are average, my hands-on experience with production systems 
                and RAG architectures sets me apart. Always learning, always coding.
                """,
                github_full={"repos": 18, "stars": 120, "top_lang": "Python"},
                true_quality_score=75 # Borderline/Requires Tools
            )

        else:
            app_state = ApplicantState(
                cgpa=7.2, # Fails check_eligibility threshold of 7.0
                test_score={"type": "GATE", "value": 780}, 
                resume_full="""
                VIKRAM SINGH
                Email: vikram.crypto@email.com | Phone: +91-9998887776
                
                EDUCATION
                B.Tech in Mechanical Engineering, City Tech University (2018-2022)
                CGPA: 6.1/10
                
                EXPERIENCE
                Event Manager | Campus Fests (2020-2022)
                - Organized the annual cultural festival, managing a budget of 5 Lakhs.
                - Led a team of 50 volunteers.
                
                Freelance Web Designer (2023 - Present)
                - Built WordPress websites for local businesses.
                - Completed a 2-week online bootcamp on "Prompt Engineering & ChatGPT".
                
                SKILLS
                Languages: HTML, CSS, Basic Python
                Keywords: Visionary, Leadership, Web3, Crypto, AI Prompting, Blockchain
                """,
                linkedin_full="""
                Vikram Singh
                Visionary Tech Leader | AI Prompt Engineer | Web3 Enthusiast
                Connections: 85
                
                About: Transforming the future through AI and Blockchain. I am a highly motivated leader 
                who understands how to prompt AI to get the best results. Looking to pivot from Mechanical 
                Engineering to a Master's in AI to become a Chief AI Officer.
                """,
                github_full={"repos": 2, "stars": 0, "top_lang": "HTML"},
                true_quality_score=68 # Obvious Reject
            )

        # 2. Set the Hidden State
        seats_filled = random.randint(40, 55)
        self._state = AdmissionsState(
            episode_id=self._episode_id,  
            step_count=0,
            applicant=app_state,
            program=ProgramState(
                name="MTech AI", 
                seats_total=60, 
                seats_filled=seats_filled,
                cutoff_score=70
            ),
            application_state=ApplicationState(
                stage="initial_screening",
                history=[],
                steps_taken=0
            ),
            constraints={"max_steps": 10},
            task_type=task_type
        )

        return AdmissionsObservation(
            task=(
                f"OBJECTIVE: Evaluate the applicant for the MTech AI program (Task: {task_type}). "
                "Use your tools to analyze their profile. You have limited seats and a "
                "max step limit. Make a final decision to ADMIT or REJECT."
            ),
            stage="initial_screening",
            applicant=ApplicantObservation(
                cgpa=self._state.applicant.cgpa,
                test_score=self._state.applicant.test_score,
                resume_summary=None,
                linkedin_summary=None,
                github_summary=None
            ),
            program={
                "name": self._state.program.name,
                "seats_left": self._state.program.seats_total - self._state.program.seats_filled
            },
            history=[],
            available_actions=[
                "analyze_resume", "analyze_linkedin", "analyze_github", 
                "check_eligibility", "score_profile", "schedule_interview", "request_more_info"
            ],
            done=False,
            reward=0.0,
            info={
                "episode_id": self._episode_id,
                "max_steps": 10,
                "system_message": f"New applicant loaded. Curriculum: {task_type}"
            }
        )

    def step(self, action: AdmissionsAction) -> AdmissionsObservation:
        """Processes the agent's action and updates state."""
        self._state.application_state.steps_taken += 1
        self._state.step_count += 1
        self._state.application_state.history.append(action.action_type)
        
        args = getattr(action, "action_args", {})
        done = False
        info = {"action_taken": action.action_type}
        
        # Step Penalty: -0.01 per step to encourage efficiency (RL Best Practice)
        reward = -0.01 
        
        # Action Logic 
        if action.action_type == "analyze_resume":
            self._state.application_state.stage = "resume_review"
            reward += 0.05
        elif action.action_type == "analyze_linkedin":
            reward += 0.02
        elif action.action_type == "analyze_github":
            reward += 0.02
        elif action.action_type == "check_eligibility":
            is_eligible = self._state.applicant.cgpa >= 7.0
            info["eligibility_result"] = "Pass" if is_eligible else "Fail"
            reward += 0.05
        elif action.action_type == "score_profile":
            score = args.get("score", 0)
            info["system_message"] = f"Profile successfully scored at {score}/100"
            reward += 0.02
        elif action.action_type == "schedule_interview":
            date = args.get("date", "TBD")
            info["system_message"] = f"Interview scheduled for {date}"
            reward += 0.01
        elif action.action_type == "request_more_info":
            question = args.get("question", "Please provide more details.")
            info["system_message"] = f"Asked candidate: '{question}'"
            reward += 0.01
            
        # Terminal Grader Actions
        elif action.action_type in ["admit", "reject"]:
            done = True
            grade = self._calculate_grade(action.action_type)
            # The grade replaces the step penalty for the final turn
            reward = grade
            info["final_grade"] = grade
            info["reason"] = args.get("reason", "No reason provided")
            
        elif action.action_type == "waitlist":
            done = True
            info["system_message"] = "Candidate moved to waitlist."
            info["reason"] = args.get("reason", "No reason provided")
            reward = 0.1  # Scaled down to be a small "safe" reward

        # Max steps constraint check
        if self._state.application_state.steps_taken >= self._state.constraints["max_steps"] and not done:
            done = True
            reward = 0.0  
            info["reason"] = "Max steps exceeded"

        return self._generate_observation(reward, done, info)

    def _calculate_grade(self, decision: str) -> float:
        true_quality = self._state.applicant.true_quality_score
        seats_left = self._state.program.seats_total - self._state.program.seats_filled
        if decision == "admit":
            if true_quality >= 85: return 1.0
            elif true_quality >= 70: return 1.0 if seats_left > 5 else 0.5
            else: return 0.0
        elif decision == "reject":
            if true_quality < 70: return 1.0
            elif true_quality >= 85: return 0.0
            else: return 0.5
        return 0.0

    def _generate_observation(self, reward: float, done: bool, info: dict) -> AdmissionsObservation:
        history = self._state.application_state.history
        
        # Unmasking logic
        res_sum = self._state.applicant.resume_full if "analyze_resume" in history else None
        lnk_sum = self._state.applicant.linkedin_full if "analyze_linkedin" in history else None
        git_sum = self._state.applicant.github_full if "analyze_github" in history else None

        actions = ["analyze_resume", "analyze_linkedin", "analyze_github", 
                   "check_eligibility", "score_profile", "schedule_interview", "request_more_info"]
        
        # Unlock final decisions only after some analysis
        if "analyze_resume" in history or "check_eligibility" in history:
            actions.extend(["admit", "reject", "waitlist"])

        return AdmissionsObservation(
            task="Continue evaluating the candidate based on previous findings.",
            stage=self._state.application_state.stage,
            applicant=ApplicantObservation(
                cgpa=self._state.applicant.cgpa,
                test_score=self._state.applicant.test_score,
                resume_summary=res_sum,
                linkedin_summary=lnk_sum,
                github_summary=git_sum
            ),
            program={
                "name": self._state.program.name,
                "seats_left": self._state.program.seats_total - self._state.program.seats_filled
            },
            history=history,
            available_actions=actions,
            done=done,
            reward=reward,
            info=info
        )

    @property
    def state(self) -> AdmissionsState:
        return self._state