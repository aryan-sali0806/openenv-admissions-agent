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
        self._episode_id = str(uuid4())

    def reset(self) -> AdmissionsObservation:
        """
        Reset the environment and load a new applicant profile.
        Returns a hardcoded 'Turn 1' observation.
        """
        self._reset_count += 1
        self._episode_id = str(uuid4())
        
        # 1. Randomly select task difficulty (Internal State)
        task_type = random.choice(["easy", "medium", "hard"])
        
        if task_type == "easy":
            app_state = ApplicantState(
                cgpa=9.2,
                test_score={"type": "GATE", "value": 750},
                resume_full="Top university graduate, 2 ML publications, internship at Google.",
                linkedin_full="Active AI researcher with 500+ professional connections.",
                github_full={"repos": 25, "stars": 300, "top_lang": "Python"},
                true_quality_score=95
            )
        elif task_type == "medium":
            app_state = ApplicantState(
                cgpa=7.5,
                test_score=None, 
                resume_full="Decent state college, worked on 3 solid open-source AI projects.",
                linkedin_full="Software Developer at a mid-sized firm, 150 connections.",
                github_full={"repos": 12, "stars": 50, "top_lang": "C++"},
                true_quality_score=75
            )
        else:
            app_state = ApplicantState(
                cgpa=6.1,
                test_score={"type": "GATE", "value": 780},
                resume_full="Poor academic record but won a national robotics hackathon.",
                linkedin_full="Robotics enthusiast, few professional connections.",
                github_full={"repos": 4, "stars": 5, "top_lang": "C"},
                true_quality_score=68
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

        # 3. THE RESET OBSERVATION (Manual Return)
        # This ensures the AI sees the OBJECTIVE and FULL GATE SCORE on Turn 1.
        return AdmissionsObservation(
            task=(
                "OBJECTIVE: Evaluate the applicant for the MTech AI program. "
                "Use your tools to analyze their profile. You have limited seats and a "
                "max step limit. Make a final decision to ADMIT or REJECT."
            ),
            stage="initial_screening",
            applicant=ApplicantObservation(
                cgpa=self._state.applicant.cgpa,
                test_score=self._state.applicant.test_score, # Full score shown!
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
                "system_message": "New applicant loaded. Initial screening stage."
            }
        )

    def step(self, action: AdmissionsAction) -> AdmissionsObservation:
        """Processes the agent's action and updates state."""
        self._state.application_state.steps_taken += 1
        self._state.step_count += 1
        self._state.application_state.history.append(action.action_type)
        
        args = getattr(action, "action_args", {})
        reward = 0.0
        done = False
        info = {"action_taken": action.action_type}
        
        # Action Logic (Rewards strictly bounded between 0.0 and 1.0)
        if action.action_type == "analyze_resume":
            self._state.application_state.stage = "resume_review"
            reward = 0.05
        elif action.action_type == "analyze_linkedin":
            reward = 0.02
        elif action.action_type == "analyze_github":
            reward = 0.02
        elif action.action_type == "check_eligibility":
            is_eligible = self._state.applicant.cgpa >= 7.0
            info["eligibility_result"] = "Pass" if is_eligible else "Fail"
            reward = 0.05
        elif action.action_type == "score_profile":
            score = args.get("score", 0)
            info["system_message"] = f"Profile successfully scored at {score}/100"
            reward = 0.02
        elif action.action_type == "schedule_interview":
            date = args.get("date", "TBD")
            info["system_message"] = f"Interview scheduled for {date}"
            reward = 0.01
        elif action.action_type == "request_more_info":
            question = args.get("question", "Please provide more details.")
            info["system_message"] = f"Asked candidate: '{question}'"
            reward = 0.01
            
        # Terminal Grader Actions
        elif action.action_type in ["admit", "reject"]:
            done = True
            grade = self._calculate_grade(action.action_type)
            # The grade is already exactly 1.0, 0.5, or 0.0. Perfect!
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
            reward = 0.0  # Kept at 0.0 minimum bound instead of negative
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
        """
        Helper for the STEP function only. 
        Handles visibility of summaries based on history.
        """
        history = self._state.application_state.history
        
        # Unmasking logic
        res_sum = self._state.applicant.resume_full if "analyze_resume" in history else None
        lnk_sum = self._state.applicant.linkedin_full if "analyze_linkedin" in history else None
        git_sum = self._state.applicant.github_full if "analyze_github" in history else None

        # Logic to update available actions
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