# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Admissions Env Environment Implementation.

A real-world simulation of a postgraduate university admission system 
where an AI agent processes applications step-by-step.
"""

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
    
    The agent must evaluate an applicant's profile using limited actions 
    (like analyze_resume) and make a final admit/reject decision.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the admissions environment."""
        self._state = None
        self._reset_count = 0
        self._episode_id = str(uuid4())
    
    def reset(self) -> AdmissionsObservation:
        """
        Reset the environment and load a new applicant profile.
        Randomly selects between Easy, Medium, and Hard tasks to test agent robustness.
        """
        self._reset_count += 1
        self._episode_id = str(uuid4())
        
        # 1. Randomly select task difficulty (The "Hidden" Reality)
        task_type = random.choice(["easy", "medium", "hard"])
        
        #  TASK 1: EASY (Obvious high-quality candidate)
        if task_type == "easy":
            app_state = ApplicantState(
                cgpa=9.2,
                test_score={"type": "GATE", "value": 750},
                resume_full="Top university graduate, 2 ML publications, internship at Google.",
                linkedin_full="Active AI researcher with 500+ professional connections.",
                github_full={"repos": 25, "stars": 300, "top_lang": "Python"},
                true_quality_score=95
            )
            
        #  TASK 2: MEDIUM (Missing test score, requires checking GitHub/Resume)
        elif task_type == "medium":
            app_state = ApplicantState(
                cgpa=7.5,
                test_score=None, 
                resume_full="Decent state college, worked on 3 solid open-source AI projects.",
                linkedin_full="Software Developer at a mid-sized firm, 150 connections.",
                github_full={"repos": 12, "stars": 50, "top_lang": "C++"},
                true_quality_score=75
            )
            
        #  TASK 3: HARD (Conflicting profile: weak CGPA, extremely high test score)
        else:
            app_state = ApplicantState(
                cgpa=6.1,
                test_score={"type": "GATE", "value": 780},
                resume_full="Poor academic record but won a national robotics hackathon.",
                linkedin_full="Robotics enthusiast, few professional connections.",
                github_full={"repos": 4, "stars": 5, "top_lang": "C"},
                true_quality_score=68
            )

        # 2. Set the Hidden State (The "Referee's" Knowledge)
        seats_filled = random.randint(40, 55) # Adds variability to seat constraints [cite: 35]
        
        self._state = AdmissionsState(
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

        # 3. Define the Agent's Objective
        task_instruction = (
            "OBJECTIVE: Evaluate the applicant for the MTech AI program. "
            "Use your tools to analyze their profile. You have limited seats and a "
            "max step limit. Make a final decision to ADMIT or REJECT."
        )

        # 4. Generate the Initial Observation (What the AI actually sees) 
        initial_obs = AdmissionsObservation(
            task=task_instruction,
            stage=self._state.application_state.stage,
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
            history=self._state.application_state.history,
            available_actions=[
                "analyze_resume", 
                "analyze_linkedin", 
                "analyze_github", 
                "check_eligibility",
                "request_more_info"
            ],
            done=False,
            reward=0.0,
            info={
                "episode_id": self._episode_id,
                "max_steps": 10
            }
        )
        
        return initial_obs

   

    def step(self, action: AdmissionsAction) -> AdmissionsObservation:
        """
        Processes the agent's action, updates state, and returns the next observation.
        """
        # 1. Update counters and history
        self._state.application_state.steps_taken += 1
        self._state.application_state.history.append(action.action_type)
        
        # Initialize default values for this step
        reward = 0.0
        done = False
        info = {"action_taken": action.action_type}
        
        # 2. Handle Action Logic
        # PROFILE ANALYSIS ACTIONS
        if action.action_type == "analyze_resume":
            self._state.application_state.stage = "resume_review"
            reward = 2.0  # Reward for information gathering [cite: 13]
            
        elif action.action_type == "analyze_linkedin":
            reward = 1.0
            
        elif action.action_type == "analyze_github":
            reward = 1.0

        # EVALUATION ACTIONS
        elif action.action_type == "check_eligibility":
            # Logical check: is CGPA above a hidden threshold?
            is_eligible = self._state.applicant.cgpa >= 7.0
            info["eligibility_result"] = "Pass" if is_eligible else "Fail"
            reward = 2.0

        # TERMINAL ACTIONS (The Graders)
        elif action.action_type in ["admit", "reject"]:
            done = True
            # Call the deterministic grader [cite: 9, 21]
            grade = self._calculate_grade(action.action_type)
            
            # Map 0.0-1.0 grade to a large reward/penalty [cite: 13]
            if grade == 1.0:
                reward = 100.0
            elif grade == 0.5:
                reward = 20.0
            else:
                reward = -80.0
            
            info["final_grade"] = grade

        # 3. Check Constraints (Max Steps)
        if self._state.application_state.steps_taken >= self._state.constraints["max_steps"] and not done:
            done = True
            reward = -10.0  # Penalty for timing out [cite: 13]
            info["reason"] = "Max steps exceeded"

        # 4. Construct the next Observation
        return self._generate_observation(reward, done, info)
        return AdmissionsObservation(
            stage=self._state.application_state.stage,
            applicant=ApplicantObservation(cgpa=0.0, test_score={}),
            program={},
            history=[],
            available_actions=[],
            done=False,
            reward=0.0
        )
    #The Deterministic Grader(Helper)
    def _calculate_grade(self, decision: str) -> float:
        """
        Deterministic grader based on true_quality_score and program constraints.
        """
        true_quality = self._state.applicant.true_quality_score
        seats_left = self._state.program.seats_total - self._state.program.seats_filled
        
        if decision == "admit":
            # High quality candidate
            if true_quality >= 85:
                return 1.0
            # Good candidate, but seats are tight
            elif true_quality >= 70:
                return 1.0 if seats_left > 5 else 0.5
            # Poor quality candidate
            else:
                return 0.0
                
        elif decision == "reject":
            # Correctly rejected a weak candidate
            if true_quality < 70:
                return 1.0
            # Incorrectly rejected a genius
            elif true_quality >= 85:
                return 0.0
            # Rejected a decent candidate (safe but not optimal)
            else:
                return 0.5
        
        return 0.0

    #The Observation Generator (Helper)
    def _generate_observation(self, reward: float, done: bool, info: dict) -> AdmissionsObservation:
        """
        Helper to reveal state data based on the agent's history.
        """
        history = self._state.application_state.history
        
        # Dynamic visibility: Only show summaries if the tool was used
        res_sum = self._state.applicant.resume_full if "analyze_resume" in history else None
        lnk_sum = self._state.applicant.linkedin_full if "analyze_linkedin" in history else None
        git_sum = self._state.applicant.github_full if "analyze_github" in history else None

        # Dynamic Actions: Only allow Admit/Reject after some analysis is done
        actions = ["analyze_resume", "analyze_linkedin", "analyze_github", "check_eligibility"]
        if "analyze_resume" in history or "check_eligibility" in history:
            actions.extend(["admit", "reject"])
        
        return AdmissionsObservation(
            task="Continue evaluating the candidate.",
            stage=self._state.application_state.stage,
            applicant=ApplicantObservation(
                cgpa=self._state.applicant.cgpa,
                test_score=self._state.applicant.test_score,
                resume_summary=res_sum,
                linkedin_summary=lnk_sum,
                github_summary=git_sum
            ),
            program={
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
        """Returns the complete, hidden ground truth."""
        return self._state