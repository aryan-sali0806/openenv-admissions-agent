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
        
        # 🟢 TASK 1: EASY (Obvious high-quality candidate)
        if task_type == "easy":
            app_state = ApplicantState(
                cgpa=9.2,
                test_score={"type": "GATE", "value": 750},
                resume_full="Top university graduate, 2 ML publications, internship at Google.",
                linkedin_full="Active AI researcher with 500+ professional connections.",
                github_full={"repos": 25, "stars": 300, "top_lang": "Python"},
                true_quality_score=95
            )
            
        # 🟡 TASK 2: MEDIUM (Missing test score, requires checking GitHub/Resume)
        elif task_type == "medium":
            app_state = ApplicantState(
                cgpa=7.5,
                test_score=None, 
                resume_full="Decent state college, worked on 3 solid open-source AI projects.",
                linkedin_full="Software Developer at a mid-sized firm, 150 connections.",
                github_full={"repos": 12, "stars": 50, "top_lang": "C++"},
                true_quality_score=75
            )
            
        # 🔴 TASK 3: HARD (Conflicting profile: weak CGPA, extremely high test score)
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

        # 4. Generate the Initial Observation (What the AI actually sees) [cite: 38]
        # NOTE: Summaries are None to force the agent to use tool-calls! 
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
        Execute a step in the environment.
        (Logic to be implemented in the next step!)
        """
        # --- SKELETON CODE ---
        # 1. Deduct 1 from max_steps
        # 2. Check which action_type was called
        # 3. Update the observation (e.g., reveal resume summary)
        # 4. Calculate partial reward
        # 5. Return new AdmissionsObservation
        
        # Temporary fallback to keep the compiler happy until we write the logic
        return AdmissionsObservation(
            stage=self._state.application_state.stage,
            applicant=ApplicantObservation(cgpa=0.0, test_score={}),
            program={},
            history=[],
            available_actions=[],
            done=False,
            reward=0.0
        )

    @property
    def state(self) -> AdmissionsState:
        """Returns the complete, hidden ground truth."""
        return self._state