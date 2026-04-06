# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data models for the Intelligent Postgraduate Admission Workflow Environment.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation

# ==========================================
# 1. STATE (The Hidden Ground Truth)
# ==========================================
class ApplicantState(BaseModel):
    cgpa: float
    test_score: Optional[Dict[str, Any]] = None
    resume_full: str
    linkedin_full: str
    github_full: Dict[str, Any]
    true_quality_score: int

class ProgramState(BaseModel):
    name: str
    seats_total: int
    seats_filled: int
    cutoff_score: int

class ApplicationState(BaseModel):
    stage: str
    history: List[str]
    steps_taken: int

class AdmissionsState(BaseModel):
    """The absolute truth of the simulation. The agent never sees this directly."""
    applicant: ApplicantState
    program: ProgramState
    application_state: ApplicationState
    constraints: Dict[str, int] = Field(default_factory=lambda: {"max_steps": 10})
    task_type: str

# ==========================================
# 2. OBSERVATION (What the Agent Actually Sees)
# ==========================================
class ApplicantObservation(BaseModel):
    cgpa: float
    test_score: Optional[Dict[str, Any]] = None
    # Summaries start as None until the agent uses tools to reveal them
    resume_summary: Optional[str] = None
    linkedin_summary: Optional[str] = None
    github_summary: Optional[Dict[str, Any]] = None

class AdmissionsObservation(Observation):
    """The limited data fed to the agent at each step."""
    task: str = Field(..., description="Instructions for the agent")
    stage: str = Field(..., description="Current stage of the application workflow")
    applicant: ApplicantObservation = Field(..., description="Visible details of the applicant")
    program: Dict[str, Any] = Field(..., description="Program details, e.g., {'name': 'MTech AI', 'seats_left': 12}")
    history: List[str] = Field(default_factory=list, description="Log of actions taken so far")
    available_actions: List[str] = Field(..., description="List of valid tool calls the agent can make")
    info: Dict[str, Any] = Field(default_factory=dict)

# ==========================================
# 3. ACTION (The Agent's Tool Calls)
# ==========================================
class AdmissionsAction(Action):
    """The structured JSON the LLM must return to take a step."""
    
    # We use Literal to force the LLM to pick exactly one of these strings.
    action_type: Literal[
        # Profile Analysis
        "analyze_resume", 
        "analyze_linkedin", 
        "analyze_github", 
        
        # Evaluation
        "score_profile",
        "check_eligibility",
        
        # Workflow
        "send_to_review",
        "schedule_interview",
        "waitlist",
        "admit", 
        "reject",
        
        # Utility
        "request_more_info",
        "revise_decision"
    ] = Field(..., description="The tool to use.")
    
    action_args: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Arguments for the action, e.g., {'reason': 'low cgpa'} for reject"
    )