from client import AdmissionsEnv
from models import AdmissionsAction
import time

def run_full_test():
    print("MTech Admissions Simulator - Initializing...")
    
    # Connect to your local server
    with AdmissionsEnv(base_url="http://127.0.0.1:8080").sync() as env:
        
        # ---------------------------------------------------------
        print("\n=================================================")
        print("  TURN 1: RESETTING ENVIRONMENT (Getting Candidate)")
        print("=================================================")
        result = env.reset()
        
        # We print just the applicant details to keep the terminal clean
        print(result.observation.model_dump_json(indent=2))
        print(f"\nAvailable Actions: {result.observation.available_actions}")
        time.sleep(2)

        # ---------------------------------------------------------
        print("\n=================================================")
        print("  TURN 2: CALLING 'analyze_resume'")
        print("=================================================")
        action_1 = AdmissionsAction(action_type="analyze_resume", action_args={})
        result = env.step(action_1)
        
        # Look! The resume text should now magically appear in the observation!
        print(result.observation.applicant.model_dump_json(indent=2))
        print(f"\nCurrent Reward: {result.reward}")
        time.sleep(2)

        # ---------------------------------------------------------
        print("\n=================================================")
        print("  TURN 3: CALLING 'score_profile' (Passing Context)")
        print("=================================================")
        action_2 = AdmissionsAction(action_type="score_profile", action_args={"score": 88})
        result = env.step(action_2)
        
        # Look at the 'info' dictionary to see if the system accepted our score
        print(f"System Info: {result.observation.info}")
        print(f"Current Reward: {result.reward}")
        time.sleep(2)

        # ---------------------------------------------------------
        print("\n=================================================")
        print("  TURN 4: MAKING FINAL DECISION ('admit')")
        print("=================================================")
        action_3 = AdmissionsAction(action_type="admit", action_args={"reason": "Strong resume and good score!"})
        result = env.step(action_3)
        
        print(f"System Info: {result.observation.info}")
        print(f"\n FINAL REWARD: {result.reward}")
        print(f" IS GAME OVER?: {result.done}")
        print("=================================================\n")

if __name__ == "__main__":
    run_full_test()