import subprocess
import os

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def run_command(cmd, description=None):
    if description:
        print(f"\n> {description}")
    print(f"   $ {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"Warning: {result.stderr}")
    return result.returncode == 0

def main():
    print_header("Emotion Classification Research Project")
    print("   1. BiGRU Training")
    print("   2. DistilBERT + LoRA Training")
    print("   3. Model Evaluation")
    print("   4. FastAPI Deployment")
    
    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Step 1: Train BiGRU
    print_header("STEP 1: Training BiGRU Model")
    run_command("python -m src.train_bigru", "Training BiGRU")
    
    # Step 2: Train DistilBERT
    print_header("STEP 2: Training DistilBERT + LoRA")
    run_command("python -m src.train_distilbert", "Training DistilBERT")
    
    # Step 3: Evaluate
    print_header("STEP 3: Evaluating Models")
    run_command("python -m src.evaluate", "Model Evaluation")
    
    # Step 4: Start FastAPI
    print_header("STEP 4: Starting FastAPI Server")
    print("API: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    
    print("\n" + "=" * 70)
    print("Project Complete!")
    print("   API: http://localhost:8000")
    print("=" * 70)
    
    run_command("python -m src.deploy", "Start FastAPI Server")

if __name__ == "__main__":
    main()