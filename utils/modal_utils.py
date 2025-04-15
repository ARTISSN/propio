# scripts/modal_utils.py
import modal

stub = modal.Stub("face-swap-sdxl-pipeline")

@stub.function(gpu="A100")
def run_remote_pipeline():
    import subprocess
    subprocess.run(["python", "scripts/extract_embeddings.py"])
    subprocess.run(["python", "scripts/train_lora.py"])
    subprocess.run(["python", "scripts/inference.py"])

def run_modal_pipeline():
    print("Dispatching pipeline to Modal A100 instance...")
    stub.deploy()
    stub.run_remote_pipeline.call()
