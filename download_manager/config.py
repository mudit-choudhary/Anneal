# --- Paths ---
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent.parent / "data")
PDF_DIR = BASE_DIR + "/raw_pdfs"



# --- AWS Bedrock Settings ---
# Ensure your EC2 instance has an IAM role with Bedrock Access, 
# or set AWS_ACCESS_KEY_ID / SECRET in env vars.
BEDROCK_REGION = "us-east-1"  # Claude 3.5 Sonnet is available here
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0" 

# --- Domains to Download ---
DOMAINS = [
    "Generative AI",
    "Large Language Models",
    "Retrieval Augmented Generation",
    "Machine Learning",
    "Quantum Artificial Intelligence",
    "Neural Networks",
    "Computer Vision",
    "Graph Neural Networks",
    "Deep Neural Networks",
    "RAG"
]

REGISTRY_URL = 'http://127.0.0.1:4000'