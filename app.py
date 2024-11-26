import os
import json
import time
from rich import print
from mistralai.client import MistralClient
from mistralai.models.jobs import TrainingParameters
from mistralai.models.chat_completion import ChatMessage
from mistralai.models.jobs import WandbIntegrationIn

api_key = os.environ.get("MISTRAL_API_KEY")
client = MistralClient(api_key=api_key)

def pprint(obj):
    print(json.dumps(obj.dict(), indent=4))

# 1. upload the dataset
with open("ultrachat_chunk_train.jsonl", "rb") as f:
    ultrachat_chunk_train = client.files.create(file=("ultrachat_chunk_train.jsonl", f))
with open("ultrachat_chunk_eval.jsonl", "rb") as f:
    ultrachat_chunk_eval = client.files.create(file=("ultrachat_chunk_eval.jsonl", f))

print("Data:")
pprint(ultrachat_chunk_train)
pprint(ultrachat_chunk_eval)

# 2. create fine tuning job
created_jobs = client.jobs.create(
    model="open-mistral-7b",
    training_files=[ultrachat_chunk_train.id],
    validation_files=[ultrachat_chunk_eval.id],
    hyperparameters=TrainingParameters(
        training_steps=10,
        learning_rate=0.0001,
    ),
    integrations=[
        WandbIntegrationIn(
            project="test_ft_api",
            run_name="test",
            api_key=os.environ.get("WANDB_API_KEY"),
        ).dict()
    ],
)
print("\nCreated Jobs:")
pprint(created_jobs)

# 3. check the status of the job
print("\nChecking Job Status:")
retrieved_job = client.jobs.retrieve(created_jobs.id)
while retrieved_job.status in ["RUNNING", "QUEUED"]:
    retrieved_job = client.jobs.retrieve(created_jobs.id)
    pprint(retrieved_job)
    print(f"Job is {retrieved_job.status}, waiting 10 seconds")
    time.sleep(10)

jobs = client.jobs.list()
pprint(jobs)

retrieved_jobs = client.jobs.retrieve(created_jobs.id)
pprint(retrieved_jobs)

# 4. use the fine tuned model
chat_response = client.chat(
    model=retrieved_jobs.fine_tuned_model,
    messages=[ChatMessage(role='user', content='What is the best French cheese?')]
)
print("\nTesting Fine Tuned Model:")
pprint(chat_response)
