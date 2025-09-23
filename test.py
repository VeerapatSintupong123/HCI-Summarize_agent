from smolagents import CodeAgent, InferenceClientModel
from dotenv import load_dotenv
import os
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from langfuse import get_client
load_dotenv()  # โหลด .env เข้ามาเป็น environment variables

langfuse = get_client()
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
SmolagentsInstrumentor().instrument()

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)

model = InferenceClientModel(token=HF_TOKEN)

agent = CodeAgent(tools=[], model=model)
result = agent.run("Calculate the sum of numbers from 1 to 10")
print(result)