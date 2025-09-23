from smolagents import CodeAgent, InferenceClientModel
from dotenv import load_dotenv
import os

load_dotenv()

model = InferenceClientModel(token=os.environ.get('HUGGINGFACEHUB_API_TOKEN'))

agent = CodeAgent(tools=[], model=model)

result = agent.run("Calculate the sum of numbers from 1 to 10")
print(result)