from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel()

agent = CodeAgent(tools=[], model=model)

result = agent.run("Calculate the sum of numbers from 1 to 10")
print(result)