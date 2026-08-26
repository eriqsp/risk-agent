from dotenv import load_dotenv
from data.portfolio import load_portfolio
from agent.tools import create_tools
from agent.agent import create_risk_agent


load_dotenv()


portfolio = load_portfolio('data/portfolio.txt')
tools = create_tools(portfolio)
agent = create_risk_agent(tools)


# using the concept of context windows
messages = []
while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    messages.append({
        "role": "user",
        "content": question
    })

    result = agent.invoke({
        "messages": messages
    })

    print("\nAgent:")

    messages = result["messages"]
    message = messages[-1].content
    try:
        print(message[0]['text'])
    except KeyError:
        print(message[1]['text'])