from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from data.portfolio import load_portfolio_history
from agent.tools import create_tools
from agent.agent import create_risk_agent


load_dotenv()


portfolio_history = load_portfolio_history("data/portfolio.csv")
tools = create_tools(portfolio_history)
agent = create_risk_agent(tools)


messages = []
while True:
    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    messages.append(HumanMessage(content=question))

    result = agent.invoke({
        "messages": messages
    })

    messages = result["messages"]

    response = messages[-1].content

    print("\nAgent:")

    if isinstance(response, str):
        print(response)

    elif isinstance(response, list):
        for block in response:
            if isinstance(block, dict) and block.get("type") == "text":
                print(block["text"])
