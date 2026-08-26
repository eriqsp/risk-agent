from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from portfolio import load_portfolio
from tools import create_tools


load_dotenv()


portfolio = load_portfolio('portfolio.txt')
tools = create_tools(portfolio)


model = ChatOpenAI(
    model="gpt-5.6-luna",
    use_responses_api=True,
    reasoning_effort="none"
)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
    You are an AI portfolio risk analyst.

    Your job is to analyze the user's portfolio using the
    available quantitative tools.

    Rules:

    1. Use the available tools whenever a calculation is required.
    2. Never invent financial data or numerical results.
    3. Do not perform complex financial calculations yourself 
    when a tool is available for that calculation.
    4. Clearly distinguish between historical analysis and 
    hypothetical stress tests.
    5. Explain the results in a concise and professional way.
    6. If the available tools cannot answer a question reliably, 
    say so rather than guessing.
    7. When reporting percentages, make clear what the percentage 
    represents.
    """
)


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