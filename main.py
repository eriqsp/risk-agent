from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools import (
    calculate_portfolio_volatility,
    calculate_asset_correlations,
    stress_test
)

load_dotenv()

model = ChatOpenAI(
    model="gpt-5.6-luna",
    use_responses_api=True
)

agent = create_agent(
    model=model,
    tools=[
        calculate_portfolio_volatility,
        calculate_asset_correlations,
        stress_test
    ],
    system_prompt="""
    You are a financial risk analyst.
    Use the available tools to perform financial calculations.
    """
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "What happens to my portfolio if "
                "AAPL falls 20%, MSFT falls 10%, "
                "and GOOG falls 15%?"
            )
        }
    ]
})

final_message = response["messages"][-1]

print(final_message.content[0]["text"])