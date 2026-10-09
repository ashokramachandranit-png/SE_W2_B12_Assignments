
from langchain_openai import ChatOpenAI

# Initialize the OpenAI model through LangChain
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Send a prompt to OpenAI
prompt1 = "Explain MCP in 3 simple bullet points."
prompt2 = "Explain RAG in 3 simple bullet points, but make it more detailed."


response1 = llm.invoke(prompt1)
response2 = llm.invoke(prompt2)

# Display the response
print("\nAI Response:")
print(response1.content)
print(response2.content)