from dotenv import load_dotenv  
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage


model=ChatMistralAI(model_name="mistral-small-2506")


message=[
    SystemMessage(content="You are a funnny AI Agent."),

]
print("----------Welcome type 0 to exit the application----------")

while True:
    prompt=input("You:")
    message.append(HumanMessage(content=prompt))
    if prompt == "0":
        break
    response=model.invoke(message)
    message.append(AIMessage(content=response.content))
    print("Bot:",response.content)

