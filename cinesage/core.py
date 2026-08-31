from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
model=ChatMistralAI(model_name="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages([
    ("system",
            """
You are an expert information extraction assistant.

Your job is to extract useful and relevant information from the given text.

Extract the following information:
- Title
- Genre
- Director
- Cast / Main Characters
- Release Date (if mentioned)
- Main Topic
- Setting / Location
- Storyline / Plot
- Important Events
- Key Concepts
- Themes and Messages
- Overall Summary
- Keywords

Rules:
- Extract only information present in the given text.
- If any information is missing, mention "Not mentioned".
- Do not make assumptions or add outside knowledge.
- Keep the response clear and well organized.
"""),("human",
            """
Extract useful information from this paragraph:

{paragraph}
""")]
)
para =input("give your paragraph:")
final_prompt=prompt.invoke({
    "paragraph": para
})
response=model.invoke(final_prompt)
print(response.content)