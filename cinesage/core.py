from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
class Movie(BaseModel):
    title:str
    release_year:Optional[int]
    genre:List[str]
    director:Optional[str]
    cast:List[str]
    rating:Optional[float]
    summary:str

parser =PydanticOutputParser(pydantic_object=Movie)

model=ChatMistralAI(model_name="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages([
    ('system',
     """Extract movie information fromm th paragraph{format_instructions}"""
    ),
    ('human',"{paragraph}")
])
para =input("give your paragraph:")
final_prompt=prompt.invoke({
    "paragraph": para,
'format_instructions': parser.get_format_instructions()
})
response=model.invoke(final_prompt)
movie_data=parser.parse(response.content)
print(movie_data)