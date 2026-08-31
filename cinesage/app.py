from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel
from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser


# Pydantic Model

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


# Parser

parser = PydanticOutputParser(
    pydantic_object=Movie
)


# Model

model = ChatMistralAI(
    model_name="mistral-small-2506"
)


# Prompt

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Extract movie information from the paragraph.

{format_instructions}
"""
        ),
        (
            "human",
            "{paragraph}"
        )
    ]
)


# Streamlit UI

st.title("🎬 Movie Information Extractor")

st.write(
    "Enter a movie paragraph and extract structured information using Mistral AI."
)


paragraph = st.text_area(
    "Enter movie paragraph:",
    height=250
)


if st.button("Extract Information"):

    if paragraph:

        final_prompt = prompt.invoke(
            {
                "paragraph": paragraph,
                "format_instructions": parser.get_format_instructions()
            }
        )

        response = model.invoke(final_prompt)

        st.subheader("Extracted Movie Details")

        st.write(response.content)

    else:
        st.warning("Please enter a paragraph.")