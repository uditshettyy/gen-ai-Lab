from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# Model
model = ChatMistralAI(
    model_name="mistral-small-2506"
)


# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
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
"""
    ),
    (
        "human",
        """
Extract useful information from this paragraph:

{paragraph}
"""
    )
])


# Streamlit UI

st.title("🎬 CineSage - Information Extractor")

st.write(
    "Enter a movie paragraph and extract useful information using AI."
)


paragraph = st.text_area(
    "Enter your paragraph:",
    height=250
)


if st.button("Extract Information"):

    if paragraph:

        final_prompt = prompt.invoke(
            {
                "paragraph": paragraph
            }
        )

        response = model.invoke(final_prompt)

        st.subheader("Extracted Information")

        st.write(response.content)

    else:
        st.warning("Please enter a paragraph.")