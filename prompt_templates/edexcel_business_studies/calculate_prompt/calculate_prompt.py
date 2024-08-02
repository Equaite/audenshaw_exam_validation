from langchain.prompts import PromptTemplate
from pathlib import Path
from llm_assessor import construct_prompt_template
import json


if __name__== "__main__":

    question_type = "calculate"
    subject = "edexcel_business_studies"

    # Save Directory
    save_dir = Path(f"./prompt_templates/{subject}/{question_type}_prompt")

    # Import Examples
    # with open(save_dir / f"{question_type}_few_shot_examples.json", "r") as openfile:
    #     # Reading from json file
    #     few_shot_examples = json.load(openfile)

    # Weave examples into the recursive prompt
    instruction = """
    You will be given a Question, Answer, and a Mark Scheme to grade the Answer.
    The Question involves some basic mathematical calculations. You are encouraged to award a mark for the correct working despite the final answer being incorrect.
    The maximum marks you can award are 2.

    Question:
    {question}

    Mark Scheme:
    {mark_scheme}

    Answer:
    {answer}

    Response:
    """.strip()

    # Make Prompt Template
    #prompt_template = construct_prompt_template(instruction=instruction, few_shot_examples=few_shot_examples).strip()
    prompt_template = instruction

    GRADE_ANSWER_PROMPT = PromptTemplate.from_template(
        prompt_template
    )

    GRADE_ANSWER_PROMPT.save(save_dir / f"grade_answer_{subject}_{question_type}_prompt.json")