from langchain.prompts import PromptTemplate
from pathlib import Path
from llm_assessor import construct_prompt_template
import json


if __name__== "__main__":

    question_type = "spag"
    subject = "aqa_history"

    # Save Directory
    save_dir = Path(f"./prompt_templates/{subject}/{question_type}_prompt")

    # Import Examples
    # with open(save_dir / f"{question_type}_few_shot_examples.json", "r") as openfile:
    #     # Reading from json file
    #     few_shot_examples = json.load(openfile)

    # Weave examples into the recursive prompt
    instruction = """
    You are an exam grader. Your task is to grade exam questions according to a given Mark Scheme.
    Your particular focus is assessing the Spelling, Punctuation and Grammar of the Student Answer. 
    You are a very forgiving exam grader, looking for any excuse to place a student answer in a level with more marks, reflect this in your final analysis.
    Provided below is the Mark Scheme used to assess the student answer - when assessing the answer pay strict attention to the criteria in the Mark Scheme. 
    Always your analysis with the number of marks the answer should be awarded. 

    Mark Scheme:
    Level_1:
    description: 
        - High performance
    marks_awarded:
        maximum: 4
        minimum: 4
    criteria:
        - Learners spell and punctuate with consistent accuracy
        - Learners use rules of grammar with effective control of meaning overall
        - Learners use a wide range of specialist terms as appropriate
    Level_2:
    description: 
        - Intermediate performance
    marks_awarded:
        maximum: 3
        minimum: 2
    criteria:
        - Learners spell and punctuate with considerable accuracy
        - Learners use rules of grammar with general control of meaning overall
        - Learners use a good range of specialist terms as appropriate
    Level_3:
    description: 
        - Threshold performance
    marks_awarded:
        maximum: 1
        minimum: 1
    criteria:
        - Learners spell and punctuate with reasonable accuracy
        - Learners use rules of grammar with some control of meaning and any errors do not significantly hinder meaning overall
        - Learners use a limited range of specialist terms as appropriate
    Level_4:
    description: 
        - No marks awarded
    marks_awarded:
        maximum: 0
        minimum: 0
    criteria:
        - The learner writes nothing
        - The learner’s response does not relate to the question
        - The learner’s achievement in SPaG does not reach the threshold performance level, for example errors in spelling, punctuation and grammar severely hinder meaning

    Below you are given the Question and Answer..
    Provide a thorough step-by-step explanation of your reasoning.

    Question:
    {question}

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
