from langchain.prompts import PromptTemplate
from pathlib import Path
from llm_assessor import construct_prompt_template
import json


if __name__== "__main__":

    question_type = "explain"
    subject = "edexcel_business_studies"

    # Save Directory
    save_dir = Path(f"./prompt_templates/{subject}/{question_type}_prompt")

    # Import Examples
    with open(save_dir / f"{question_type}_few_shot_examples.json", "r") as openfile:
        # Reading from json file
        few_shot_examples = json.load(openfile)

    # Weave examples into the recursive prompt
    instruction = """
    You will be given a Question, Answer, Context and a Mark Scheme to grade the Answer.
    For each bulleted marking criteria in the Mark Scheme, mention if the Answer successfully answers the Question and fulfills the criteria in the Mark Scheme, using evidence from the answer.
    The Mark Scheme has example answers that you can use to guide how to award marks.
    The Mark Scheme has marking notes which gives additional information for grading the answer.

    Here are some strict guidelines to abide by:
    1. Quote the answer as much as possible and accurately, do not invent anything.
    2. Refer to the mark scheme as often as possible, justify any of your remarks with reference to the mark scheme.
    3. Remark on each criteria sentence in the mark scheme and mention if the answer satisfies the criteria.
    4. Use Example Answers as a Guide. The example answers in the mark scheme is a guide and not exhaustive. Credit valid points made by the student even if they are not listed in the example answers.
    5. At the end of your answer, give the number of marks that should be awarded. The Mark Scheme has the total marks to be awarded for the question, do not award more than that amount.
    6. Do not award the same point with more than one mark.
    7. When awarding marks for justification, ensure that any decision or opinion made is supported with a rationale.
    8. When the Answer is referring to the business in question, ensure that the reference is supported from the Context.

    Report the Response/ Examiner Commentary only. Do not invent anything.
    Below are examples to guide you. 
    """.strip()

    # Make Prompt Template
    prompt_template = construct_prompt_template(instruction=instruction, few_shot_examples=few_shot_examples).strip()

    GRADE_ANSWER_PROMPT = PromptTemplate.from_template(
        prompt_template
    )

    GRADE_ANSWER_PROMPT.save(save_dir / f"grade_answer_{subject}_{question_type}_prompt.json")
