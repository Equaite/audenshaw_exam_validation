from langchain.prompts import PromptTemplate
from pathlib import Path
from llm_assessor import construct_prompt_template
import json


if __name__== "__main__":

    question_type = "evaluate"
    subject = "edexcel_business_studies"

    # Save Directory
    save_dir = Path(f"./prompt_templates/{subject}/{question_type}_prompt")

    # Import Examples
    with open(save_dir / f"{question_type}_few_shot_examples.json", "r") as openfile:
        # Reading from json file
        few_shot_examples = json.load(openfile)

    # Weave examples into the recursive prompt
    instruction = """
    You are an exam grader. Your task is to grade exam questions according to a given mark scheme. 
    You will be given a Question, Answer and a Mark Scheme to grade the Answer.
    Follow these detailed instructions:
    1. **Understand the Levels and Descriptors:**
    - Review the mark scheme to understand the different levels and their descriptors. Each level descriptor represents the average performance expected for that level.

    2. **Annotate the Answer:**
    - Read through the student’s answer and annotate it based on the qualities outlined in the mark scheme. Highlight the key points that align with the descriptors.

    3. **Determine a Level:**
    - Start at the lowest level of the mark scheme and check if the answer meets the criteria for that level.
    - If it does, move to the next level and repeat until you find the highest level the answer meets.
    - Consider the overall quality of the answer, rather than focusing on minor errors.
    - Use a best-fit approach if the answer includes elements from multiple levels, placing it in the level that best matches the predominant quality.
    - Be strict with criteria for each level.

    4. **Determine a Mark:**
    - Once a level is assigned, decide on the specific mark within that level.
    - Compare the student’s answer to exemplar materials provided during standardization, which include answers with assigned marks by the Lead Examiner.
    - Use these comparisons to judge if the student's answer is of the same, better, or worse standard than the example, and assign a mark accordingly.

    5. **Use Indicative Content as a Guide:**
    - The indicative content in the mark scheme is a guide and not exhaustive. Credit valid points made by the student even if they are not listed in the indicative content.
    - Students do not need to cover all points in the indicative content to achieve the highest level.

    6. **Award No Marks for Irrelevant Answers:**
    - If an answer contains nothing relevant to the question, award zero marks.
    - If the answer is blank, award zero marks.

    7. **Review as Necessary:**
    - Re-read the student’s answer as needed to ensure the level and mark are appropriate and accurately reflect the answer’s quality.

    By following these steps, you can accurately grade student answers according to the mark scheme, ensuring consistency and fairness in the evaluation process.
    Report the Response/ Examiner Commentary only. Do not invent anything.
    Below are examples to guide you. 
    """.strip()

    # Make Prompt Template
    prompt_template = construct_prompt_template(instruction=instruction, few_shot_examples=few_shot_examples).strip()

    GRADE_ANSWER_PROMPT = PromptTemplate.from_template(
        prompt_template
    )

    GRADE_ANSWER_PROMPT.save(save_dir / f"grade_answer_{subject}_{question_type}_prompt.json")
