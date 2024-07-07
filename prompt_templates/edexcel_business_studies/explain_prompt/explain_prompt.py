from langchain.prompts import PromptTemplate
from pathlib import Path
from llm_assessor import construct_prompt_template
import json

# Save Directory
save_dir = Path("./prompt_templates/edexcel_business_studies/explain_prompt")

# Import Examples
with open(save_dir / "explain_few_shot_examples.json", "r") as openfile:
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
4. At the end of your answer, give the number of marks that should be awarded. The Mark Scheme has the total marks to be awarded for the question, do not award more than that amount.
5. Do not award the same point with more than one mark.
6. When awarding marks for justification, ensure that any decision or opinion made is supported with a rationale.
7. When the Answer is referring to the business in question, ensure that the reference is supported from the Context.

Provide a thorough step-by-step explanation of your reasoning.
"""

# Make Prompt Template
prompt_template = construct_prompt_template(instruction=instruction, few_shot_examples=few_shot_examples)

GRADE_ANSWER_PROMPT = PromptTemplate.from_template(
    prompt_template
)

GRADE_ANSWER_PROMPT.save(save_dir / "grade_answer_knowledge_with_justification_prompt.json")
