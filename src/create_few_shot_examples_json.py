import re
import json
import argparse

def parse_markdown(markdown_text):
    # Regex to match headings and their content
    heading_regex = r'(?m)^##\s+(.*?):\s*$'
    
    # Split the markdown text by the headings
    content_sections = re.split(heading_regex, markdown_text)[1:]

    # Save each example into a list
    num_examples = content_sections.count("Question")
    lines_offset = (len(content_sections) // num_examples)
    examples_lst = []

    for j in range(0, num_examples):

        content_dict = {}
        
        for i in range(j * lines_offset, lines_offset * (j+1), 2):
            heading = content_sections[i].strip()
            content = content_sections[i + 1].strip()
            content_dict[heading] = content
        
        examples_lst.append(content_dict)
    
    return examples_lst

if __name__ == "__main__":

    # Run example in terminal
    '''
    python src/create_few_shot_examples_json.py --inputfile ./prompt_templates/aqa_history/judgement_prompt/judgement_few_shot_examples.md --outfile ./prompt_templates/aqa_history/judgement_prompt/judgement_few_shot_examples.json
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile", 
        help="Path to examples txt file",
        type=str
        )
    parser.add_argument(
        "--outfile", 
        help="Path to output, formatted examples json file",
        type=str
        )
    args = parser.parse_args()

    # Check input file format
    file_ext = args.inputfile.split(".")[-1]
    assert file_ext in ["md", "mkd", "mdwn", "mdown", "mdtxt", "mdtext", "markdown", "text"], "Input File must be Markdown"

    # Load File
    with open(args.inputfile, 'r') as file:
        markdown_text = file.read()

    # Partition into examples
    processed_json = parse_markdown(markdown_text)
    few_shot_examples_json = json.dumps(processed_json)

    # Check if every example has the required fields
    fields = ['Question', 'Mark Scheme', 'Context', 'Answer', 'Response']
    #fields = ['Question', 'Mark Scheme', 'Answer', 'Response']
    assert [sum([field in list(x.keys())  for field in fields]) for x in processed_json]  == [len(fields)] * len(processed_json), "All Keys not represented in every few shot example"

    # Check if any examples are duplicated
    assert len([i for n, i in enumerate(processed_json) if i not in processed_json[:n]]) == len(processed_json), "One of more of the examples are repeated"

    # Save examples to JSON
    with open(args.outfile, "w") as outfile:
        outfile.write(few_shot_examples_json)