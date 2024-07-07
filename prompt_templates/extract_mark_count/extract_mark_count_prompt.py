
from langchain.prompts import PromptTemplate
from pathlib import Path

# Save Directory
save_dir = Path("./prompt_templates/extract_mark_count")


# Extract Marks Awarded
mark_count_template = """
    Instruction:
    You will be given some text in which there is a mention of marks awarded. You will return how many marks should be awarded given the answer.
    Remember, you want to give the exact number of marks to award. If it's a range of marks, give the maximum of the range. 
    E.g. 10-14 marks would be 14.
    Return only the number of marks, no additional text.  
    E.g. 0 marks should return 0, 1 mark should return 1, 2 marks should return 2 and so on.
    If there is no number mentioned in the answer then consider the following cases:
    - if the term 'full marks' or any variant denoting the same meaning is mentioned, return 'fm'
    - if the term 'no marks' or any variant denoting the same meaning is mentioned, return 0
    - if there are no explicit number of marks mentioned, return 'na'

    Here are some examples to direct you:
    
    Answer: The student answer does not successfully address the question asked. \nBased on the mark scheme, the answer falls into Level 0 and should therefore be awarded 0 marks.
    Awarded Marks: 0

    Answer: The student answer provides a detailed analysis based on the context provided. The answer correctly mentions that regardless of the number of stores, all doughnuts are made in one of the 13 larger stores. This point is substantiated by the observation that 'the same staff will be making these products to the original recipe with only a few new staff requiring training'. The answer also provides a second point of analysis, mentioning that Krispy Kreme can deliver high quality training to new staff to maintain the quality of the doughnuts as the business grows. \n\nAll business areas are fully analysed and the student applies knowledge and understanding to the context sufficiently.\n\nBased on the mark scheme, the answer falls into Level 3 and should therefore be awarded 6 marks.
    Awarded Marks: 6

    Answer: The student answer provides a sound analysis based on the context provided. The student correctly defines price skimming and applies it to the context of Apple's iPhone 6s. The student also correctly mentions that the high initial price targets early adopters and that the price decreases when the next iPhone is released. The student also correctly identifies a potential issue with this strategy, namely that global sales are slowing down and therefore the high initial price may not be affecting demand as much as it used to.\n\nBased on the mark scheme, the answer falls into Level 2 and should therefore be awarded 4 marks.
    Awarded Marks: 4

    Answer: The student answer provides a detailed analysis based on the context provided. The student correctly mentions that Apple can charge a high price for the iPhone 6s due to high demand and customer loyalty. The student also correctly mentions that the new 3D touch technology, which is unique to the iPhone, will attract customers who want the latest technology. The student then correctly concludes that this will increase Apple's sales revenue, as a high number of people will be buying the new phone at a high price until the next version is released.\n\nAll business areas are fully analysed and the student applies knowledge and understanding to the context sufficiently.\n\nBased on the mark scheme, the answer falls into Level 3 and should therefore be awarded full marks.
    Awarded Marks: fm

    Answer: The student answer does not successfully address the question asked. The student only provides one reason for creating a business plan and does not provide an explanation for why this is important. \n\nBased on the mark scheme, the answer should not be awarded any marks.
    Awarded Marks: 0

    Answer: The student answer provides a developed integrated analysis and evaluation of topics with sustained judgement based on context. The student provides a coherent, relevant line of reasoning with a conclusion that is fully justified. The student fully analyses the interdependent nature of business areas and applies knowledge and understanding to the context successfully drawing together several functional areas of business. \n\nThe student correctly identifies that flow production would be beneficial for PDL due to the standardised nature of the products until the finishing point. The student also correctly identifies that flow production would reduce lead time and labour costs, and increase customer satisfaction and profit. The student also correctly identifies that the initial cost of new machinery and training would be high, but that revenue would increase in the long term. \n\nIn terms of staff motivation, the student correctly identifies that the workers may feel demotivated due to feeling less responsible for the products and the potential for job loss. However, the student also correctly identifies that some workers may feel more valued due to the training they would receive.
    Awarded Marks: na

    Answer: The answer successfully identifies two relevant ways SBG can involve its employees in decision-making. The first way mentioned is 'developing a democratic leadership style'. This is a valid way of involving employees in decision-making and is mentioned in the example answers section of the mark scheme. This should be awarded 1 mark for identification of a relevant way. The second way mentioned is 'allowing employees to form quality circles'. This is also a valid way of involving employees in decision-making and is similar to the 'encourage team working' point mentioned in the example answers section. This should be awarded 1 mark for identification of a relevant way. The answer does not make any relevant references to SBG when discussing the ways to involve employees in decision-making. The points made are general and could be applied to any business. Therefore, the answer should be awarded 0 marks for each relevant reference made to this business. The answer provides relevant explanations for both ways mentioned. For the democratic leadership style, it is mentioned that 'employees would be able to give their views during decision-making and their input would contribute to the final decision made by the manager'. This is a correct explanation of how a democratic leadership style would involve employees in decision-making. This should be awarded 1 mark for a relevant explanation. For the quality circles, it is mentioned that 'employees meet together to discuss business related issues' and 'the feedback is presented to managers so that it is used in making future decisions'. This is a correct explanation of how quality circles would involve employees in decision-making. This should be awarded 1 mark for a relevant explanation. Overall, this answer should be awarded 4 out of a total 6 marks.
    Awarded Marks: 4

    Answer: The answer successfully identifies one relevant way SBG can involve its employees in decision-making by suggesting 'having meetings in small groups/department'. This is a valid method of involving employees in decision-making and aligns with the 'meet with employees' point in the example answers section of the mark scheme. This should be awarded 1 mark for identification of a relevant way. However, the answer does not provide a second way for SBG to involve its employees in decision-making, so it should be awarded 0 marks for the second identification of a relevant way. The answer makes a relevant reference to SBG when discussing the first way by mentioning 'each at the 6 factories'. This is a relevant reference to the business as it is mentioned in the context that SBG has 6 factories. This should be awarded 1 mark for a relevant reference made to this business. The answer provides an explanation for the first way by stating 'In meetings especially if they are small, employees would feel comfortable to give their input in decision making'. This is a valid explanation as it explains why small group meetings can facilitate employee involvement in decision-making. This should be awarded 1 mark for a relevant explanation. Overall, this answer should be awarded 3 out of a total 6 marks.
    Awarded Marks: 3

    Answer: {answer}
    Awarded Marks:"""

MARK_COUNT_PROMPT = PromptTemplate.from_template(
    mark_count_template
    )

MARK_COUNT_PROMPT.save(save_dir/ "extract_mark_count_prompt.json")