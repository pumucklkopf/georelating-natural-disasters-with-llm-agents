from langchain_core.prompts import FewShotPromptWithTemplates, PromptTemplate, StringPromptTemplate

examples = []
example_prompt = PromptTemplate.from_template(
    """
    News Article Heading: {heading}\n
    News Article: {news_article}\n
    Toponym List: {toponym_list}\n
"""

)

few_shot_template = FewShotPromptWithTemplates( # todo nothing works yet
    examples=examples,
    example_selector=None, #todo: dynamically select examples based on the context with BM25 or other methods
    example_prompt=example_prompt,
    prefix={"Here are a few examples of how to provide the search arguments for a given news article. \n"},
    suffix={"Make sure the search arguments are as specific as possible, so that the search returns only a few, highly relevant results."},
    example_separator=lambda index: f"\n example {index + 1} \n",
    validate_template=False
)
print("test")
