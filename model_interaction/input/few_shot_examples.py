import json
from typing import List, Dict
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from data_handler.xml_parsing import XMLDataHandler

class FewShotExampleHandler:
    FEW_SHOT_EXAMPLE_PATH = 'data/few_shot_examples.json'

    def __init__(self, data_directory: str, xml_file: str):
        self.data_handler = XMLDataHandler(data_directory)
        self.data_handler.parse_xml(xml_file)
        self.examples = self._load_examples()

    def _load_ground_truth(self, few_shot_example_path: str = FEW_SHOT_EXAMPLE_PATH) -> List[Dict]:
        with open(few_shot_example_path, 'r') as f:
            few_shot_examples = json.load(f)
        return few_shot_examples

    def _load_examples(self) -> List[Dict]:
        examples = []
        for gt_example in self._load_ground_truth():
            _id = gt_example['docid']
            _article = self.data_handler.get_article_for_prompting(_id)
            toponym_list = self.data_handler.get_short_toponyms_for_article(_id)

            # Assuming _article is a DataFrame
            example = {
                "input__heading": _article.get('title'),
                "input__news_article": _article.get('text'),
                "input__toponym_list": str(toponym_list),
                "output": str(gt_example['ground_truth']) # TODO: Problem are the {s!!!
            }
            examples.append(example)
        return examples

    def create_example_selector(self, k: int = 2) -> SemanticSimilarityExampleSelector:
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return SemanticSimilarityExampleSelector.from_examples(
            examples=self.examples,
            embeddings=embeddings,
            vectorstore_cls=Chroma,
            k=k
        )

    def create_few_shot_template(self) -> FewShotPromptTemplate:
        example_selector = self.create_example_selector()
        example_template =  """
                            Input:
                            News Article Heading: {input__heading}
                            News Article: {input__news_article}
                            Toponym List: {input__toponym_list}
                            
                            Output: 
                            {output}
                            """

        example_prompt = PromptTemplate(
            input_variables=["input__heading", "input__news_article", "input__toponym_list", "output"],
            template=example_template
        )
        return FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="Here are a few examples of how to provide the search arguments for a given news article:",
            suffix=example_template,
            example_separator="\n\nNext example:\n",
            input_variables=["input__heading", "input__news_article", "input__toponym_list"]
        )

# Example usage
if __name__ == "__main__":
    handler = FewShotExampleHandler(data_directory='data/', xml_file='LGL_test.xml')
    few_shot_template = handler.create_few_shot_template()
    article = handler.data_handler.get_article_for_prompting('38543434')
    tops = str(handler.data_handler.get_short_toponyms_for_article('38543434'))
    sim = few_shot_template.format(
        input__heading=article.get('title'),
        input__news_article=article.get('text'),
        input__toponym_list=tops
    )
    print(few_shot_template)
