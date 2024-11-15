from langchain_core.prompts import FewShotPromptWithTemplates, PromptTemplate, StringPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from data_handler.xml_parsing import XMLDataHandler


def format_output(toponyms, search_arguments, duplicate_of):
    output_lines = []
    for toponym, search_arg, duplicate in zip(toponyms, search_arguments, duplicate_of):
        output_lines.append(f"Toponym: {toponym}\nSearch Arguments: {search_arg}\nDuplicate Of: {duplicate}\n")
    return "\n".join(output_lines)

data_handler = XMLDataHandler('data/')
data_handler.parse_xml('LGL_test.xml')
# get 10 articles for prompting
for example_article in data_handler.get_articles_for_prompting().head(10).iterrows():
    toponym_list = data_handler.get_short_toponyms_for_article(example_article[1]['docid']).values.tolist()
    example = {
        "input.heading": example_article[1]['title'],
        "input.news_article": example_article[1]['text'],
        "input.toponym_list":
        "output": format_output(toponym_list, search_arguments_list, duplicate_of_list)
    }
article_docid = '39423136'
toponyms_for_article = data_handler.get_toponyms_for_article(article_docid)



examples = []

vector_store = Chroma(
    collection_name="foo",
    embedding_function=OpenAIEmbeddings(),
)




document_1 = Document(page_content="foo", metadata={"baz": "bar"})
document_2 = Document(page_content="thud", metadata={"bar": "baz"})
document_3 = Document(page_content="i will be deleted :(")

documents = [document_1, document_2, document_3]
ids = ["1", "2", "3"]
vector_store.add_documents(documents=documents, ids=ids)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vector_store,
    k=2,  # Number of examples to select
)
example_prompt = PromptTemplate(
    input_variables=["input.heading", "input.news_article", "input.toponym_list", "output"],
    template=   """
                Input:\n
                News Article Heading: {input.heading}\n
                News Article: {input.news_article}\n
                Toponym List: {input.toponym_list}\n
                Output:\n
                {output}
                """

)

few_shot_template = FewShotPromptWithTemplates( # todo nothing works yet
    examples=examples,
    example_selector=example_selector, #todo: dynamically select examples based on the context with BM25 or other methods
    example_prompt=example_prompt,
    prefix="Here are a few examples of how to provide the search arguments for a given news article:",
    suffix={"Make sure the search arguments are as specific as possible, so that the search returns only a few, highly relevant results."},
    example_separator=lambda index: f"\n example {index + 1} \n",
    validate_template=False
)
print("test")
