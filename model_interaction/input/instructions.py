from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate

from model_interaction.chatAI import ChatAIHandler


class SearchAgentSetup:
    def __init__(self, documentation_file='geonames_websearch_documentation.md'):
        self.documentation_file = documentation_file
        self.documentation = self._load_documentation()
        self.system_instructions_prompt = self._create_system_instructions()
        self.task_instructions_prompt = self._create_task_instructions()
        self.documentation_prompt = self._create_documentation_message()

    def _load_documentation(self):
        loader = TextLoader(self.documentation_file, encoding='utf-8')
        documentation = loader.load()
        return documentation

    def _create_system_instructions(self):
        return PromptTemplate.from_template(
             "System:\nYou are a confident and helpful API search assistant with extensive geographical knowledge. "
             "Your task is to generate search arguments for the GeoNames Websearch API based on any given news "
             "article. Ensure the search arguments are strictly in JSON format and adhere precisely to the GeoNames "
             "Websearch API documentation provided. Your role is to be precise and helpful, which you can best "
             "achieve by following all instructions accurately and without deviation."
        )

    def _create_task_instructions(self):
        template = '''
        Human:

        Please generate the search arguments for the GeoNames Websearch API based on the provided news article.

        Your Task:

        1. Read the news article provided under the key "News Article" to understand its context.
        2. Identify all the toponyms listed under the key "Toponym List" within the article.
        3. For each toponym in the "Toponym List," provide the search arguments for the GeoNames Websearch API in JSON format.
        4. Adhere strictly to the JSON output format: [{"toponym": "<toponym>", "params": {"<search argument>": "<search_value>"}}].
        5. Ensure that the search arguments comply with the GeoNames Websearch API documentation.
        6. If any toponyms are duplicated based on the context of the news article, reference the first occurrence of the toponym by using the "duplicate_of" key in the output JSON instead of the "params" key.
        7. Typically, use the search argument "q" with the toponym as the value, along with other relevant information such as upper administrative orders.
        8. Set the search argument "isNameRequired" to "true" to ensure relevant search results.
        9. Use the "maxRows" search argument to limit the number of results returned.
        10. Dynamically select additional search arguments based on the context of the news article.
        11. Ensure the search arguments are as specific as possible to return only a few, highly relevant results.
        '''

        return PromptTemplate(
            template=template,
            template_format="mustache",
            input_variables=["schema"]
        )

    def _create_documentation_message(self):
        return PromptTemplate.from_template(
            f"Here is the documentation for the GeoNames Websearch API provided in Markdown:\n"
            f"{self.documentation[0].page_content}"
        )

# Example usage
if __name__ == "__main__":
    handler = ChatAIHandler()
    model = handler.get_model("codestral-22b")
    search_agent_setup = SearchAgentSetup()
    chain = (search_agent_setup.system_instructions_prompt |
             search_agent_setup.task_instructions_prompt |
             search_agent_setup.documentation_prompt |
             model)
    llm_answer = chain.invoke({"news_article": "The news article content", "toponym_list": ["toponym1", "toponym2"]})
    print(llm_answer.content)
