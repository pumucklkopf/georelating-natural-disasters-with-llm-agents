from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate

from agent_components.llms.chatAI import ChatAIHandler


class LongTermMemory:
    def __init__(self, documentation_file='external_tool_documentation/geonames_websearch_documentation.md'):
        self.documentation_file = documentation_file
        self.documentation = self._load_documentation()
        self.system_instructions_prompt = self._create_system_instructions()
        self.task_instructions_prompt = self._create_task_instructions()
        self.documentation_prompt = self.create_documentation_message()

    def _load_documentation(self):
        loader = TextLoader(self.documentation_file, encoding='utf-8')
        documentation = loader.load()
        return documentation

    def _create_system_instructions(self):
        return PromptTemplate.from_template(
              "System:\nYou are an expert API search assistant with comprehensive geographical knowledge. Your task "
              "is to create search parameters for the GeoNames Websearch API based on any provided news article. "
              "Ensure the search parameters are formatted strictly in JSON and comply exactly with the GeoNames "
              "Websearch API documentation. Your goal is to be precise and helpful, which you can best accomplish by "
              "following all instructions accurately and without deviation."
        )

    def _create_task_instructions(self):
        template =  '''
            Human:
            Please create the search arguments for the GeoNames Websearch API based on the given news article.
            
            Your Task:
            
            1. Read the news article under the key 'News Article' to understand its content.
            2. Identify all the toponyms listed under the key 'Toponym List' within the article.
            3. For each toponym in the 'Toponym List,' generate the search arguments for the GeoNames Websearch API in JSON format.
            4. Strictly follow the JSON output format: [{"toponym": "<toponym>", "params": {"<search argument>": "<search_value>"}}].
            5. Ensure that the search arguments comply with the GeoNames Websearch API documentation.
            6. If any toponyms are duplicated based on the context of the news article, use the 'duplicate_of' key in the output JSON to reference the first occurrence of the toponym instead of the 'params' key.
            7. Typically, use the search argument 'q' with the toponym as the value, along with other relevant information such as upper administrative orders.
            8. Set the search argument 'isNameRequired' to 'true' to ensure relevant search results.
            9. Use the 'maxRows' search argument to limit the number of results returned.
            10. Dynamically select additional search arguments based on the context of the news article.
            11. Ensure the search arguments are as specific as possible to return only a few, highly relevant results.'''

        return PromptTemplate(
            template=template,
            template_format="mustache",
            input_variables=[]
        )

    def create_documentation_message(self):
        return PromptTemplate.from_template(
            f"Here is the documentation for the GeoNames Websearch API provided in Markdown:\n"
            f"{self.documentation[0].page_content}"
        )

# Example usage
if __name__ == "__main__":
    handler = ChatAIHandler()
    model = handler.get_model("codestral-22b")
    long_term_memory = LongTermMemory()
    chain = (long_term_memory.system_instructions_prompt |
             long_term_memory.task_instructions_prompt |
             long_term_memory.documentation_prompt |
             model)
    llm_answer = chain.invoke({"news_article": "The news article content", "toponym_list": ["toponym1", "toponym2"]})
    print(llm_answer.content)
