from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from model_interaction.chatAI import ChatAIHandler


class SearchAgentSetup:
    def __init__(self, documentation_file='geonames_websearch_documentation.md'):
        self.documentation_file = documentation_file
        self.documentation = self._load_documentation()
        self.system_instructions = self._create_system_instructions()
        self.task_instructions = self._create_task_instructions()
        self.instructions = self._create_instructions()

    def _load_documentation(self):
        loader = TextLoader(self.documentation_file, encoding='utf-8')
        documentation = loader.load()
        return documentation

    def _create_system_instructions(self):
        return SystemMessage(
            content="""
                You are a helpful API search assistant which is very confident about its geographical knowledge.
                You can therefore help in providing the search arguments for the GeoNames Websearch API for any given news
                article as input. The search arguments should be strictly in JSON format. Make extremely sure that the
                search arguments adhere to the GeoNames Websearch API documentation, which will be provided to you.
                In general, your role is to be very precise and helpful, which you can best achieve by adhering to all
                the instructions provided to you and not experimenting or deviating from them.
                """
        )

    def _create_task_instructions(self):
        return HumanMessage(
            content=f"""
                Please provide the search arguments for the GeoNames Websearch API for the given news article. Your task is
                specifically:
                1. Read the news article provided to you with the key "News Article" to understand the context of the news.
                2. In the article, identify all the toponyms which are provided with the key "Toponym List".
                3. For every toponym in the "Toponym List", provide the search arguments for the GeoNames Websearch API in JSON format.
                4. For this, adhere strictly to the output format: {[{"toponym": "<toponym>", "params": "<search arguments>", "duplicate_of": "<toponym>"}]}.
                5. Make sure that the search arguments adhere strictly to the GeoNames Websearch API documentation.
                6. If you think that some toponyms are duplicated given the context of the news article (not necessarily the
                same name), you can just reference the first occurrence of the toponym by providing the toponym to the
                "duplicate_of" key in the output JSON.
                7. Usually, the search argument "q" is appropriate to use, and it can be provided with the toponym as the value, as well as other information such as upper administrative orders.
                8. Usually, the search argument "isNameRequired" is set to "true" to ensure that the search results are relevant.
                9. Make sure to make use of the "maxRows" search argument to limit the number of results returned.
                10. Other than that, you can dynamically select the search arguments based on the context of the news article.
                11. Make sure the search arguments are as specific as possible, so that the search returns only a few, highly relevant results.

                Here is the documentation for the GeoNames Websearch API provided in markdown:
                {self.documentation}
                """
        )

    def _create_instructions(self):
        return ChatPromptTemplate.from_messages([
            self.system_instructions,
            self.task_instructions
        ])

# Example usage
if __name__ == "__main__":
    handler = ChatAIHandler()
    model = handler.get_model("mistral-large-instruct")
    search_agent_setup = SearchAgentSetup()
    chain = search_agent_setup.instructions | model
    llm_answer = chain.invoke({"news_article": "The news article content", "toponym_list": ["toponym1", "toponym2"]})
    print(llm_answer.content)

