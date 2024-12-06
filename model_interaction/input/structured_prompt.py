from typing import Optional, List, Dict
import json

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from few_shot_examples import FewShotExampleHandler
from instructions import SearchAgentSetup
from model_interaction.chatAI import ChatAIHandler


class StructuredPrompt:
    def __init__(self):
        self.few_shot_handler = FewShotExampleHandler(data_directory='data/', xml_file='LGL_test.xml')
        self.search_agent_setup = SearchAgentSetup(documentation_file="model_interaction/input/geonames_websearch_documentation.md")

    def create_final_prompt(self) -> PipelinePromptTemplate:
        final_template = "{system_instructions}\n{task_instructions}\n{documentation}\n{few_shot_examples}"
        final_prompt = PromptTemplate.from_template(final_template)
        pipeline_prompts = [
            ("system_instructions", self.search_agent_setup.system_instructions_prompt),
            ("task_instructions", self.search_agent_setup.task_instructions_prompt),
            ("documentation", self.search_agent_setup.documentation_prompt),
            ("few_shot_examples", self.few_shot_handler.few_shot_template)
        ]
        return PipelinePromptTemplate(
            final_prompt=final_prompt,
            pipeline_prompts=pipeline_prompts,
            input_variables=self.few_shot_handler.few_shot_template.input_variables.extend(
                self.search_agent_setup.task_instructions_prompt.input_variables
            )
        )

class OutputParser(PydanticOutputParser):
    class ToponymSearchArguments(BaseModel):
        """Search arguments for one toponym."""
        toponym: str = Field(description="The toponym to search for")
        params: Optional[Dict] = Field(default=None,
                                       description="The search arguments to use for the GeoNames API call")
        duplicate_of: Optional[str] = Field(default=None,
                                            description="The toponym to reference if the toponym is duplicated")

    def __init__(self):
        super().__init__(pydantic_object=self.ToponymSearchArguments)

    def extract_output(self, message: AIMessage) -> [ToponymSearchArguments]:
        search_list = []
        # if the string does not start with a '[', strip of the characters before the first '['
        if message.content[0] != '[':
            message.content = message.content[message.content.find('['):]
            if message.content[-1] != ']':
                message.content = message.content[:message.content.rfind(']') + 1]
        for location_mention in json.loads(message.content):
            search_list.append(self.ToponymSearchArguments.model_validate(location_mention))
        return search_list


# Example usage
if __name__ == "__main__":
    prompt_generator = StructuredPrompt()
    prompt = prompt_generator.create_final_prompt()
    parser = OutputParser()
    article = prompt_generator.few_shot_handler.data_handler.get_article_for_prompting('44148898')
    tops = str(prompt_generator.few_shot_handler.data_handler.get_short_toponyms_for_article('44148898'))
    handler = ChatAIHandler()
    model = handler.get_model("meta-llama-3.1-8b-instruct")
    chain = prompt | model | parser.extract_output
    sim = prompt.format(
        schema = str(parser.ToponymSearchArguments.model_json_schema()),
        input__heading=article.get('title'),
        input__news_article=article.get('text'),
        input__toponym_list=tops
    )
    llm_answer = chain.invoke(
        {
            "schema": str(parser.ToponymSearchArguments.model_json_schema()),
            "input__heading": article.get('title'),
            "input__news_article": article.get('text'),
            "input__toponym_list": tops
        }
    )
    print(sim)
    #print(parser.get_format_instructions())
    print(llm_answer)
