from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate

from agent_components.memory.episodic_memory import EpisodicMemory
from agent_components.memory.long_term_memory import LongTermMemory
from agent_components.llms.chatAI import ChatAIHandler


class WorkingMemory:
    def __init__(self):
        self.few_shot_handler = EpisodicMemory(data_directory='data/', xml_file='LGL_test.xml')
        self.long_term_memory = LongTermMemory(documentation_file="agent_components/memory/external_tool_documentation/geonames_websearch_documentation.md")

    def create_final_prompt(self) -> PipelinePromptTemplate:
        final_template = "{system_instructions}\n{task_instructions}\n{documentation}\n{few_shot_examples}"
        final_prompt = PromptTemplate.from_template(final_template)
        pipeline_prompts = [
            ("system_instructions", self.long_term_memory.system_instructions_prompt),
            ("task_instructions", self.long_term_memory.task_instructions_prompt),
            ("documentation", self.long_term_memory.documentation_prompt),
            ("few_shot_examples", self.few_shot_handler.few_shot_template)
        ]
        # ToDo redo prompt to properly format double quotes
        return PipelinePromptTemplate(
            final_prompt=final_prompt,
            pipeline_prompts=pipeline_prompts,
            input_variables=self.few_shot_handler.few_shot_template.input_variables.extend(
                self.long_term_memory.task_instructions_prompt.input_variables
            )
        )

# Example usage
if __name__ == "__main__":
    working_memory = WorkingMemory()
    prompt = working_memory.create_final_prompt()
    article = working_memory.few_shot_handler.data_handler.get_article_for_prompting('44228209')
    tops = str(working_memory.few_shot_handler.data_handler.get_short_toponyms_for_article('44228209'))
    handler = ChatAIHandler()
    model = handler.get_model("meta-llama-3.1-8b-instruct")
    chain = prompt | model
    sim = prompt.format(
        input__heading=article.get('title'),
        input__news_article=article.get('text'),
        input__toponym_list=tops
    )
    llm_answer = chain.invoke(
        {
            "input__heading": article.get('title'),
            "input__news_article": article.get('text'),
            "input__toponym_list": tops
        }
    )
    print(sim)
    print(llm_answer)
