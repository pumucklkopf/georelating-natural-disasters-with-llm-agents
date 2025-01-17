import random

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate

from agent_components.llms.chatAI import ChatAIHandler
from models.candidates import CandidateGenerationState, GeoCodingState, ReflectionPhase
from models.errors import Error

string_seperator = "\n\n----------------------------------------\n"


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

    @staticmethod
    def _create_system_instructions():
        return PromptTemplate.from_template(
            "System:\nYou are an expert API search assistant with comprehensive geographical knowledge. Your task "
            "is to create search parameters for the GeoNames Websearch API based on any provided news article. "
            "Ensure the search parameters are formatted strictly in JSON and comply exactly with the GeoNames "
            "Websearch API documentation. Your goal is to be precise and helpful, which you can best accomplish by "
            "following all instructions accurately and without deviation."
        )

    @staticmethod
    def _create_task_instructions():
        template = '''
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

    def _generate_initial_generation_prompt_text(self, state: CandidateGenerationState | GeoCodingState) -> str:
        generated_output = str(state.raw_output.content)

        # exclude documentation from the initial prompt to make it shorter
        docu_string = self.create_documentation_message().format()
        initial_prompt = state.initial_prompt.replace(docu_string, "")

        initial_generation_prompt_text = (
            f"Original Prompt Text: \n {initial_prompt + string_seperator}"
            f"Generated Output: \n {generated_output}")
        return initial_generation_prompt_text

    def generate_critic_prompt_for_fatal_errors(self,
                                                state: CandidateGenerationState | GeoCodingState,
                                                fatal_errors: list[Error],
                                                initial_prompt: str = "") -> str:
        critic_system_prompt = (
            "System: \n"
            "You are a constructive critic for the actor LLM. Please analyze the errors in the generated output "
            "and provide actionable feedback to fix them. Your feedback will be directly used to guide the actor "
            "LLM to generate better outputs in the future. Focus on identifying the specific execution step where "
            "the error occurred and provide feedback ONLY for the cause of this error. Be as concise as possible."
            # Actionable feedback, c.g. Madaan et al. (2023) and Lauscher et al. (2024)
        )
        initial_generation_prompt = initial_prompt if initial_prompt else self._generate_initial_generation_prompt_text(
            state)
        fatal_errors = str([error.model_dump() for error in fatal_errors])
        critic_instruction = "Your feedback:\n"

        critic_prompt_text = (critic_system_prompt + string_seperator +
                              initial_generation_prompt + string_seperator +
                              f"Fatal Errors: \n {fatal_errors}" + string_seperator +
                              critic_instruction)
        prompt = PromptTemplate(
            template=critic_prompt_text,
            template_format="mustache",
            input_variables=[]
        )
        return prompt.format()

    def generate_critic_prompt_for_invalid_toponyms(self, state: CandidateGenerationState | GeoCodingState) -> str:
        critic_system_prompt = (
            "System: \n"
            "You are a constructive critic for the actor LLM which generated the output below. Please analyze the "
            "invalid toponyms and their corresponding errors in the generated output. Provide actionable feedback "
            "to fix these errors. Make sure your feedback adheres closely to the instructions in the original "
            "prompt. Your feedback will be directly used to guide the actor LLM to generate better outputs in the "
            "future. Focus only on the invalid toponyms based on the error message for each of them, using the "
            "valid toponyms as a reference. Be as concise as possible and ensure to include every invalid toponym "
            "provided below."
            # Actionable feedback, c.g. Madaan et al. (2023) and Lauscher et al. (2024)
        )
        initial_generation_prompt = self._generate_initial_generation_prompt_text(state)

        valid_examples, valid_examples_text = [], ""
        if len(state.valid_toponyms) > 2:
            valid_examples = [f"{topo.model_dump_json(indent=4)},\n" for topo in
                              random.sample(state.valid_toponyms, 2)]
        if len(state.duplicate_toponyms) > 2:
            valid_examples.extend(
                [f"{topo.model_dump_json(indent=4)},\n" for topo in random.sample(state.duplicate_toponyms, 2)])
        if valid_examples:
            valid_examples = "".join(valid_examples)
            valid_examples_text = f"Some valid generations for reference: \n [{valid_examples}]"
        invalid_toponyms = [f"{topo.model_dump_json(indent=4)},\n" for topo in state.invalid_toponyms]
        invalid_toponyms = "".join(invalid_toponyms)
        invalid_toponyms_text = f"All incorrect toponyms with errors: \n [{invalid_toponyms}]"
        # TODO: Check if new format of topos is better
        critic_instruction = "Your feedback:\n"

        critic_prompt_text = (critic_system_prompt + string_seperator +
                              initial_generation_prompt + string_seperator +
                              valid_examples_text + string_seperator +
                              invalid_toponyms_text + string_seperator +
                              critic_instruction)
        prompt = PromptTemplate(
            template=critic_prompt_text,
            template_format="mustache",
            input_variables=[]
        )
        return prompt.format()

    def generate_reflected_actor_prompt(self, state: CandidateGenerationState | GeoCodingState) -> str:
        reflective_actor_system_prompt = (
            "System: \n"
            "You are a reflective actor. Please strictly follow the feedback provided by the critic to generate a "
            "new output which does not lead to the errors your previous generation caused."
        )
        initial_generation_prompt_text = self._generate_initial_generation_prompt_text(state)
        feedback = f"Actionable feedback by the critic: \n{str(state.critic_feedback.content)}"
        reflective_actor_instruction_prompt_text = "Your new output in plain JSON:\n"

        invalid_prompt_part = ""

        if state.reflection_phase == ReflectionPhase.ACTOR_RETRY_ON_INVALID_TOPONYMS:
            invalids = str([topo.toponym for topo in state.invalid_toponyms])
            invalid_prompt_part = (f"Invalid toponyms: \n"
                                   f"{invalids + string_seperator}"
                                   "Now generate the search arguments ONLY for all invalid toponyms, NOT for the "
                                   "valid or duplicate ones. \n")

        reflected_prompt_text = (reflective_actor_system_prompt + string_seperator +
                                 initial_generation_prompt_text + string_seperator +
                                 feedback + string_seperator +
                                 invalid_prompt_part +
                                 reflective_actor_instruction_prompt_text)
        prompt = PromptTemplate(
            template=reflected_prompt_text,
            template_format="mustache",
            input_variables=[]
        )
        return prompt.format()

    @staticmethod
    def _generate_initial_generation_prompt_for_resolution(state: GeoCodingState) -> str:
        # Generate the initial generation prompt text
        generated_output = str(state.resolution_raw_output.content)
        initial_generation_prompt = state.resolution_initial_prompt

        initial_generation_prompt = (
            f"Original Prompt Text: \n {initial_generation_prompt + string_seperator}"
            f"Generated Output: \n {generated_output}"
        )
        return initial_generation_prompt

    def generate_resolution_critic_prompt_for_fatal_errors(self, state: GeoCodingState) -> str:
        """
        Generate the prompt for the resolution critic in case of fatal errors.
        :param state: The current state of the reflective candidate resolution.
        :return: The formatted prompt for the resolution critic.
        """
        initial_generation_prompt = self._generate_initial_generation_prompt_for_resolution(state)
        return self.generate_critic_prompt_for_fatal_errors(state=state,
                                                            fatal_errors=state.resolution_fatal_errors,
                                                            initial_prompt=initial_generation_prompt)

    def generate_resolution_critic_prompt_for_invalid_toponyms(self, state: GeoCodingState) -> str:
        """
        Generate the prompt for the resolution critic in case of invalid, but not fatal outputs.
        :param state: The current state of the reflective candidate resolution.
        :return: The formatted prompt for the resolution critic.
        """
        critic_system_prompt = (
            "System: \n"
            "You are a constructive critic for the actor LLM which generated the output below. Please analyze the "
            "invalid output and their corresponding errors in the generated output. Provide actionable feedback "
            "to fix these errors. Make sure your feedback adheres closely to the instructions in the original "
            "prompt. Your feedback will be directly used to guide the actor LLM to generate better outputs in the "
            "future. Focus only on the invalid outputs based on the error message for each of them."
            " Be as concise as possible and ensure to include every invalid output provided below."
            # Actionable feedback, c.g. Madaan et al. (2023) and Lauscher et al. (2024)
        )
        initial_generation_prompt = self._generate_initial_generation_prompt_for_resolution(state)

        invalid_outputs = [f"{topo.model_dump_json(indent=4)},\n" for topo in state.invalid_geocoded_toponyms]
        invalid_outputs_text = f"All incorrect toponyms with errors: \n [{invalid_outputs}]"

        critic_instruction = "Your feedback:\n"

        critic_prompt_text = (critic_system_prompt + string_seperator +
                              initial_generation_prompt + string_seperator +
                              invalid_outputs_text + string_seperator +
                              critic_instruction)

        prompt = PromptTemplate(
            template=critic_prompt_text,
            template_format="mustache",
            input_variables=[]
        )
        return prompt.format()


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
