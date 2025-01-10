import pickle
import os
import time

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import END
from langgraph.graph import StateGraph
from tqdm import tqdm

from agent_components.environment.external_tools import GeoNamesAPI
from agent_components.environment.internal_tools import ArticleSyntaxValidator, OutputParser
from agent_components.llms.api_error_handler import handle_api_errors
from agent_components.llms.chatAI import ChatAIHandler
from agent_components.memory.working_memory import WorkingMemory
from models.candidates import CandidateGenerationState, CandidateGenerationOutput, ReflectionPhase
from models.errors import Error, ExecutionStep
from models.llm_output import LLMOutput, ValidatedOutput

string_seperator = "\n\n----------------------------------------\n"


class ReflectiveCandidateGenerator:
    def __init__(self,
                 actor_model_name: str = "meta-llama-3.1-8b-instruct",
                 critic_model_name: str = "meta-llama-3.1-8b-instruct"):
        load_dotenv()
        self.working_memory = WorkingMemory()
        self.data_handler = self.working_memory.few_shot_handler.data_handler
        self.llm_handler = ChatAIHandler()
        self.llm = self.llm_handler.get_model(actor_model_name)
        self.critic_llm = self.llm_handler.get_model(critic_model_name)
        self.validator = ArticleSyntaxValidator()
        self.geonames = GeoNamesAPI()
        self.call_times = []

    """
    Node functions
    """

    def create_prompt(self, input_state: CandidateGenerationState) -> CandidateGenerationState:
        toponyms = self.data_handler.get_short_toponyms_for_article(input_state.article_id)
        prompt = self.working_memory.create_final_prompt()
        formatted_prompt = prompt.format(
                input__heading=input_state.article_title,
                input__news_article=input_state.article_text,
                input__toponym_list=str(toponyms)
        )
        # Replace all "&quot;" with double quotes
        formatted_prompt = formatted_prompt.replace("&quot;", "\"") #todo: remove when fixed in langchain
        return CandidateGenerationState(
            article_id=input_state.article_id,
            article_title=input_state.article_title,
            article_text=input_state.article_text,
            toponyms=toponyms,
            reflection_phase=ReflectionPhase.INITIAL_ACTOR_GENERATION,
            initial_prompt=formatted_prompt
        )

    def call_actor(self, state: CandidateGenerationState) -> LLMOutput:
        # First check in which state of the reflective candidate generation we are
        if state.reflection_phase == ReflectionPhase.INITIAL_ACTOR_GENERATION:
            prompt = state.initial_prompt
        else:
            prompt = state.reflected_prompt
        # Build the LLM call sub-chain

        @handle_api_errors(call_times=self.call_times)
        def _invoke_llm(input_prompt: str):
            return self.llm.invoke(input_prompt)

        try:
            llm_answer = _invoke_llm(prompt)

            return LLMOutput(
                article_id=state.article_id,
                toponyms=state.toponyms,
                raw_output=AIMessage(**llm_answer.model_dump())
            )
        except Exception as e:
            return LLMOutput(
                article_id=state.article_id,
                toponyms=state.toponyms,
                fatal_errors=[Error(
                    execution_step=ExecutionStep.ACTOR,
                    error_message=str(e)
                )]
            )

    def extract_output(self, state: LLMOutput) -> LLMOutput:
        parser = OutputParser(article_id=state.article_id,
                              toponym_list=state.toponyms)
        parsed_output = parser.extract_output(state.raw_output)
        # Ensure no duplicate keys by unpacking parsed_output selectively
        parsed_data = parsed_output.model_dump()
        parsed_data.update({
            "article_title": state.article_title,
            "article_text": state.article_text,
        })
        return LLMOutput(**parsed_data)

    def validate_output(self, state: LLMOutput) -> ValidatedOutput:
        return self.validator.validate_toponyms_of_article(state)

    def retrieve_candidates(self, state: ValidatedOutput) -> CandidateGenerationOutput:
        return self.geonames.retrieve_candidates(state)

    def criticize(self, state: CandidateGenerationState) -> CandidateGenerationState:
        def _generate_initial_generation_prompt_text() -> str:
            generated_output = str(state.raw_output.content)
            initial_generation_prompt_text = (
                f"Original Prompt Text: \n {state.initial_prompt + string_seperator}"
                f"Generated Output: \n {generated_output}")
            return initial_generation_prompt_text

        def _generate_critic_prompt_for_fatal_errors() -> str:
            critic_system_prompt = (
                "System: \n"
                "You are a constructive critic for the actor LLM. Please analyze the errors in the generated output "
                "and provide actionable feedback to fix them. Your feedback will be directly used to guide the actor "
                "LLM to generate better outputs in the future. Only provide feedback to the errors under the Fatal "
                "Errors key, nothing else, and be as concise as possible."
                # Actionable feedback, c.g. Madaan et al. (2023) and Lauscher et al. (2024)
                # TODO: creation-step specific feedback instructions for fatal errors
            )
            initial_generation_prompt = _generate_initial_generation_prompt_text()
            fatal_errors = str([error.model_dump() for error in state.fatal_errors])
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

        def _generate_critic_prompt_for_invalid_toponyms() -> str:
            critic_system_prompt = (
                "System: \n"
                "You are a constructive critic for the actor LLM which generated the output below. Please analyze the "
                "invalid toponyms and corresponding errors in the generated output and provide actionable feedback to "
                "fix them. Your feedback will be directly used to guide the actor LLM to generate better outputs in "
                "the future. Only provide feedback to the invalid toponyms, not the others, and use the valid toponyms "
                "as a reference. Be as concise as possible."
                # Actionable feedback, c.g. Madaan et al. (2023) and Lauscher et al. (2024)
                # TODO enhance darstellung von valid and invalid topos (lists to dicts or something)
            )
            initial_generation_prompt = _generate_initial_generation_prompt_text()
            critic_instruction = "Your feedback:\n"

            critic_prompt_text = (critic_system_prompt + string_seperator +
                                  initial_generation_prompt + string_seperator +
                                  f"Valid Toponyms with Candidates: \n {str(state.toponyms_with_candidates)}" + string_seperator +  #todo not all valids to include?
                                  f"Incorrect Toponyms with errors: \n {str(state.invalid_toponyms)}" + string_seperator +
                                  critic_instruction)
            prompt = PromptTemplate(
                template=critic_prompt_text,
                template_format="mustache",
                input_variables=[]
            )
            return prompt.format()

        def _generate_reflected_actor_prompt() -> str:
            reflective_actor_system_prompt = (
                "System: \n"
                "You are a reflective actor. Please strictly follow the feedback provided by the critic to generate a "
                "new output which does not lead to the errors your previous generation caused."
            )
            initial_generation_prompt_text = _generate_initial_generation_prompt_text()
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
        # TODO feedback for duplicates sucks

        @handle_api_errors(call_times=self.call_times)
        def _invoke_llm(input_prompt: str):
            return self.llm.invoke(input_prompt)

        # todo: what to do if fatal error occurs in actor_retry_on_invalid_toponyms?
        # compile final critic prompt
        if state.fatal_errors:
            state.reflection_phase = ReflectionPhase.CRITIC_GENERATION_FOR_FATAL_ERRORS
            state.critic_prompt = _generate_critic_prompt_for_fatal_errors()
        else:
            state.reflection_phase = ReflectionPhase.CRITIC_GENERATION_FOR_INVALID_TOPONYMS
            state.critic_prompt = _generate_critic_prompt_for_invalid_toponyms()

        try:
            # invoke critic and use feedback to generate reflected actor prompt
            critic_feedback = _invoke_llm(state.critic_prompt)
            state.critic_feedback = AIMessage(**critic_feedback.model_dump())

            # generate the matching reflected actor prompt
            if state.reflection_phase == ReflectionPhase.CRITIC_GENERATION_FOR_FATAL_ERRORS:
                state.reflection_phase = ReflectionPhase.ACTOR_RETRY_AFTER_FATAL_ERROR
            else:
                state.reflection_phase = ReflectionPhase.ACTOR_RETRY_ON_INVALID_TOPONYMS
            state.reflected_prompt = _generate_reflected_actor_prompt()

            # Reset fatal errors for the next iteration of the reflective candidate generation
            state.fatal_errors = []
            return state

        except Exception as e:
            state.fatal_errors.append(Error(
                execution_step=ExecutionStep.CRITIC,
                error_message=str(e)
            ))
            return state

    def add_critique_error(self, state: CandidateGenerationState) -> CandidateGenerationState:
        if state.fatal_errors:
            state.fatal_errors.append(Error(
                execution_step=ExecutionStep.CRITIC,
                error_message="Critique did not solve fatal errors. Exiting."
            ))
        return state

    """
    Routing functions
    """

    def has_fatal_errors(self, state: CandidateGenerationState) -> str:
        if state.fatal_errors:
            if state.reflection_phase == ReflectionPhase.ACTOR_RETRY_AFTER_FATAL_ERROR:
                return "critique_did_not_solve_fatal_errors"
            else:
                return "has_fatal_errors"
        else:
            return "successful"

    def has_invalid_toponyms(self, state: CandidateGenerationState)-> str:
        if state.fatal_errors:
            if state.reflection_phase == ReflectionPhase.ACTOR_RETRY_AFTER_FATAL_ERROR:
                return "critique_did_not_solve_fatal_errors_or_invalid_toponyms"
            else:
                return "has_fatal_errors_or_invalid_toponyms"
        else:
            if state.invalid_toponyms:
                if state.reflection_phase == ReflectionPhase.ACTOR_RETRY_ON_INVALID_TOPONYMS:
                    return "critique_did_not_solve_fatal_errors_or_invalid_toponyms"
                else:
                    return "has_fatal_errors_or_invalid_toponyms"
            else:
                return "everything_valid"

    def critique_was_generated(self, state: CandidateGenerationState) -> str:
        if state.fatal_errors:
            return "critique_failed"
        else:
            return "critique_generated"

    """
    Grap Structure
    """

    def build_graph(self):
        graph_builder = StateGraph(CandidateGenerationState)

        # Add nodes to the graph
        graph_builder.add_node("create_prompt", self.create_prompt)
        graph_builder.add_node("call_actor", self.call_actor)
        graph_builder.add_node("extract_output", self.extract_output)
        graph_builder.add_node("validate_output", self.validate_output)
        graph_builder.add_node("retrieve_candidates", self.retrieve_candidates)
        graph_builder.add_node("criticize", self.criticize)
        graph_builder.add_node("add_critique_error", self.add_critique_error)

        # Set the entry point and define edges
        graph_builder.set_entry_point("create_prompt")
        graph_builder.add_edge("create_prompt", "call_actor")
        graph_builder.add_conditional_edges("call_actor",
                                            self.has_fatal_errors,
                                            {
                                                "has_fatal_errors": "criticize",
                                                "successful": "extract_output",
                                                "critique_did_not_solve_fatal_errors": "add_critique_error"
                                            })
        graph_builder.add_conditional_edges("extract_output",
                                            self.has_fatal_errors,
                                            {
                                                "has_fatal_errors": "criticize",
                                                "successful": "validate_output",
                                                "critique_did_not_solve_fatal_errors": "add_critique_error"
                                            })
        graph_builder.add_conditional_edges("validate_output",
                                            self.has_fatal_errors,
                                            {
                                                "has_fatal_errors": "criticize",
                                                "successful": "retrieve_candidates",
                                                "critique_did_not_solve_fatal_errors": "add_critique_error"
                                            })
        graph_builder.add_conditional_edges("retrieve_candidates",
                                            self.has_invalid_toponyms,
                                            {
                                                "has_fatal_errors_or_invalid_toponyms": "criticize",
                                                "everything_valid": END,
                                                "critique_did_not_solve_fatal_errors_or_invalid_toponyms": "add_critique_error"
                                            })
        graph_builder.add_conditional_edges("criticize",
                                            self.critique_was_generated,
                                            {
                                                "critique_generated": "call_actor",
                                                "critique_failed": "add_critique_error"
                                            })
        graph_builder.add_edge("add_critique_error", END)
        return graph_builder

    """
    Compile and run the graph
    """

    def run_graph(self, article: pd.Series):
        graph_builder = self.build_graph()
        agent_graph = graph_builder.compile()

        # Run the graph
        input_state = {
            "article_id": article['docid'],
            "article_title": article['title'],
            "article_text": article['text']
        }
        agent_graph_answer = agent_graph.invoke(input_state)

        return agent_graph_answer

    def reflectively_generate_candidates_for_evaluation(self,
                                                        seed: int = 42,
                                                        nof_articles: int = 100,
                                                        output_dir: str = 'output/'):
        os.makedirs(output_dir, exist_ok=True)

        articles_df = self.data_handler.get_random_articles_for_evaluation(
            seed=seed, n=nof_articles
        )
        for index, article in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Processing articles"):
            # First check if the article has already been processed
            article_id = article['docid']

            if os.path.exists(os.path.join(output_dir, f'{article_id}.pkl')):
                continue

            agent_graph_answer = self.run_graph(article)

            # Parse dictionary (default langgraph output) to CandidateGenerationState
            agent_graph_answer = CandidateGenerationState(**agent_graph_answer)

            with open(os.path.join(output_dir, f'{article_id}.pkl'), 'wb') as f:
                pickle.dump(agent_graph_answer, f)

    def generate_graph_image(self, output_file_path: str = "new_new_graph_layout_image.png"):
        graph_builder = self.build_graph()
        agent_graph = graph_builder.compile()
        image_data = agent_graph.get_graph().draw_mermaid_png(output_file_path=output_file_path)
        # Open the image
        os.system(output_file_path)
        return image_data


if __name__ == "__main__":
    generator = ReflectiveCandidateGenerator(
        actor_model_name="meta-llama-3.1-8b-instruct",
        critic_model_name="meta-llama-3.1-8b-instruct"
    )
    # generator.generate_graph_image()
    start = time.time()
    generator.reflectively_generate_candidates_for_evaluation(
        seed=42,
        nof_articles=100,
        output_dir=f'output/reflective_candidate_generation/llama_8b_with_llama_8b_critic/{pd.Timestamp.now().strftime("%Y%m%d")}_seed_42_100_articles'
    )
    print(f"Time taken: {time.time() - start} seconds.")