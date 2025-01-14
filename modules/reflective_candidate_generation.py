import json
import pickle
import os
import random
import time

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import END
from langgraph.graph import StateGraph
from openai import InternalServerError
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
                 critic_model_name: str = "meta-llama-3.1-8b-instruct",
                 call_times: list = None):
        load_dotenv()
        self.working_memory = WorkingMemory()
        self.data_handler = self.working_memory.few_shot_handler.data_handler
        self.llm_handler = ChatAIHandler()
        self.llm = self.llm_handler.get_model(actor_model_name)
        self.critic_llm = self.llm_handler.get_model(critic_model_name)
        self.validator = ArticleSyntaxValidator()
        self.geonames = GeoNamesAPI()
        self.call_times = call_times if call_times else []

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

        @handle_api_errors(call_times=self.call_times)
        def _invoke_llm(input_prompt: str):
            return self.llm.invoke(input_prompt)

        try:
            llm_answer = _invoke_llm(prompt)

            if type(llm_answer) == InternalServerError:
                return LLMOutput(
                    article_id=state.article_id,
                    toponyms=state.toponyms,
                    fatal_errors=[Error(
                        execution_step=ExecutionStep.ACTOR,
                        error_message=llm_answer.message
                    )]
                )
            else:
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

            # exclude documentation from the initial prompt to make it shorter
            docu_string = self.working_memory.long_term_memory.create_documentation_message().format()
            initial_prompt = state.initial_prompt.replace(docu_string, "")

            initial_generation_prompt_text = (
                f"Original Prompt Text: \n {initial_prompt + string_seperator}"
                f"Generated Output: \n {generated_output}")
            return initial_generation_prompt_text

        def _generate_critic_prompt_for_fatal_errors() -> str:
            critic_system_prompt = (
                "System: \n"
                "You are a constructive critic for the actor LLM. Please analyze the errors in the generated output "
                "and provide actionable feedback to fix them. Your feedback will be directly used to guide the actor "
                "LLM to generate better outputs in the future. Focus on identifying the specific execution step where "
                "the error occurred and provide feedback ONLY for the cause of this error. Be as concise as possible."
                # Actionable feedback, c.g. Madaan et al. (2023) and Lauscher et al. (2024)
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
                "invalid toponyms and their corresponding errors in the generated output. Provide actionable feedback "
                "to fix these errors. Make sure your feedback adheres closely to the instructions in the original "
                "prompt. Your feedback will be directly used to guide the actor LLM to generate better outputs in the "
                "future. Focus only on the invalid toponyms based on the error message for each of them, using the "
                "valid toponyms as a reference. Be as concise as possible and ensure to include every invalid toponym "
                "provided below."
                # Actionable feedback, c.g. Madaan et al. (2023) and Lauscher et al. (2024)
            )
            initial_generation_prompt = _generate_initial_generation_prompt_text()

            valid_examples, valid_examples_text = [], ""
            if len(state.valid_toponyms) > 2:
                valid_examples = [f"{topo.model_dump_json(indent=4)},\n" for topo in random.sample(state.valid_toponyms, 2)]
            if len(state.duplicate_toponyms) > 2:
                valid_examples.extend([f"{topo.model_dump_json(indent=4)},\n" for topo in random.sample(state.duplicate_toponyms, 2)])
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
            return self.critic_llm.invoke(input_prompt)

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
            if not type(critic_feedback) == InternalServerError:
                state.critic_feedback = AIMessage(**critic_feedback.model_dump())
            else:
                state.fatal_errors.append(Error(
                    execution_step=ExecutionStep.CRITIC,
                    error_message=critic_feedback.message
                ))
                return state

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

    def resolve_candidates(self, state: CandidateGenerationState):
        """
        Resolve the candidates retrieved from the GeoNames API to one correct toponym.
        :param state: The current state of the reflective candidate generation.
        :return: The updated state of the reflective candidate generation.
        """
        # Get the list of toponyms with candidates
        toponyms_with_candidates = state.toponyms_with_candidates
        instructions = (f"System: \n"
                        f"You are an expert with comprehensive geographical knowledge. Your task is to select the "
                        f"right candidate out of the options provided to you for every single toponym. You should "
                        f"use your geographic understanding to reason why a certain candidate is the correct one for "
                        f"the given toponym provided the context of the news article. Be as precise as possible and "
                        f"ensure to select the most relevant candidate for each toponym.\n"
                        f"If you are very sure that none of the candidates fits the toponym, you can also select "
                        f"None as the selected candidate and provide your reason in detail.\n")
        # load example from json file
        with open("data/few_shot_examples_selection.json", "r") as f:
            example = json.load(f)
        example_text = (f"Example: \n"
                        f"Title: {example.title}\n"
                        f"News Article: {example.news_article}\n"
                        f"Toponyms with candidates: \n{example.toponyms_with_candidates}\n"
                        f"Output: \n{example.ground_truth}\n")

    """
    Routing functions
    """

    def has_fatal_errors(self, state: CandidateGenerationState) -> str:
        if state.fatal_errors:
            if state.reflection_phase == ReflectionPhase.ACTOR_RETRY_AFTER_FATAL_ERROR or \
                    state.reflection_phase == ReflectionPhase.ACTOR_RETRY_ON_INVALID_TOPONYMS:
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
    call_times = []
    for actor in ["llama-3.3-70b-instruct", "mistral-large-instruct"]:
        for critic in ["meta-llama-3.1-8b-instruct", "llama-3.3-70b-instruct", "mistral-large-instruct"]:
            seed = 24
            nof_articles = 100

            generator = ReflectiveCandidateGenerator(
                actor_model_name=actor,
                critic_model_name=critic,
                call_times=call_times
            )

            start = time.time()
            generator.reflectively_generate_candidates_for_evaluation(
                seed=seed,
                nof_articles=nof_articles,
                output_dir=f'output/reflective_candidate_generation/fatal_error_and_invalid_correction/{actor}_with_{critic}_critic/{pd.Timestamp.now().strftime("%Y%m%d")}_seed_{seed}_{nof_articles}_articles'

            )
            call_times.extend(generator.call_times)
            print(f"Time taken: {time.time() - start} seconds.")


    actor = "meta-llama-3.1-8b-instruct"  # ["meta-llama-3.1-8b-instruct", "llama-3.3-70b-instruct", "mistral-large-instruct"]
    critic = "meta-llama-3.1-8b-instruct"

    generator = ReflectiveCandidateGenerator(
        actor_model_name=actor,
        critic_model_name=critic,
        call_times=call_times
    )

    # generator.generate_graph_image()

    start = time.time()
    generator.reflectively_generate_candidates_for_evaluation(
        seed=seed,
        nof_articles=nof_articles,
        # output_dir="output/reflective_candidate_generation/mistral-large-instruct_with_mistral-large-instruct_critic/20250112_seed_42_100_articles_2"
        output_dir=f'output/reflective_candidate_generation/fatal_error_and_invalid_correction/{actor}_with_{critic}_critic/{pd.Timestamp.now().strftime("%Y%m%d")}_seed_{seed}_{nof_articles}_articles'
    )
    print(f"Time taken: {time.time() - start} seconds.")