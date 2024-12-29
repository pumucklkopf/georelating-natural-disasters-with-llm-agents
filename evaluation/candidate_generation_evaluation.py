import os
import pickle

from pydantic import BaseModel, Field

from data_handler.xml_parsing import XMLDataHandler
from models.candidates import CandidateGenerationOutput


class CandidateGenerationMetrics(BaseModel):
    recall_at_10: float = Field(description="Recall@10 metric for the candidate generation task",
                                default=0)

    percentage_toponyms_with_fatal_errors: float = Field(
        description="Number of toponyms for which a fatal error occurred during the candidate generation",
        default=0)
    percentage_toponyms_without_valid_search_arguments: float = Field(
        description="Number of toponyms for which no parsable search arguments were generated",
        default=0)
    percentage_toponyms_without_candidates: float = Field(
        description="Number of toponyms for which the API call did not return any candidates",
        default=0)
    percentage_toponyms_without_correct_candidates: float = Field(
        description="Number of toponyms for which the correct candidate was not found in the top 10 candidates",
        default=0)
    percentage_too_many_generated_toponym_candidates: float = Field(
        description="Number of toponyms for which too many toponym candidates were generated",
        default=0)

    avg_nof_candidates: float = Field(description="Average number of candidates per toponym if candidates were found",
                                      default=0)
    nof_articles_with_fatal_errors: int = Field(description="Number of articles for which a fatal error occurred during the candidate generation",
                                                default=0)
    nof_all_gt_toponyms: int = Field(description="Total number of ground truth toponyms",
                                     default=0)
    nof_all_generated_toponyms: int = Field(description="Total number of generated toponyms",
                                            default=0)

class CandidateGenerationEvaluator:
    def __init__(self, data_directory: str, output_directory: str):
        self.data_directory = data_directory
        self.output_directory = output_directory
        self.data_handler = XMLDataHandler(data_directory)

    def load_generation_for_article(self, docid: str) -> CandidateGenerationOutput:
        file_path = os.path.join(self.output_directory, f"{docid}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Generated candidates for docid {docid} not found in {self.output_directory}.")
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def calculate_metrics(self) -> CandidateGenerationMetrics:
        total_toponyms = 0
        matched_toponyms = 0

        nof_toponyms_with_candidates = 0
        total_nof_candidates = 0

        nof_toponyms_without_valid_search_arguments = 0
        nof_toponyms_without_candidates = 0

        too_many_generated_toponyms = 0
        nof_articles_with_too_many_generated_toponyms = 0

        too_few_generated_toponyms = 0
        nof_articles_with_too_few_generated_toponyms = 0

        nof_all_generated_toponyms = 0

        articles_with_fatal_errors = 0
        toponyms_with_fatal_errors = 0

        self.data_handler.parse_xml("LGL_test.xml")

        # Iterate through generated candidate files
        for candidate_file in os.listdir(self.output_directory):
            if not candidate_file.endswith(".pkl"):
                continue

            docid = candidate_file.replace(".pkl", "")

            try:
                generation_for_article = self.load_generation_for_article(docid)
            except FileNotFoundError:
                continue

            # Retrieve toponyms for the current article
            article_toponyms = self.data_handler.get_toponyms_for_article(docid)

            nof_article_generated_toponyms = len(
                generation_for_article.toponyms_with_candidates + generation_for_article.invalid_toponyms)
            nof_all_generated_toponyms += nof_article_generated_toponyms

            if generation_for_article.fatal_errors:
                articles_with_fatal_errors += 1
                if nof_article_generated_toponyms == 0:
                    toponyms_with_fatal_errors += len(article_toponyms)
            else:

                nof_toponyms_without_valid_search_arguments += len(generation_for_article.invalid_toponyms)


                # only for code error checking
                nof_article_toponyms = len(article_toponyms)
                if nof_article_generated_toponyms > nof_article_toponyms:
                    too_many_generated_toponyms += nof_article_generated_toponyms - nof_article_toponyms
                    nof_articles_with_too_many_generated_toponyms += 1
                elif nof_article_generated_toponyms < nof_article_toponyms:
                    too_few_generated_toponyms += nof_article_toponyms - nof_article_generated_toponyms
                    nof_articles_with_too_few_generated_toponyms += 1

            for toponym_row in article_toponyms:
                total_toponyms += 1
                correct_geonameid = toponym_row["geonameid"]

                # Check if the correct geonameid is in the top 10 candidates
                generated_toponyms_with_candidates = generation_for_article.toponyms_with_candidates.copy()
                for toponym in generated_toponyms_with_candidates:
                    if toponym.toponym_with_search_arguments.toponym == toponym_row["phrase"]:
                        generation_for_article.toponyms_with_candidates.remove(toponym)
                        if toponym.total_results == 0:
                            nof_toponyms_without_candidates += 1
                            break
                        else:
                            nof_toponyms_with_candidates += 1
                            total_nof_candidates += toponym.total_results
                            if any(str(candidate.get("geonameId")) == correct_geonameid for candidate in toponym.candidates[:10]):
                                matched_toponyms += 1
                                break

        if total_toponyms > 0:
            recall_at_10 = matched_toponyms / total_toponyms

            percentage_toponyms_without_valid_search_arguments = nof_toponyms_without_valid_search_arguments / total_toponyms
            percentage_toponyms_without_candidates = nof_toponyms_without_candidates / total_toponyms
            percentage_toponyms_without_correct_candidates = (nof_toponyms_with_candidates - matched_toponyms) / total_toponyms
            percentage_toponyms_with_fatal_errors = toponyms_with_fatal_errors / total_toponyms
            percentage_too_many_generated_toponyms = too_many_generated_toponyms / total_toponyms
        else:
            recall_at_10 = \
                percentage_toponyms_without_valid_search_arguments = \
                percentage_toponyms_without_candidates = \
                percentage_toponyms_without_correct_candidates = \
                percentage_too_many_generated_toponyms = \
                percentage_toponyms_with_fatal_errors = \
                0


        # Calculate average number of candidates per toponym
        avg_nof_candidates = total_nof_candidates / nof_toponyms_with_candidates if nof_toponyms_with_candidates > 0 else 0

        if nof_articles_with_too_few_generated_toponyms > 0:
            print(f'FATAL: articles with too few generated toponyms: {nof_articles_with_too_few_generated_toponyms}')
            print(f'FATAL: too few generated toponyms: {too_few_generated_toponyms}')


        return CandidateGenerationMetrics(
            recall_at_10=recall_at_10,
            percentage_toponyms_with_fatal_errors=percentage_toponyms_with_fatal_errors,
            percentage_toponyms_without_valid_search_arguments=percentage_toponyms_without_valid_search_arguments,
            percentage_toponyms_without_candidates=percentage_toponyms_without_candidates,
            percentage_toponyms_without_correct_candidates=percentage_toponyms_without_correct_candidates,
            percentage_too_many_generated_toponym_candidates=percentage_too_many_generated_toponyms,
            avg_nof_candidates = avg_nof_candidates,
            nof_articles_with_fatal_errors=articles_with_fatal_errors,
            nof_all_gt_toponyms=total_toponyms,
            nof_all_generated_toponyms=nof_all_generated_toponyms
        )

# Example usage
data_dir = "data"
output_dir = "output/baseline_mistral-large_candidate_generation/20241228_seed_42_100_articles"
evaluator = CandidateGenerationEvaluator(data_dir, output_dir)
metrics = evaluator.calculate_metrics()

print("--------\nCandidate generation metrics:")
print(f"Recall@10: {metrics.recall_at_10:.4f}")
print(f"Percentage of toponyms with fatal errors: {metrics.percentage_toponyms_with_fatal_errors:.4f}")
print(f"Percentage of toponyms without valid search arguments: {metrics.percentage_toponyms_without_valid_search_arguments:.4f}")
print(f"Percentage of toponyms without any candidates: {metrics.percentage_toponyms_without_candidates:.4f}")
print(f"Percentage of toponyms without correct candidates: {metrics.percentage_toponyms_without_correct_candidates:.4f}")
print(f"Percentage of too many generated toponym candidates: {metrics.percentage_too_many_generated_toponym_candidates:.4f}\n--------")

print(f"Number of all ground truth toponyms: {metrics.nof_all_gt_toponyms}")
print(f"Number of all generated toponyms: {metrics.nof_all_generated_toponyms}\n--------")

print(f"Average number of candidates per toponym if candidates were found: {metrics.avg_nof_candidates:.2f}")
print(f"Number of articles with fatal errors: {metrics.nof_articles_with_fatal_errors}")