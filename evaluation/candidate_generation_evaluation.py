import os
import pickle

from pydantic import BaseModel, Field
from geopy.distance import geodesic
import numpy as np

from data_handler.xml_parsing import XMLDataHandler
from models.candidates import CandidateGenerationOutput, CandidateGenerationState, GeoCodingState


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

    def load_generation_for_article(self, docid: str) -> CandidateGenerationOutput | CandidateGenerationState | GeoCodingState | dict:
        file_path = os.path.join(self.output_directory, f"{docid}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Generated candidates for docid {docid} not found in {self.output_directory}.")
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def calculate_candidate_generation_metrics(self) -> CandidateGenerationMetrics:
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

        generated_by_retry = 0
        article_with_critic = 0

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

            if generation_for_article.reflected_prompt is not None:
                article_with_critic += 1

            # Retrieve toponyms for the current article
            article_toponyms = self.data_handler.get_toponyms_for_article(docid)

            nof_article_generated_toponyms = len(
                generation_for_article.toponyms_with_candidates + generation_for_article.invalid_toponyms)
            nof_all_generated_toponyms += nof_article_generated_toponyms

            if generation_for_article.fatal_errors:
                articles_with_fatal_errors += 1
                print(f"Fatal error for article {docid}: {generation_for_article.fatal_errors}")
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
                                if toponym.toponym_with_search_arguments.generated_by_retry:
                                    generated_by_retry += 1
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

        print(f'Generated by retry: {generated_by_retry}, articles with critic: {article_with_critic}')
        print(f'w/o_critic: {(matched_toponyms-generated_by_retry)/(total_toponyms)}\n'
              f'w_critic: {matched_toponyms/(total_toponyms)}\n'
              f'critic_percentage: {generated_by_retry/(total_toponyms)}')
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

    def calculate_candidate_resolution_metrics(self,
                                               directory: str = "",
                                               k=161):
        """
        Calculate Accuracy@k and AUC for geocoding.

        Parameters:
        - ground_truth_coords: List of tuples [(lat1, lon1), ...] for ground truth points.
        - predicted_coords: List of tuples [(lat2, lon2), ...] for predicted points.
        - k: Threshold distance in km for Accuracy@k (default 161 km).

        Returns:
        - accuracy_at_k: Fraction of points within k km.
        - auc: Area under the curve value.
        """
        articles_with_resolution_fatal_errors = 0

        total_toponyms = 0
        topos_with_incorrect_geonameid = 0
        topos_without_geonameid = 0

        error_distances = []

        self.data_handler.parse_xml("LGL_test.xml")

        # Iterate through generated candidate files
        for candidate_file in os.listdir(directory):
            if not candidate_file.endswith(".pkl"):
                continue

            docid = candidate_file.replace(".pkl", "")

            try:
                generation_for_article = self.load_generation_for_article(docid)
                if not isinstance(generation_for_article, GeoCodingState):
                    generation_for_article = GeoCodingState(**generation_for_article)
            except FileNotFoundError:
                continue
            # Retrieve toponyms for the current article
            article_toponyms = self.data_handler.get_toponyms_for_article(docid)

            if generation_for_article.resolution_fatal_errors:
                articles_with_resolution_fatal_errors += 1
                print(f"Fatal error for article {docid}: {generation_for_article.resolution_fatal_errors}")

            selected_candidates = generation_for_article.valid_geocoded_toponyms.copy()
            toponyms_with_candidates = generation_for_article.toponyms_with_candidates.copy()
            for gt_toponym in article_toponyms:
                total_toponyms += 1
                correct_geonameid = gt_toponym["geonameid"]
                correct_latitude = gt_toponym["lat"]
                correct_longitude = gt_toponym["lon"]
                gt_coords = (correct_latitude, correct_longitude)
                generated_coords = None

                for toponym in selected_candidates:
                    if toponym.toponym.casefold() == gt_toponym["phrase"].casefold():
                        if toponym.selected_candidate_geonameId in [None, 0]:
                            topos_without_geonameid += 1
                            selected_candidates.remove(toponym)
                            break
                        for item in toponyms_with_candidates:
                            if item.toponym_with_search_arguments.toponym.casefold() == toponym.toponym.casefold():
                                # get the candidate with the correct geonameid
                                incorrect_geonameid = True
                                for candidate in item.candidates:
                                    if toponym.selected_candidate_geonameId == candidate["geonameId"]:
                                        generated_coords = (candidate["lat"], candidate["lng"])
                                        error_distances.append(geodesic(gt_coords, generated_coords).kilometers)
                                        incorrect_geonameid = False
                                        break
                                if incorrect_geonameid:
                                    topos_with_incorrect_geonameid += 1
                                toponyms_with_candidates.remove(item)
                                break
                        selected_candidates.remove(toponym)
                        break

        # Accuracy@k
        within_k = [d <= k for d in error_distances]
        accuracy_at_k = sum(within_k) / len(error_distances)

        # AUC
        def calculate_auc(sorted_values):
            max_error = 20039  # Earth's circumference in km / 2 (maximum possible distance)
            size = len(sorted_values)
            if size <= 1:
                return 0.0

            h = 1  # step size
            sum = 0.5 * (np.log(1 + sorted_values[0]) / np.log(max_error) + np.log(
                1 + sorted_values[-1]) / np.log(max_error))  # initial area

            for i in range(1, size - 1):
                sum += np.log(1 + sorted_values[i]) / np.log(max_error)

            auc = sum * h / (size - 1)
            return auc

        sorted_error_distances = sorted(error_distances)  # assuming error_distances is a dictionary with error error_distances
        auc = calculate_auc(sorted_error_distances)
        print(f"AUC: {auc}")

        # Mean error distance
        mean_error_distance = np.mean(error_distances)
        print(f"Mean error distance: {mean_error_distance}")

        # Median error distance
        median_error_distance = np.median(error_distances)
        print(f"Median error distance: {median_error_distance}")

        return accuracy_at_k, auc


# Example usage
if __name__ == "__main__":
    data_dir = "data"
    output_dir = "output/reflective_candidate_resolution/fatal_error_and_invalid_correction/llama-3.3-70b-instruct_with_mistral-large-instruct_critic/20250120_seed_24_1000_articles"
    evaluator = CandidateGenerationEvaluator(data_dir, output_dir)

    accuracy_at_161, auc = evaluator.calculate_candidate_resolution_metrics(directory=output_dir)
    # metrics = evaluator.calculate_candidate_generation_metrics()
    #
    # print("--------\nCandidate generation metrics:")
    # print(f"Recall@10: {metrics.recall_at_10:.4f}")
    # print(f"Percentage of toponyms with fatal errors: {metrics.percentage_toponyms_with_fatal_errors:.4f}")
    # print(f"Percentage of toponyms without valid search arguments: {metrics.percentage_toponyms_without_valid_search_arguments:.4f}")
    # print(f"Percentage of toponyms without any candidates: {metrics.percentage_toponyms_without_candidates:.4f}")
    # print(f"Percentage of toponyms without correct candidates: {metrics.percentage_toponyms_without_correct_candidates:.4f}")
    # print(f"Percentage of too many generated toponym candidates: {metrics.percentage_too_many_generated_toponym_candidates:.4f}\n--------")
    #
    # print(f"Number of all ground truth toponyms: {metrics.nof_all_gt_toponyms}")
    # print(f"Number of all generated toponyms: {metrics.nof_all_generated_toponyms}\n--------")
    #
    # print(f"Average number of candidates per toponym if candidates were found: {metrics.avg_nof_candidates:.2f}")
    # print(f"Number of articles with fatal errors: {metrics.nof_articles_with_fatal_errors}")