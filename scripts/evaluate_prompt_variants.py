import json
import csv
from pathlib import Path
from statistics import mean


# ==========================
# Configuration
# ==========================

INPUT_FILE = Path(
    "outputs/prompt_optimization/prompt_variant_results.json"
)


OUTPUT_FILE = Path(
    "reports/prompt_variant_summary.csv"
)


# ==========================
# Attribute Matching
# ==========================

def check_label_match(
    label: str,
    category: str
) -> bool:
    """
    Check whether detected label
    contains target attribute.
    """

    label = label.lower()

    category = category.lower()


    keywords = {

        "sleeve": [
            "sleeve"
        ],

        "collar": [
            "collar"
        ],

        "button": [
            "button"
        ],

        "zipper": [
            "zipper"
        ]
    }


    for keyword in keywords.get(
        category,
        []
    ):

        if keyword in label:
            return True


    return False


# ==========================
# Evaluation Function
# ==========================

def evaluate_prompt(
    detections,
    category
):

    total_images = 0

    detected_images = 0

    matched_images = 0

    confidence_scores = []

    detection_count = 0


    for image_name, image_results in detections.items():

        total_images += 1


        if len(image_results) > 0:

            detected_images += 1


            detection_count += len(
                image_results
            )


            for result in image_results:

                score = result.get(
                    "score",
                    0
                )

                label = result.get(
                    "label",
                    ""
                )


                confidence_scores.append(
                    score
                )


                if check_label_match(
                    label,
                    category
                ):

                    matched_images += 1


                    break



    detection_rate = (
        detected_images / total_images
        if total_images
        else 0
    )


    match_rate = (
        matched_images / total_images
        if total_images
        else 0
    )


    avg_confidence = (
        mean(confidence_scores)
        if confidence_scores
        else 0
    )


    best_score = (
        max(confidence_scores)
        if confidence_scores
        else 0
    )


    return {

        "Total Images": total_images,

        "Detected Images": detected_images,

        "Detection Rate": round(
            detection_rate,
            4
        ),

        "Matched Images": matched_images,

        "Matched Label Rate": round(
            match_rate,
            4
        ),

        "Average Confidence": round(
            avg_confidence,
            4
        ),

        "Best Confidence": round(
            best_score,
            4
        ),

        "Total Detections": detection_count

    }



# ==========================
# Main
# ==========================

def main():

    print(
        "Loading:",
        INPUT_FILE
    )


    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)



    records = []


    for category, prompts in data.items():


        for prompt, results in prompts.items():


            metrics = evaluate_prompt(
                results,
                category
            )


            record = {

                "Category": category,

                "Prompt": prompt,

                **metrics

            }


            records.append(
                record
            )



    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )



    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:


        writer = csv.DictWriter(
            f,
            fieldnames=records[0].keys()
        )


        writer.writeheader()

        writer.writerows(
            records
        )



    print(
        "\nFinished!"
    )


    print(
        "Saved:",
        OUTPUT_FILE
    )


    print("\nSummary:")


    for r in records:

        print(r)



if __name__ == "__main__":

    main()