import os
import json
import argparse
from statistics import mean


# ==========================
# Load Results
# ==========================

def load_results(folder):

    json_results = []

    for file in os.listdir(folder):

        if file.endswith(".json"):

            path = os.path.join(
                folder,
                file
            )

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            json_results.append(data)

    return json_results


# ==========================
# Evaluation
# ==========================

def evaluate(results):

    total_images = 0
    detected_images = 0

    confidence_scores = []


    for result in results:

        for image_name, detections in result.items():

            total_images += 1


            # detection exists
            if len(detections) > 0:

                detected_images += 1


                for detection in detections:

                    confidence_scores.append(
                        detection["score"]
                    )


    detection_rate = (
        detected_images / total_images
        if total_images > 0
        else 0
    )


    average_confidence = (
        mean(confidence_scores)
        if len(confidence_scores) > 0
        else 0
    )


    return {

        "total_images": total_images,

        "detected_images": detected_images,

        "detection_rate": detection_rate,

        "average_confidence": average_confidence

    }


# ==========================
# Save Markdown Report
# ==========================

def save_report(metrics, output_path):

    output_dir = os.path.dirname(output_path)

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:


        f.write(
            "# Grounding DINO Evaluation Summary\n\n"
        )


        f.write(
            "| Metric | Value |\n"
        )

        f.write(
            "|---|---|\n"
        )


        f.write(
            f"| Total Images | {metrics['total_images']} |\n"
        )


        f.write(
            f"| Detected Images | {metrics['detected_images']} |\n"
        )


        f.write(
            f"| Detection Rate | {metrics['detection_rate']:.2%} |\n"
        )


        f.write(
            f"| Average Confidence | {metrics['average_confidence']:.4f} |\n"
        )



# ==========================
# Main
# ==========================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate Grounding DINO detection results"
    )


    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing json result files"
    )


    parser.add_argument(
        "--output",
        default="reports/grounding_evaluation_summary.md",
        help="Output markdown report path"
    )


    args = parser.parse_args()


    results = load_results(
        args.input
    )


    metrics = evaluate(
        results
    )


    print("\nEvaluation Results")
    print("----------------------")


    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )


    save_report(
        metrics,
        args.output
    )


    print(
        "\nSaved report:",
        args.output
    )



if __name__ == "__main__":

    main()