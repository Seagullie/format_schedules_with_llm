import argparse
import time
from openai import OpenAI
from tqdm import tqdm

from constants.constants import SCHEDULE_CORRECTION_PROMPT_TEMPLATE
from constants.keys import OPEN_AI_API_KEY
from utils import get_schedule_paths_from_dir, read_schedule, save_schedule


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Format schedule JSON files with OpenAI LLM"
    )

    # Single file mode arguments
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input schedule JSON file",
        type=str,
        dest="path_to_schedule_json",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to save formatted schedule JSON file",
        type=str,
        dest="path_to_formatted_schedule_json",
    )

    # Batch mode arguments

    parser.add_argument(
        "-b",
        "--batch-mode",
        help="Batch mode flag",
        action="store_true",
        dest="batch_mode",
    )

    parser.add_argument(
        "-id",
        "--input-dir",
        help="Path to directory containing schedule JSON files. Requires -b flag",
        type=str,
        default="./input/gpt4o_batch",
        dest="path_to_schedules_dir",
    )
    parser.add_argument(
        "-od",
        "--output-dir",
        help="Path to directory for saving formatted schedule JSON files. Requires -b flag",
        type=str,
        default="./output/gpt4o_batch_output",
        dest="path_to_output_dir",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Name of the OpenAI model to use",
        type=str,
        default="gpt-4o",
        dest="llm_model_name",
    )

    return parser.parse_args()


args = parse_arguments()

# - - USER VARIABLES - -
# Path to schedule JSON file
path_to_schedule_json: str = args.path_to_schedule_json
# Path to use when saving corrected schedule JSON file
path_to_formatted_schedule_json: str = args.path_to_formatted_schedule_json

path_to_schedules_dir: str = args.path_to_schedules_dir
path_to_output_dir: str = args.path_to_output_dir

llm_model_name: str = args.llm_model_name

# - - USER VARIABLES END - -

# allowed requests per minute (more than enough in this case)
RPM = 5_000
TIME_PER_ONE_REQUEST = 60 / RPM

client = OpenAI(api_key=OPEN_AI_API_KEY)


def format_schedule(
    schedule_json_str: str, path_to_formatted_schedule_json: str
) -> float:
    """Formats schedule with OpenAI LLM. Returns elapsed time in seconds."""

    start_time = time.time()

    message: str = SCHEDULE_CORRECTION_PROMPT_TEMPLATE.format(schedule_json_str)

    response = client.chat.completions.create(
        model=llm_model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message,
                    }
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.25,
        max_completion_tokens=10000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    response_text = response.choices[0].message.content

    if not response_text:
        raise ValueError("Empty response from LLM")

    print(response_text)

    save_schedule(path_to_formatted_schedule_json, response_text)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Format schedule completed in {elapsed_time:.2f} seconds")

    return elapsed_time


if __name__ == "__main__":

    # either work with single schedule or batch
    if (
        not args.batch_mode
        and path_to_schedule_json
        and path_to_formatted_schedule_json
    ):
        schedule_json_str: str = read_schedule(path_to_schedule_json)

        format_schedule(schedule_json_str, path_to_formatted_schedule_json)

    elif args.batch_mode and path_to_schedules_dir and path_to_output_dir:
        schedule_paths = get_schedule_paths_from_dir(path_to_schedules_dir)

        for schedule_path in tqdm(
            schedule_paths, desc="Formatting schedules", unit="file"
        ):
            print("\n")
            schedule_json_str = read_schedule(schedule_path)
            elapsed_time = format_schedule(
                schedule_json_str,
                schedule_path.replace(path_to_schedules_dir, path_to_output_dir),
            )

            # Sleep to avoid rate limiting
            if elapsed_time < TIME_PER_ONE_REQUEST:
                time.sleep(TIME_PER_ONE_REQUEST - elapsed_time + 0.1)
    else:
        raise ValueError("No valid input/output paths provided")
