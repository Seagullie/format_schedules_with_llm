import argparse
from constants.constants import SCHEDULE_CORRECTION_PROMPT_TEMPLATE
from constants.keys import GOOGLE_API_KEY_OR_NONE
from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import generation_types

from utils import get_schedule_paths_from_dir, read_schedule, save_schedule

import time
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Format schedule JSON files using Gemini LLM"
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
        default="./input/gemini_batch",
        dest="path_to_schedules_dir",
    )
    parser.add_argument(
        "-od",
        "--output-dir",
        help="Path to directory for saving formatted schedule JSON files. Requires -b flag",
        type=str,
        default="./output/gemini_batch_output",
        dest="path_to_output_dir",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Name of Google LLM model to use",
        type=str,
        default="gemini-2.0-flash-exp",
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

# allowed requests per minute
RPM = 10
TIME_PER_ONE_REQUEST = 60 / RPM

configure(api_key=GOOGLE_API_KEY_OR_NONE)

# Create model configuration
generation_config = generation_types.GenerationConfig(
    temperature=0.25,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
    candidate_count=1,
    response_mime_type="application/json",
)

model = GenerativeModel(
    model_name=llm_model_name,
    generation_config=generation_config,
)


def format_schedule(
    schedule_json_str: str, path_to_formatted_schedule_json: str
) -> float:
    """Formats schedule with Gemini LLM. Returns elapsed time in seconds."""
    start_time = time.time()

    chat_session = model.start_chat()

    message: str = SCHEDULE_CORRECTION_PROMPT_TEMPLATE.format(schedule_json_str)
    response = chat_session.send_message(message)

    print(response.text)

    save_schedule(path_to_formatted_schedule_json, response.text)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Format schedule completed in {elapsed_time:.2f} seconds")

    return elapsed_time


# either work with single schedule or batch
if not args.batch_mode and path_to_schedule_json and path_to_formatted_schedule_json:
    schedule_json_str: str = read_schedule(path_to_schedule_json)

    format_schedule(schedule_json_str, path_to_formatted_schedule_json)
elif args.batch_mode and path_to_schedules_dir and path_to_output_dir:
    schedule_paths = get_schedule_paths_from_dir(path_to_schedules_dir)

    for schedule_path in tqdm(schedule_paths, desc="Formatting schedules", unit="file"):
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
