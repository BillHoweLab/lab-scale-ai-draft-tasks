import json
import logging
import os
from functools import partial
from typing import Any

import pandas as pd
import typer
from cdp_backend.database import models as cdp_db_models
from cdp_backend.pipeline.transcript_model import Transcript
from cdp_data import CDPInstances
from cdp_data import datasets as cdp_datasets
from cdp_data.utils import connect_to_infrastructure
from datasets import Dataset, DatasetDict  # type: ignore
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

app = typer.Typer()

###############################################################################


def _setup_logging(
    debug: bool = False,
    supress_other_module_logging: bool = True,
) -> None:
    # Handle logging
    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    if supress_other_module_logging:
        # Silence other loggers
        # cdp-data is noisy
        for log_name, log_obj in logging.Logger.manager.loggerDict.items():
            if log_name != __name__:
                log_obj.disabled = True  # type: ignore

    # Setup logging
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
    )


def _read_and_check_meeting_over_max_duration(
    transcript_path: str,
    max_duration_meeting_minutes: int = 120,
) -> tuple[bool, str | None]:
    # Read transcript
    with open(transcript_path) as open_file:
        transcript = Transcript.from_json(open_file.read())

    # Handle weird errors in the transcripts
    if len(transcript.sentences) == 0:
        too_long = True
    else:
        # Get duration of meeting by using the last sentence end_time (in seconds)
        meeting_duration = transcript.sentences[-1].end_time / 60

        # Check if meeting is too long
        too_long = meeting_duration > max_duration_meeting_minutes

    # Only concat all meeting content if meeting is not too long
    if too_long:
        meeting_content = None
    else:
        # Concat all meeting content into one string
        meeting_content = " ".join(
            [sentence.text.strip() for sentence in transcript.sentences]
        )

    return too_long, meeting_content


def _get_minutes_items_for_each_meeting(
    event_id: str,
) -> list[dict[str, Any]] | None:
    # Get event minutes item connections
    emis = list(
        cdp_db_models.EventMinutesItem.collection.filter(
            "event_ref",
            "==",
            f"event/{event_id}",
        )
        .order(
            "index",
        )
        .fetch()
    )

    # No emis return None
    if len(emis) == 0:
        return None

    # For each event minutes item connection, get the minutes item
    minutes_items = [emi.minutes_item_ref.get() for emi in emis]
    return [{"name": mi.name, "description": mi.description} for mi in minutes_items]


@app.command()
def get_minutes_items(
    start_date: str = "2020-01-01",
    end_date: str = "2023-06-01",
    max_duration_meeting_minutes: int = 120,
    debug: bool = False,
    supress_other_module_logging: bool = True,
) -> None:
    _setup_logging(
        debug=debug,
        supress_other_module_logging=supress_other_module_logging,
    )

    # Councils of interest list
    councils = [
        CDPInstances.Seattle,
        CDPInstances.KingCounty,
        CDPInstances.Portland,
        CDPInstances.Denver,
        CDPInstances.Alameda,
        CDPInstances.Boston,
        CDPInstances.Charlotte,
        CDPInstances.SanJose,
        CDPInstances.MountainView,
        CDPInstances.Milwaukee,
        CDPInstances.LongBeach,
        CDPInstances.Albuquerque,
        CDPInstances.Richmond,
        CDPInstances.Louisville,
        CDPInstances.Atlanta,
    ]

    # Pull transcripts for councils of interest
    transcript_dataframes = []
    log.info("Getting base meeting dataset for each council.")
    for council in tqdm(councils, desc="Processing councils", leave=False):
        council_data = cdp_datasets.get_session_dataset(
            council,
            start_datetime=start_date,
            end_datetime=end_date,
            store_transcript=True,
            raise_on_error=False,
            replace_py_objects=True,
            tqdm_kws={"leave": False},
        )

        # Add the council name to the dataframe
        council_data["council"] = council

        # Append to list of dataframes
        transcript_dataframes.append(council_data)

    # Concatenate dataframes
    all_transcripts = pd.concat(transcript_dataframes)

    # Log current number of transcripts
    log.debug(f"Number of unfiltered transcripts: {len(all_transcripts)}")

    # Filter out transcripts that are too long
    log.info(
        f"Filtering out meetings longer than {max_duration_meeting_minutes} minutes."
    )

    # Create partial function to pass to thread_map
    read_and_check_meeting_over_max_duration = partial(
        _read_and_check_meeting_over_max_duration,
        max_duration_meeting_minutes=max_duration_meeting_minutes,
    )
    # Create a mask of bools for each transcript
    # Final return of process map is a list of tuple[bool, str | None]]
    # We need the mask to be a list of bool
    # And we need the read transcripts to be a list of str
    results = process_map(
        read_and_check_meeting_over_max_duration,
        all_transcripts["transcript_path"],
        desc="Filtering out long meetings",
    )
    mask = pd.Series([result[0] for result in results])
    read_transcripts = [result[1] for result in results]

    # Replace the transcript path with the read transcript
    all_transcripts["transcript"] = read_transcripts

    # Actual filter
    transcripts_under_max_duration = all_transcripts.loc[~mask].copy()
    log.debug(
        f"Number of transcripts under {max_duration_meeting_minutes} minutes: "
        f"{len(transcripts_under_max_duration)}"
    )

    # Ensure drop that there is a transcript
    transcripts_under_max_duration = transcripts_under_max_duration.dropna(
        subset=["transcript"],
    )

    # Get minutes items for each meeting
    # In order to use thread_map, we need to do this with group_by council
    # so that we can connect to the correct infrastructure
    t_with_mi_dfs = []
    for council_slug, group in tqdm(
        transcripts_under_max_duration.groupby("council"),
        desc="Getting minutes items for each council",
        leave=False,
    ):
        # Connect to infrastructure
        connect_to_infrastructure(council_slug)

        # Make copy
        t_with_mi_df = group.copy()

        # Get minutes items for each meeting
        t_with_mi_df["minutes_items"] = thread_map(
            _get_minutes_items_for_each_meeting,
            t_with_mi_df["event_id"],
            desc="Getting minutes items for each meeting",
            leave=False,
        )

        # Append to dfs list
        t_with_mi_dfs.append(t_with_mi_df)

    # Concat
    transcripts_with_mi = pd.concat(t_with_mi_dfs, ignore_index=True)

    # Filter out meetings with no minutes items
    log.info("Filtering out meetings with no minutes items.")
    transcripts_with_mi = transcripts_with_mi.dropna(
        axis=0, subset=["minutes_items"]
    ).copy()
    log.debug(
        f"Number of transcripts with minutes items: " f"{len(transcripts_with_mi)}"
    )

    # Format into constructed dataset
    log.info("Formatting into constructed dataset.")
    transcripts_with_mi = transcripts_with_mi[
        [
            "transcript",
            "minutes_items",
            "council",
            "key",
        ]
    ]
    transcripts_with_mi = transcripts_with_mi.rename(
        columns={
            "key": "meta_session_key",
            "council": "meta_council",
        }
    )

    # Store minutes items dict as JSON string
    transcripts_with_mi["minutes_items"] = transcripts_with_mi.minutes_items.apply(
        json.dumps
    )

    # Save to disk
    log.info("Saving to disk.")
    transcripts_with_mi = transcripts_with_mi.reset_index(drop=True)
    transcripts_with_mi.to_parquet(
        "councils-in-action-minutes-items-prediction.parquet"
    )


@app.command()
def upload_minutes_items(
    debug: bool = False,
    supress_other_module_logging: bool = True,
) -> None:
    _setup_logging(
        debug=debug,
        supress_other_module_logging=supress_other_module_logging,
    )

    # Read in dataset
    df = pd.read_parquet("councils-in-action-minutes-items-prediction.parquet")

    # Split out holdout councils
    # Holdout councils are those that we will not use for training
    # We will use them for validation
    holdout_councils = [
        CDPInstances.Milwaukee,
        CDPInstances.KingCounty,
        CDPInstances.Boston,
    ]

    # Filter out holdout councils
    valid_set = df.loc[df.meta_council.isin(holdout_councils)].copy()
    train_set = df.loc[~df.meta_council.isin(holdout_councils)].copy()

    # Create splits
    train_set, test_set = train_test_split(
        train_set,
        test_size=0.2,
        random_state=42,
        stratify=train_set["meta_council"],
    )

    # Print ds info
    # print the overall split sizes
    # then print the counts of meta council for train and test sets
    print(
        f"Split sizes, "
        f"train: {len(train_set)} "
        f"({round(len(train_set) / len(df), 3)}); "
        f"test: {len(test_set)} "
        f"({round(len(test_set) / len(df), 3)}); "
        f"valid: {len(valid_set)} "
        f"({round(len(valid_set) / len(df), 3)})"
    )

    # Create dataset dict
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_set, preserve_index=False),
            "test": Dataset.from_pandas(test_set, preserve_index=False),
            "valid": Dataset.from_pandas(valid_set, preserve_index=False),
        }
    )

    # Shuffle
    dataset_dict = dataset_dict.shuffle(seed=42)

    # Load env and get HF_TOKEN
    load_dotenv()
    token = os.environ["HF_TOKEN"]

    # Push to hub
    dataset_dict.push_to_hub(
        "evamaxfield/councils-in-action-minutes-items-prediction",
        token=token,
    )


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
