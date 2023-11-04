import json
import logging
from functools import partial
from typing import Any

import pandas as pd
import typer
from cdp_backend.database import models as cdp_db_models
from cdp_backend.pipeline.transcript_model import Transcript
from cdp_data import CDPInstances
from cdp_data import datasets as cdp_datasets
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

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
        ).fetch()
    )

    # No emis return None
    if len(emis) == 0:
        return None

    # For each event minutes item connection, get the minutes item
    minutes_items = [emi.minutes_item_ref.get() for emi in emis]
    return [{"name": mi.name, "description": mi.description} for mi in minutes_items]


@app.command()
def minutes_items(
    start_date: str = "2022-01-01",
    end_date: str = "2022-08-01",
    max_duration_meeting_minutes: int = 120,
    include_votes: bool = False,
    debug: bool = False,
    supress_other_module_logging: bool = True,
) -> str:
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
    for council in tqdm(councils, desc="Processing councils"):
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

    # Get minutes items for each meeting
    log.info("Getting minutes items for each meeting.")
    transcripts_under_max_duration[
        "minutes_items"
    ] = transcripts_under_max_duration.event_id.apply(
        _get_minutes_items_for_each_meeting,
    )

    # Filter out meetings with no minutes items
    log.info("Filtering out meetings with no minutes items.")
    transcripts_with_minutes_items = transcripts_under_max_duration.dropna(
        axis=0, subset=["minutes_items"]
    ).copy()
    log.debug(
        f"Number of transcripts with minutes items: "
        f"{len(transcripts_with_minutes_items)}"
    )

    # Format into constructed dataset
    log.info("Formatting into constructed dataset.")
    transcripts_with_minutes_items = transcripts_with_minutes_items[
        [
            "transcript",
            "minutes_items",
            "council",
            "key",
        ]
    ]
    transcripts_with_minutes_items = transcripts_with_minutes_items.rename(
        columns={
            "key": "meta_session_key",
            "council": "meta_council",
        }
    )

    # Store minutes items dict as JSON string
    transcripts_with_minutes_items[
        "minutes_items"
    ] = transcripts_with_minutes_items.minutes_items.apply(json.dumps)

    # Save to disk
    log.info("Saving to disk.")
    transcripts_with_minutes_items.to_parquet("example-dataset.parquet")

    return "example-dataset.parquet"


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    app()
