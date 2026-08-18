import hashlib
import json
import math
import re
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

RUNS: dict[str, dict[str, Any]] = {}
LOCK = threading.RLock()

SAFE_MIN = -9007199254740991
SAFE_MAX = 9007199254740991

TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# ============================================================
# BASIC HELPERS
# ============================================================

def is_safe_int(x: Any) -> bool:
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and SAFE_MIN <= x <= SAFE_MAX
    )


def is_finite_number(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def parse_timestamp(x: Any):
    """
    Accept exactly:
      YYYY-MM-DDTHH:mm:ss
      YYYY-MM-DDTHH:mm:ss.s
      YYYY-MM-DDTHH:mm:ss.ss
      YYYY-MM-DDTHH:mm:ss.sss
    followed by Z or ±HH:mm.

    Return UTC datetime.
    """
    if not isinstance(x, str):
        return None

    if TS_RE.fullmatch(x) is None:
        return None

    try:
        s = x[:-1] + "+00:00" if x.endswith("Z") else x
        dt = datetime.fromisoformat(s)

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)

    except Exception:
        return None


def utf8_key(s: str) -> bytes:
    return s.encode("utf-8")


def compact_json(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def canonical_request(obj: Any) -> str:
    """
    Used only for detecting exact logical request replay.
    JSON object key order does not matter for replay identity.
    """
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_compact_json(obj: Any) -> str:
    return hashlib.sha256(
        compact_json(obj).encode("utf-8")
    ).hexdigest()


def add_code(codes: list[str], code: str) -> None:
    if code not in codes:
        codes.append(code)


def sort_codes(codes: list[str]) -> list[str]:
    return sorted(set(codes), key=utf8_key)


# ============================================================
# SELECTION ROW VALIDATION
# ============================================================

def validate_selection(req: dict[str, Any]) -> list[str]:
    codes: list[str] = []

    if req.get("phase") != "select":
        add_code(codes, "INVALID_INPUT")

    # runId: explicitly non-empty, <=128 characters.
    run_id = req.get("runId")

    if (
        not isinstance(run_id, str)
        or len(run_id) == 0
        or len(run_id) > 128
    ):
        add_code(codes, "INVALID_INPUT")

    # forbiddenFeatures must be an array.
    forbidden = req.get("forbiddenFeatures")

    if not isinstance(forbidden, list):
        add_code(codes, "INVALID_INPUT")
        forbidden = []

    # Feature names are strings.
    for name in forbidden:
        if not isinstance(name, str):
            add_code(codes, "INVALID_INPUT")

    # Positive integer.
    limit = req.get("numTrialsLimit")

    if not is_safe_int(limit) or limit <= 0:
        add_code(codes, "INVALID_INPUT")

    # Selection rows must be non-empty.
    rows = req.get("rows")

    if not isinstance(rows, list) or len(rows) == 0:
        add_code(codes, "INVALID_INPUT")
        rows = []

    # Trials must be an array.
    trials = req.get("trials")

    if not isinstance(trials, list):
        add_code(codes, "INVALID_INPUT")
        trials = []

    # --------------------------------------------------------
    # ROWS
    # --------------------------------------------------------

    row_ids: set[str] = set()

    for row in rows:

        if not isinstance(row, dict):
            add_code(codes, "INVALID_INPUT")
            continue

        row_id = row.get("id")

        # The contract says IDs are unique within the array.
        # It does NOT require them to be non-empty.
        if not isinstance(row_id, str):
            add_code(codes, "INVALID_INPUT")
        else:
            if row_id in row_ids:
                add_code(codes, "INVALID_INPUT")
            row_ids.add(row_id)

        # entity is data. Only its presence/type is validated.
        if not isinstance(row.get("entity"), str):
            add_code(codes, "INVALID_INPUT")

        # version is non-negative safe integer.
        version = row.get("version")

        if not is_safe_int(version) or version < 0:
            add_code(codes, "INVALID_INPUT")

        # split is exactly TRAIN/EVAL.
        if row.get("split") not in ("TRAIN", "EVAL"):
            add_code(codes, "INVALID_INPUT")

        # Both timestamps must be syntactically valid instants.
        event_dt = parse_timestamp(row.get("eventTime"))
        prediction_dt = parse_timestamp(row.get("predictionTime"))

        if event_dt is None:
            add_code(codes, "INVALID_INPUT")

        if prediction_dt is None:
            add_code(codes, "INVALID_INPUT")

        # IMPORTANT:
        # Do NOT reject eventTime > predictionTime.
        # That restriction is not in the contract.

        features = row.get("features")

        if not isinstance(features, dict):
            add_code(codes, "INVALID_INPUT")
            continue

        for feature_name, feature in features.items():

            # Feature names are JSON object keys, therefore strings.
            # Empty names are not prohibited by the specification.
            if not isinstance(feature_name, str):
                add_code(codes, "INVALID_INPUT")
                continue

            if not isinstance(feature, dict):
                add_code(codes, "INVALID_INPUT")
                continue

            # Both fields are required.
            if "value" not in feature:
                add_code(codes, "INVALID_INPUT")

            if "availableAt" not in feature:
                add_code(codes, "INVALID_INPUT")
                continue

            available_dt = parse_timestamp(
                feature.get("availableAt")
            )

            if available_dt is None:
                add_code(codes, "INVALID_INPUT")

            # IMPORTANT:
            # availableAt > predictionTime is NOT malformed input.
            # It merely makes that feature ineligible.

    # --------------------------------------------------------
    # TRIALS
    # --------------------------------------------------------

    trial_ids: set[int] = set()

    for trial in trials:

        if not isinstance(trial, dict):
            add_code(codes, "INVALID_INPUT")
            continue

        trial_id = trial.get("trialId")

        if not is_safe_int(trial_id) or trial_id < 0:
            add_code(codes, "INVALID_INPUT")
        else:
            if trial_id in trial_ids:
                add_code(codes, "INVALID_INPUT")
            trial_ids.add(trial_id)

        if trial.get("status") not in ("SUCCEEDED", "FAILED"):
            add_code(codes, "INVALID_INPUT")

        if not is_finite_number(trial.get("evalMetric")):
            add_code(codes, "INVALID_INPUT")

    # More trials than permitted is a contract failure.
    if (
        is_safe_int(limit)
        and limit > 0
        and len(trials) > limit
    ):
        add_code(codes, "TRIAL_LIMIT_EXCEEDED")

    return sort_codes(codes)


# ============================================================
# SELECTION
# ============================================================

def perform_selection(req: dict[str, Any]) -> dict[str, Any]:

    codes = validate_selection(req)

    run_id = req.get("runId")
    if not isinstance(run_id, str):
        run_id = None

    result = {
        "runId": run_id,
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": codes,
    }

    # Any actual malformed selection has null digest.
    if codes:
        return result

    rows = req["rows"]
    forbidden = set(req["forbiddenFeatures"])

    # --------------------------------------------------------
    # DEDUPLICATION
    #
    # Key:
    #   [entity, UTC(eventTime)]
    #
    # Highest version wins.
    # If version ties, UTF-8-smallest ID wins.
    # --------------------------------------------------------

    retained: dict[tuple[str, datetime], dict[str, Any]] = {}

    for row in rows:

        event_utc = parse_timestamp(row["eventTime"])

        key = (
            row["entity"],
            event_utc,
        )

        current = retained.get(key)

        if current is None:
            retained[key] = row
            continue

        if row["version"] > current["version"]:
            retained[key] = row

        elif row["version"] == current["version"]:

            if utf8_key(row["id"]) < utf8_key(current["id"]):
                retained[key] = row

    retained_rows = list(retained.values())

    # --------------------------------------------------------
    # SHARED FEATURES
    #
    # A feature must appear in EVERY retained row.
    # --------------------------------------------------------

    shared_features: set[str] | None = None

    for row in retained_rows:

        names = set(row["features"].keys())

        if shared_features is None:
            shared_features = names
        else:
            shared_features &= names

    if shared_features is None:
        shared_features = set()

    # --------------------------------------------------------
    # POINT-IN-TIME FEATURE FILTER
    #
    # Feature eligible iff:
    #   - shared by every retained row
    #   - not forbidden
    #   - availableAt <= predictionTime for EVERY retained row
    #
    # A future feature is excluded, NOT an INVALID_INPUT.
    # --------------------------------------------------------

    eligible_features: list[str] = []

    for feature_name in shared_features:

        if feature_name in forbidden:
            continue

        eligible = True

        for row in retained_rows:

            available = parse_timestamp(
                row["features"][feature_name]["availableAt"]
            )

            prediction = parse_timestamp(
                row["predictionTime"]
            )

            # These were already validated above.
            if available is None or prediction is None:
                eligible = False
                break

            if available > prediction:
                eligible = False
                break

        if eligible:
            eligible_features.append(feature_name)

    eligible_features.sort(key=utf8_key)

    # --------------------------------------------------------
    # SPLIT IDS
    # --------------------------------------------------------

    train_ids = sorted(
        [
            row["id"]
            for row in retained_rows
            if row["split"] == "TRAIN"
        ],
        key=utf8_key,
    )

    eval_ids = sorted(
        [
            row["id"]
            for row in retained_rows
            if row["split"] == "EVAL"
        ],
        key=utf8_key,
    )

    # --------------------------------------------------------
    # TRIAL SELECTION
    #
    # Only SUCCEEDED + finite metrics.
    #
    # Highest metric.
    # Exact metric tie -> smallest integer trialId.
    # --------------------------------------------------------

    eligible_trials = []

    for trial in req["trials"]:

        if trial["status"] != "SUCCEEDED":
            continue

        if not is_finite_number(trial["evalMetric"]):
            continue

        eligible_trials.append(trial)

    if not eligible_trials:

        result["trainRowIds"] = train_ids
        result["evalRowIds"] = eval_ids
        result["featureNames"] = eligible_features
        result["reasonCodes"] = ["NO_SUCCESSFUL_TRIAL"]

        return result

    best = eligible_trials[0]

    for trial in eligible_trials[1:]:

        candidate_metric = float(trial["evalMetric"])
        best_metric = float(best["evalMetric"])

        if candidate_metric > best_metric:
            best = trial

        elif (
            candidate_metric == best_metric
            and trial["trialId"] < best["trialId"]
        ):
            best = trial

    # --------------------------------------------------------
    # DATASET DIGEST
    #
    # Exact shape/order:
    # {
    #   trainRowIds,
    #   evalRowIds,
    #   featureNames
    # }
    # --------------------------------------------------------

    result["selectedTrialId"] = best["trialId"]
    result["trainRowIds"] = train_ids
    result["evalRowIds"] = eval_ids
    result["featureNames"] = eligible_features

    digest_object = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": eligible_features,
    }

    result["datasetDigest"] = sha256_compact_json(
        digest_object
    )

    return result


# ============================================================
# EVALUATION
# ============================================================

def validate_evaluation(req: dict[str, Any]):

    codes: list[str] = []

    if req.get("phase") != "evaluate":
        add_code(codes, "INVALID_INPUT")

    run_id = req.get("runId")

    if not isinstance(run_id, str) or not run_id:
        add_code(codes, "INVALID_INPUT")

    trial_id = req.get("selectedTrialId")

    if not is_safe_int(trial_id) or trial_id < 0:
        add_code(codes, "INVALID_INPUT")

    digest = req.get("datasetDigest")

    if (
        not isinstance(digest, str)
        or HEX64_RE.fullmatch(digest) is None
    ):
        add_code(codes, "INVALID_INPUT")

    metric_floor = req.get("metricFloor")

    if (
        not is_finite_number(metric_floor)
        or not 0 <= float(metric_floor) <= 1
    ):
        add_code(codes, "INVALID_INPUT")

    required_slices = req.get("requiredSlices")

    if not isinstance(required_slices, dict):
        add_code(codes, "INVALID_INPUT")
        required_slices = {}

    for name, floor in required_slices.items():

        if not isinstance(name, str) or not name:
            add_code(codes, "INVALID_INPUT")

        if (
            not is_finite_number(floor)
            or not 0 <= float(floor) <= 1
        ):
            add_code(codes, "INVALID_INPUT")

    rows = req.get("rows")

    if not isinstance(rows, list):
        add_code(codes, "INVALID_INPUT")
        rows = []

    bytes_processed = req.get("bytesProcessed")
    max_bytes = req.get("maxBytes")

    if (
        not is_safe_int(bytes_processed)
        or bytes_processed < 0
    ):
        add_code(codes, "INVALID_INPUT")

    if (
        not is_safe_int(max_bytes)
        or max_bytes < 0
    ):
        add_code(codes, "INVALID_INPUT")

    return sort_codes(codes), rows, required_slices


def round12(value: float) -> float:
    return float(f"{value:.12f}")


def perform_evaluation(req: dict[str, Any]):

    base_codes, rows, required_slices = validate_evaluation(req)

    run_id = req.get("runId")
    selected_trial = req.get("selectedTrialId")
    digest = req.get("datasetDigest")

    bytes_processed = (
        req.get("bytesProcessed")
        if is_safe_int(req.get("bytesProcessed"))
        else 0
    )

    result = {
        "runId": run_id if isinstance(run_id, str) else None,
        "selectedTrialId": (
            selected_trial
            if is_safe_int(selected_trial)
            else None
        ),
        "datasetDigest": (
            digest
            if isinstance(digest, str)
            else None
        ),
        "testMetric": None,
        "criticalSlicePass": False,
        "decision": "reject",
        "bytesProcessed": bytes_processed,
        "reasonCodes": [],
    }

    codes = list(base_codes)

    # --------------------------------------------------------
    # FROZEN LINEAGE
    # --------------------------------------------------------

    stored = None

    if not base_codes:

        with LOCK:
            stored = RUNS.get(run_id)

        if stored is None:

            add_code(codes, "INVALID_LINEAGE")

        else:

            frozen = stored["response"]

            if (
                frozen["selectedTrialId"] is None
                or frozen["datasetDigest"] is None
                or frozen["reasonCodes"]
                or frozen["selectedTrialId"] != selected_trial
                or frozen["datasetDigest"] != digest
            ):
                add_code(codes, "INVALID_LINEAGE")

    # --------------------------------------------------------
    # BYTE LIMIT
    # --------------------------------------------------------

    max_bytes = req.get("maxBytes")

    if (
        is_safe_int(bytes_processed)
        and is_safe_int(max_bytes)
        and bytes_processed > max_bytes
    ):
        add_code(codes, "BYTE_LIMIT")

    # --------------------------------------------------------
    # FINAL TEST ROW VALIDATION
    # --------------------------------------------------------

    invalid_test_row = False

    for row in rows:

        if not isinstance(row, dict):
            invalid_test_row = True
            break

        label = row.get("label")
        prediction = row.get("prediction")
        slice_name = row.get("slice")

        if (
            not is_safe_int(label)
            or label not in (0, 1)
            or not is_safe_int(prediction)
            or prediction not in (0, 1)
            or not isinstance(slice_name, str)
            or not slice_name
        ):
            invalid_test_row = True
            break

    # Empty or invalid final-test data:
    # no metric, no aggregate gate, no slice gate.
    if len(rows) == 0 or invalid_test_row:

        if invalid_test_row:
            add_code(codes, "INVALID_TEST_ROW")

        result["testMetric"] = None
        result["criticalSlicePass"] = False
        result["decision"] = "reject"
        result["reasonCodes"] = sort_codes(codes)

        return result

    # --------------------------------------------------------
    # AGGREGATE ACCURACY
    # --------------------------------------------------------

    correct = sum(
        1
        for row in rows
        if row["label"] == row["prediction"]
    )

    test_metric = round12(correct / len(rows))

    result["testMetric"] = test_metric

    metric_floor = float(req["metricFloor"])

    if test_metric < metric_floor:
        add_code(codes, "AGGREGATE_FLOOR")

    # --------------------------------------------------------
    # REQUIRED SLICES
    # --------------------------------------------------------

    slices: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        slices.setdefault(row["slice"], []).append(row)

    required_slices_pass = True

    for slice_name in sorted(
        required_slices.keys(),
        key=utf8_key,
    ):

        if slice_name not in slices:

            add_code(
                codes,
                f"MISSING_SLICE:{slice_name}",
            )

            required_slices_pass = False
            continue

        slice_rows = slices[slice_name]

        slice_correct = sum(
            1
            for row in slice_rows
            if row["label"] == row["prediction"]
        )

        slice_metric = round12(
            slice_correct / len(slice_rows)
        )

        floor = float(required_slices[slice_name])

        if slice_metric < floor:

            add_code(
                codes,
                f"SLICE_FLOOR:{slice_name}",
            )

            required_slices_pass = False

    # --------------------------------------------------------
    # criticalSlicePass
    #
    # Does NOT represent aggregate or byte checks.
    # --------------------------------------------------------

    result["criticalSlicePass"] = bool(
        not base_codes
        and stored is not None
        and "INVALID_LINEAGE" not in codes
        and required_slices_pass
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    admit = (
        not base_codes
        and stored is not None
        and "INVALID_LINEAGE" not in codes
        and not invalid_test_row
        and test_metric >= metric_floor
        and required_slices_pass
        and is_safe_int(bytes_processed)
        and is_safe_int(max_bytes)
        and bytes_processed <= max_bytes
    )

    result["decision"] = (
        "admit" if admit else "reject"
    )

    result["reasonCodes"] = sort_codes(codes)

    return result


# ============================================================
# ENDPOINT
# ============================================================

@app.post("/bqml")
async def bqml(request: Request):

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    phase = body.get("phase")

    if phase not in ("select", "evaluate"):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    # ========================================================
    # SELECT
    # ========================================================

    if phase == "select":

        run_id = body.get("runId")

        if (
            isinstance(run_id, str)
            and len(run_id) > 0
            and len(run_id) <= 128
        ):

            identity = canonical_request(body)

            with LOCK:
                old = RUNS.get(run_id)

            if old is not None:

                if old["requestIdentity"] == identity:
                    return JSONResponse(
                        status_code=200,
                        content=old["response"],
                    )

                return JSONResponse(
                    status_code=409,
                    content={"error": "RUN_ID_CONFLICT"},
                )

        result = perform_selection(body)

        if (
            isinstance(run_id, str)
            and len(run_id) > 0
            and len(run_id) <= 128
        ):

            identity = canonical_request(body)

            with LOCK:

                old = RUNS.get(run_id)

                if old is not None:

                    if old["requestIdentity"] == identity:
                        return JSONResponse(
                            status_code=200,
                            content=old["response"],
                        )

                    return JSONResponse(
                        status_code=409,
                        content={"error": "RUN_ID_CONFLICT"},
                    )

                RUNS[run_id] = {
                    "requestIdentity": identity,
                    "response": result,
                }

        return JSONResponse(
            status_code=200,
            content=result,
        )

    # ========================================================
    # EVALUATE
    # ========================================================

    result = perform_evaluation(body)

    return JSONResponse(
        status_code=200,
        content=result,
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True}
