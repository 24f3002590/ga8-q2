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

RUNS = {}
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
# HELPERS
# ============================================================

def safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and SAFE_MIN <= x <= SAFE_MAX
    )


def finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def parse_timestamp(x):
    if not isinstance(x, str):
        return None

    if TS_RE.fullmatch(x) is None:
        return None

    try:
        if x.endswith("Z"):
            x = x[:-1] + "+00:00"

        dt = datetime.fromisoformat(x)

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)

    except Exception:
        return None


def utf8_key(x):
    return x.encode("utf-8")


def add_code(codes, code):
    if code not in codes:
        codes.append(code)


def sorted_codes(codes):
    return sorted(set(codes), key=utf8_key)


def compact_json(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def request_identity(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def digest(obj):
    return hashlib.sha256(
        compact_json(obj).encode("utf-8")
    ).hexdigest()


# ============================================================
# SELECTION VALIDATION
# ============================================================

def validate_select(req):
    codes = []

    if req.get("phase") != "select":
        add_code(codes, "INVALID_INPUT")

    run_id = req.get("runId")

    if (
        not isinstance(run_id, str)
        or len(run_id) == 0
        or len(run_id) > 128
    ):
        add_code(codes, "INVALID_INPUT")

    forbidden = req.get("forbiddenFeatures")

    if not isinstance(forbidden, list):
        add_code(codes, "INVALID_INPUT")
        forbidden = []

    for f in forbidden:
        if not isinstance(f, str):
            add_code(codes, "INVALID_INPUT")

    limit = req.get("numTrialsLimit")

    if not safe_int(limit) or limit <= 0:
        add_code(codes, "INVALID_INPUT")

    rows = req.get("rows")

    if not isinstance(rows, list) or len(rows) == 0:
        add_code(codes, "INVALID_INPUT")
        rows = []

    trials = req.get("trials")

    if not isinstance(trials, list):
        add_code(codes, "INVALID_INPUT")
        trials = []

    # --------------------------------------------------------
    # ROWS
    # --------------------------------------------------------

    seen_row_ids = set()

    for row in rows:

        if not isinstance(row, dict):
            add_code(codes, "INVALID_INPUT")
            continue

        row_id = row.get("id")
        entity = row.get("entity")

        # The examples use strings and the contract says IDs are
        # unique. Require strings, but don't impose whitespace rules.
        if not isinstance(row_id, str):
            add_code(codes, "INVALID_INPUT")
        else:
            if row_id in seen_row_ids:
                add_code(codes, "INVALID_INPUT")
            seen_row_ids.add(row_id)

        if not isinstance(entity, str):
            add_code(codes, "INVALID_INPUT")

        version = row.get("version")

        if not safe_int(version) or version < 0:
            add_code(codes, "INVALID_INPUT")

        if row.get("split") not in ("TRAIN", "EVAL"):
            add_code(codes, "INVALID_INPUT")

        if parse_timestamp(row.get("eventTime")) is None:
            add_code(codes, "INVALID_INPUT")

        if parse_timestamp(row.get("predictionTime")) is None:
            add_code(codes, "INVALID_INPUT")

        features = row.get("features")

        if not isinstance(features, dict):
            add_code(codes, "INVALID_INPUT")
            continue

        for name, feature in features.items():

            if not isinstance(name, str):
                add_code(codes, "INVALID_INPUT")
                continue

            if not isinstance(feature, dict):
                add_code(codes, "INVALID_INPUT")
                continue

            if "value" not in feature:
                add_code(codes, "INVALID_INPUT")

            if "availableAt" not in feature:
                add_code(codes, "INVALID_INPUT")
                continue

            if parse_timestamp(feature["availableAt"]) is None:
                add_code(codes, "INVALID_INPUT")

    # --------------------------------------------------------
    # TRIALS
    # --------------------------------------------------------

    seen_trial_ids = set()

    for trial in trials:

        if not isinstance(trial, dict):
            add_code(codes, "INVALID_INPUT")
            continue

        trial_id = trial.get("trialId")
        status = trial.get("status")

        if not safe_int(trial_id) or trial_id < 0:
            add_code(codes, "INVALID_INPUT")
        else:
            if trial_id in seen_trial_ids:
                add_code(codes, "INVALID_INPUT")
            seen_trial_ids.add(trial_id)

        if status not in ("SUCCEEDED", "FAILED"):
            add_code(codes, "INVALID_INPUT")

        # The metric is only relevant to eligibility for SUCCEEDED.
        # A FAILED trial must not become invalid merely because its
        # metric isn't finite.
        if status == "SUCCEEDED":
            if not finite_number(trial.get("evalMetric")):
                # This is not malformed input; it is simply an
                # ineligible successful trial.
                pass

    if (
        safe_int(limit)
        and limit > 0
        and len(trials) > limit
    ):
        add_code(codes, "TRIAL_LIMIT_EXCEEDED")

    return sorted_codes(codes)


# ============================================================
# SELECTION
# ============================================================

def perform_select(req):

    codes = validate_select(req)

    run_id = req.get("runId")

    result = {
        "runId": run_id if isinstance(run_id, str) else None,
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": codes,
    }

    if codes:
        return result

    rows = req["rows"]
    forbidden = set(req["forbiddenFeatures"])

    # --------------------------------------------------------
    # DEDUPLICATE
    #
    # Key = entity + UTC(eventTime)
    #
    # Highest version wins.
    # Equal version -> UTF-8-smallest ID.
    # --------------------------------------------------------

    retained = {}

    for row in rows:

        key = (
            row["entity"],
            parse_timestamp(row["eventTime"]),
        )

        old = retained.get(key)

        if old is None:
            retained[key] = row
            continue

        if row["version"] > old["version"]:
            retained[key] = row

        elif row["version"] == old["version"]:

            if utf8_key(row["id"]) < utf8_key(old["id"]):
                retained[key] = row

    retained_rows = list(retained.values())

    # --------------------------------------------------------
    # SHARED FEATURES
    # --------------------------------------------------------

    common = None

    for row in retained_rows:

        names = set(row["features"].keys())

        if common is None:
            common = names
        else:
            common &= names

    if common is None:
        common = set()

    # --------------------------------------------------------
    # POINT-IN-TIME FILTER
    #
    # IMPORTANT:
    # Future availableAt does NOT invalidate the row.
    # It only makes that feature unavailable for selection.
    # --------------------------------------------------------

    feature_names = []

    for name in common:

        if name in forbidden:
            continue

        eligible = True

        for row in retained_rows:

            available = parse_timestamp(
                row["features"][name]["availableAt"]
            )

            prediction = parse_timestamp(
                row["predictionTime"]
            )

            if available > prediction:
                eligible = False
                break

        if eligible:
            feature_names.append(name)

    feature_names.sort(key=utf8_key)

    # --------------------------------------------------------
    # TRAIN / EVAL IDS
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

    result["trainRowIds"] = train_ids
    result["evalRowIds"] = eval_ids
    result["featureNames"] = feature_names

    # --------------------------------------------------------
    # TRIAL SELECTION
    # --------------------------------------------------------

    eligible = []

    for trial in req["trials"]:

        if trial["status"] != "SUCCEEDED":
            continue

        metric = trial.get("evalMetric")

        if not finite_number(metric):
            continue

        eligible.append(trial)

    if not eligible:
        result["reasonCodes"] = ["NO_SUCCESSFUL_TRIAL"]
        return result

    # Max metric, then smallest integer trialId.
    best = max(
        eligible,
        key=lambda t: (
            float(t["evalMetric"]),
            -t["trialId"],
        ),
    )

    result["selectedTrialId"] = best["trialId"]

    # EXACT required object/key order.
    lineage = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
    }

    result["datasetDigest"] = digest(lineage)

    return result


# ============================================================
# EVALUATION VALIDATION
# ============================================================

def validate_evaluate(req):

    codes = []

    if req.get("phase") != "evaluate":
        add_code(codes, "INVALID_INPUT")

    run_id = req.get("runId")

    if not isinstance(run_id, str) or not run_id:
        add_code(codes, "INVALID_INPUT")

    trial_id = req.get("selectedTrialId")

    if not safe_int(trial_id) or trial_id < 0:
        add_code(codes, "INVALID_INPUT")

    ds_digest = req.get("datasetDigest")

    if (
        not isinstance(ds_digest, str)
        or HEX64_RE.fullmatch(ds_digest) is None
    ):
        add_code(codes, "INVALID_INPUT")

    metric_floor = req.get("metricFloor")

    if (
        not finite_number(metric_floor)
        or float(metric_floor) < 0
        or float(metric_floor) > 1
    ):
        add_code(codes, "INVALID_INPUT")

    required = req.get("requiredSlices")

    if not isinstance(required, dict):
        add_code(codes, "INVALID_INPUT")
        required = {}

    for name, floor in required.items():

        if not isinstance(name, str) or not name:
            add_code(codes, "INVALID_INPUT")

        if (
            not finite_number(floor)
            or float(floor) < 0
            or float(floor) > 1
        ):
            add_code(codes, "INVALID_INPUT")

    rows = req.get("rows")

    if not isinstance(rows, list):
        add_code(codes, "INVALID_INPUT")
        rows = []

    bp = req.get("bytesProcessed")
    mb = req.get("maxBytes")

    if not safe_int(bp) or bp < 0:
        add_code(codes, "INVALID_INPUT")

    if not safe_int(mb) or mb < 0:
        add_code(codes, "INVALID_INPUT")

    return sorted_codes(codes), rows, required


# ============================================================
# EVALUATION
# ============================================================

def perform_evaluate(req):

    base_codes, rows, required = validate_evaluate(req)

    run_id = req.get("runId")
    selected = req.get("selectedTrialId")
    ds_digest = req.get("datasetDigest")

    bp = req.get("bytesProcessed")
    if not safe_int(bp):
        bp = 0

    result = {
        "runId": run_id if isinstance(run_id, str) else None,
        "selectedTrialId": selected if safe_int(selected) else None,
        "datasetDigest": ds_digest if isinstance(ds_digest, str) else None,
        "testMetric": None,
        "criticalSlicePass": False,
        "decision": "reject",
        "bytesProcessed": bp,
        "reasonCodes": [],
    }

    codes = list(base_codes)

    # --------------------------------------------------------
    # LINEAGE
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
                or frozen["selectedTrialId"] != selected
                or frozen["datasetDigest"] != ds_digest
            ):
                add_code(codes, "INVALID_LINEAGE")

    # --------------------------------------------------------
    # COST
    # --------------------------------------------------------

    max_bytes = req.get("maxBytes")

    if (
        safe_int(bp)
        and safe_int(max_bytes)
        and bp > max_bytes
    ):
        add_code(codes, "BYTE_LIMIT")

    # --------------------------------------------------------
    # FINAL TEST ROWS
    # --------------------------------------------------------

    invalid_row = False

    for row in rows:

        if not isinstance(row, dict):
            invalid_row = True
            break

        label = row.get("label")
        prediction = row.get("prediction")
        slice_name = row.get("slice")

        if (
            not safe_int(label)
            or label not in (0, 1)
            or not safe_int(prediction)
            or prediction not in (0, 1)
            or not isinstance(slice_name, str)
            or not slice_name
        ):
            invalid_row = True
            break

    # Empty OR invalid rows:
    # testMetric remains null.
    if not rows or invalid_row:

        if invalid_row:
            add_code(codes, "INVALID_TEST_ROW")

        result["reasonCodes"] = sorted_codes(codes)
        return result

    # --------------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------------

    correct = sum(
        row["label"] == row["prediction"]
        for row in rows
    )

    test_metric = float(
        f"{correct / len(rows):.12f}"
    )

    result["testMetric"] = test_metric

    if test_metric < float(req["metricFloor"]):
        add_code(codes, "AGGREGATE_FLOOR")

    # --------------------------------------------------------
    # SLICES
    # --------------------------------------------------------

    by_slice = {}

    for row in rows:
        by_slice.setdefault(
            row["slice"],
            [],
        ).append(row)

    all_slices_pass = True

    for name in sorted(required.keys(), key=utf8_key):

        if name not in by_slice:

            add_code(
                codes,
                "MISSING_SLICE:" + name,
            )

            all_slices_pass = False
            continue

        slice_rows = by_slice[name]

        correct_slice = sum(
            row["label"] == row["prediction"]
            for row in slice_rows
        )

        metric = float(
            f"{correct_slice / len(slice_rows):.12f}"
        )

        if metric < float(required[name]):

            add_code(
                codes,
                "SLICE_FLOOR:" + name,
            )

            all_slices_pass = False

    # criticalSlicePass is only the required-slice/validity
    # condition. It does NOT summarize aggregate or byte gates.
    result["criticalSlicePass"] = bool(
        not base_codes
        and stored is not None
        and "INVALID_LINEAGE" not in codes
        and all_slices_pass
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    admit = (
        not base_codes
        and stored is not None
        and "INVALID_LINEAGE" not in codes
        and not invalid_row
        and test_metric >= float(req["metricFloor"])
        and all_slices_pass
        and safe_int(bp)
        and safe_int(max_bytes)
        and bp <= max_bytes
    )

    result["decision"] = "admit" if admit else "reject"
    result["reasonCodes"] = sorted_codes(codes)

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

    # --------------------------------------------------------
    # SELECT
    # --------------------------------------------------------

    if phase == "select":

        run_id = body.get("runId")

        valid_run_id = (
            isinstance(run_id, str)
            and 0 < len(run_id) <= 128
        )

        if valid_run_id:

            identity = request_identity(body)

            with LOCK:
                old = RUNS.get(run_id)

            if old is not None:

                if old["identity"] == identity:
                    return JSONResponse(
                        status_code=200,
                        content=old["response"],
                    )

                return JSONResponse(
                    status_code=409,
                    content={"error": "RUN_ID_CONFLICT"},
                )

        result = perform_select(body)

        if valid_run_id:

            identity = request_identity(body)

            with LOCK:

                old = RUNS.get(run_id)

                if old is not None:

                    if old["identity"] == identity:
                        return JSONResponse(
                            status_code=200,
                            content=old["response"],
                        )

                    return JSONResponse(
                        status_code=409,
                        content={"error": "RUN_ID_CONFLICT"},
                    )

                RUNS[run_id] = {
                    "identity": identity,
                    "response": result,
                }

        return JSONResponse(
            status_code=200,
            content=result,
        )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    result = perform_evaluate(body)

    return JSONResponse(
        status_code=200,
        content=result,
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True}
