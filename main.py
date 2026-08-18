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


def utf8(s: str) -> bytes:
    return s.encode("utf-8")


def safe_int(x: Any) -> bool:
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and SAFE_MIN <= x <= SAFE_MAX
    )


def finite(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def parse_ts(x: Any):
    if not isinstance(x, str):
        return None

    if not TS_RE.fullmatch(x):
        return None

    try:
        s = x[:-1] + "+00:00" if x.endswith("Z") else x
        dt = datetime.fromisoformat(s)

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def compact_json(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def canonical_request(obj: Any) -> str:
    """
    Semantic request identity for replay/conflict detection.
    Key order in the incoming JSON must not make an otherwise
    identical request a conflict.
    """
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        compact_json(obj).encode("utf-8")
    ).hexdigest()


def add(codes: list[str], code: str):
    if code not in codes:
        codes.append(code)


def sort_codes(codes: list[str]) -> list[str]:
    return sorted(set(codes), key=utf8)


def valid_nonempty_string(x: Any, max_len: int | None = None) -> bool:
    if not isinstance(x, str) or not x:
        return False
    if max_len is not None and len(x) > max_len:
        return False
    return True


# ============================================================
# SELECTION VALIDATION
# ============================================================

def validate_selection(req: dict[str, Any]) -> list[str]:
    codes: list[str] = []

    if req.get("phase") != "select":
        add(codes, "INVALID_INPUT")

    run_id = req.get("runId")
    if not valid_nonempty_string(run_id, 128):
        add(codes, "INVALID_INPUT")

    forbidden = req.get("forbiddenFeatures")
    if not isinstance(forbidden, list):
        add(codes, "INVALID_INPUT")
        forbidden = []

    # Duplicate forbidden names are harmless and are not a
    # contract violation.
    for name in forbidden:
        if not isinstance(name, str):
            add(codes, "INVALID_INPUT")

    limit = req.get("numTrialsLimit")
    if not safe_int(limit) or limit <= 0:
        add(codes, "INVALID_INPUT")

    rows = req.get("rows")
    if not isinstance(rows, list) or len(rows) == 0:
        add(codes, "INVALID_INPUT")
        rows = []

    trials = req.get("trials")
    if not isinstance(trials, list):
        add(codes, "INVALID_INPUT")
        trials = []

    # --------------------------------------------------------
    # Row validation
    # --------------------------------------------------------

    row_ids: set[str] = set()

    for row in rows:
        if not isinstance(row, dict):
            add(codes, "INVALID_INPUT")
            continue

        rid = row.get("id")
        entity = row.get("entity")
        event_time_raw = row.get("eventTime")
        prediction_time_raw = row.get("predictionTime")
        version = row.get("version")
        split = row.get("split")
        features = row.get("features")

        if not valid_nonempty_string(rid):
            add(codes, "INVALID_INPUT")
        elif rid in row_ids:
            add(codes, "INVALID_INPUT")
        else:
            row_ids.add(rid)

        if not valid_nonempty_string(entity):
            add(codes, "INVALID_INPUT")

        if not safe_int(version) or version < 0:
            add(codes, "INVALID_INPUT")

        if split not in ("TRAIN", "EVAL"):
            add(codes, "INVALID_INPUT")

        event_time = parse_ts(event_time_raw)
        prediction_time = parse_ts(prediction_time_raw)

        if event_time is None or prediction_time is None:
            add(codes, "INVALID_INPUT")
        else:
            # A prediction cannot precede the event it predicts.
            if event_time > prediction_time:
                add(codes, "INVALID_INPUT")

        if not isinstance(features, dict):
            add(codes, "INVALID_INPUT")
            continue

        feature_names: set[str] = set()

        for fname, fobj in features.items():
            if not valid_nonempty_string(fname):
                add(codes, "INVALID_INPUT")
                continue

            if fname in feature_names:
                add(codes, "INVALID_INPUT")

            feature_names.add(fname)

            if not isinstance(fobj, dict):
                add(codes, "INVALID_INPUT")
                continue

            if "value" not in fobj or "availableAt" not in fobj:
                add(codes, "INVALID_INPUT")
                continue

            available = parse_ts(fobj.get("availableAt"))

            if available is None:
                add(codes, "INVALID_INPUT")
                continue

            if prediction_time is not None and available > prediction_time:
                add(codes, "INVALID_INPUT")

    # --------------------------------------------------------
    # Trial validation
    # --------------------------------------------------------

    trial_ids: set[int] = set()

    for trial in trials:
        if not isinstance(trial, dict):
            add(codes, "INVALID_INPUT")
            continue

        tid = trial.get("trialId")
        status = trial.get("status")
        metric = trial.get("evalMetric")

        if not safe_int(tid) or tid < 0:
            add(codes, "INVALID_INPUT")
        elif tid in trial_ids:
            add(codes, "INVALID_INPUT")
        else:
            trial_ids.add(tid)

        if status not in ("SUCCEEDED", "FAILED"):
            add(codes, "INVALID_INPUT")

        if not finite(metric):
            add(codes, "INVALID_INPUT")

    if (
        safe_int(limit)
        and limit > 0
        and len(trials) > limit
    ):
        add(codes, "TRIAL_LIMIT_EXCEEDED")

    return sort_codes(codes)


# ============================================================
# SELECTION
# ============================================================

def select(req: dict[str, Any]) -> dict[str, Any]:
    codes = validate_selection(req)

    run_id = req.get("runId")
    if not isinstance(run_id, str):
        run_id = None

    response = {
        "runId": run_id,
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": codes,
    }

    # Any malformed selection stops before deriving dataset
    # lineage. In particular, digest MUST remain null.
    if codes:
        return response

    rows = req["rows"]
    forbidden = set(req["forbiddenFeatures"])

    # --------------------------------------------------------
    # Deduplicate by:
    #   [entity, UTC(eventTime)]
    #
    # Highest version wins.
    # Exact version tie -> UTF-8-smallest ID.
    # --------------------------------------------------------

    retained: dict[tuple[str, datetime], dict[str, Any]] = {}

    for row in rows:
        event_utc = parse_ts(row["eventTime"])
        key = (row["entity"], event_utc)

        previous = retained.get(key)

        if previous is None:
            retained[key] = row
            continue

        if row["version"] > previous["version"]:
            retained[key] = row
        elif row["version"] == previous["version"]:
            if utf8(row["id"]) < utf8(previous["id"]):
                retained[key] = row

    retained_rows = list(retained.values())

    # --------------------------------------------------------
    # Shared features.
    #
    # A feature survives iff it exists in EVERY retained row,
    # isn't forbidden, and is point-in-time available in EVERY
    # retained row.
    # --------------------------------------------------------

    shared: set[str] | None = None

    for row in retained_rows:
        names = set(row["features"].keys())

        if shared is None:
            shared = names
        else:
            shared &= names

    if shared is None:
        shared = set()

    feature_names: list[str] = []

    for fname in shared:
        if fname in forbidden:
            continue

        ok = True

        for row in retained_rows:
            available = parse_ts(
                row["features"][fname]["availableAt"]
            )
            prediction = parse_ts(row["predictionTime"])

            if available is None or prediction is None:
                ok = False
                break

            # Critical point-in-time gate:
            # the feature must already exist at prediction time.
            if available > prediction:
                ok = False
                break

        if ok:
            feature_names.append(fname)

    feature_names.sort(key=utf8)

    # --------------------------------------------------------
    # Split-specific IDs.
    # Final-test rows cannot enter selection because the only
    # accepted selection splits are TRAIN and EVAL.
    # --------------------------------------------------------

    train_ids = sorted(
        [
            row["id"]
            for row in retained_rows
            if row["split"] == "TRAIN"
        ],
        key=utf8,
    )

    eval_ids = sorted(
        [
            row["id"]
            for row in retained_rows
            if row["split"] == "EVAL"
        ],
        key=utf8,
    )

    # --------------------------------------------------------
    # Trial tuning/selection.
    #
    # FAILED is excluded.
    # SUCCEEDED must have finite evalMetric.
    # Max metric wins.
    # Exact tie -> smallest trialId.
    # --------------------------------------------------------

    eligible_trials = [
        trial
        for trial in req["trials"]
        if trial["status"] == "SUCCEEDED"
        and finite(trial["evalMetric"])
    ]

    if not eligible_trials:
        response["trainRowIds"] = train_ids
        response["evalRowIds"] = eval_ids
        response["featureNames"] = feature_names
        response["reasonCodes"] = ["NO_SUCCESSFUL_TRIAL"]
        return response

    best = min(
        eligible_trials,
        key=lambda t: (
            -float(t["evalMetric"]),
            t["trialId"],
        ),
    )

    response["selectedTrialId"] = best["trialId"]
    response["trainRowIds"] = train_ids
    response["evalRowIds"] = eval_ids
    response["featureNames"] = feature_names

    # EXACT required digest shape and order.
    digest_payload = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": feature_names,
    }

    response["datasetDigest"] = sha256_json(digest_payload)

    return response


# ============================================================
# EVALUATION VALIDATION
# ============================================================

def validate_evaluation(req: dict[str, Any]):
    codes: list[str] = []

    if req.get("phase") != "evaluate":
        add(codes, "INVALID_INPUT")

    run_id = req.get("runId")
    if not valid_nonempty_string(run_id):
        add(codes, "INVALID_INPUT")

    selected = req.get("selectedTrialId")
    if not safe_int(selected) or selected < 0:
        add(codes, "INVALID_INPUT")

    digest = req.get("datasetDigest")
    if (
        not isinstance(digest, str)
        or HEX64_RE.fullmatch(digest) is None
    ):
        add(codes, "INVALID_INPUT")

    metric_floor = req.get("metricFloor")
    if (
        not finite(metric_floor)
        or not 0 <= float(metric_floor) <= 1
    ):
        add(codes, "INVALID_INPUT")

    required = req.get("requiredSlices")
    if not isinstance(required, dict):
        add(codes, "INVALID_INPUT")
        required = {}

    for name, floor in required.items():
        if not valid_nonempty_string(name):
            add(codes, "INVALID_INPUT")

        if (
            not finite(floor)
            or not 0 <= float(floor) <= 1
        ):
            add(codes, "INVALID_INPUT")

    rows = req.get("rows")
    if not isinstance(rows, list):
        add(codes, "INVALID_INPUT")
        rows = []

    bytes_processed = req.get("bytesProcessed")
    max_bytes = req.get("maxBytes")

    if (
        not safe_int(bytes_processed)
        or bytes_processed < 0
    ):
        add(codes, "INVALID_INPUT")

    if (
        not safe_int(max_bytes)
        or max_bytes < 0
    ):
        add(codes, "INVALID_INPUT")

    return sort_codes(codes), rows, required


def r12(x: float) -> float:
    return float(f"{x:.12f}")


# ============================================================
# EVALUATION
# ============================================================

def evaluate(req: dict[str, Any]) -> dict[str, Any]:
    base_codes, rows, required = validate_evaluation(req)

    run_id = req.get("runId")
    selected = req.get("selectedTrialId")
    digest = req.get("datasetDigest")

    bytes_processed = (
        req.get("bytesProcessed")
        if safe_int(req.get("bytesProcessed"))
        else 0
    )

    response = {
        "runId": run_id if isinstance(run_id, str) else None,
        "selectedTrialId": selected if safe_int(selected) else None,
        "datasetDigest": digest if isinstance(digest, str) else None,
        "testMetric": None,
        "criticalSlicePass": False,
        "decision": "reject",
        "bytesProcessed": bytes_processed,
        "reasonCodes": [],
    }

    codes = list(base_codes)

    # --------------------------------------------------------
    # Frozen lineage.
    # --------------------------------------------------------

    stored = None

    if not base_codes:
        with LOCK:
            stored = RUNS.get(run_id)

        if stored is None:
            add(codes, "INVALID_LINEAGE")
        else:
            frozen = stored["response"]

            if (
                frozen["selectedTrialId"] is None
                or frozen["datasetDigest"] is None
                or frozen["reasonCodes"]
                or frozen["selectedTrialId"] != selected
                or frozen["datasetDigest"] != digest
            ):
                add(codes, "INVALID_LINEAGE")

    # --------------------------------------------------------
    # Cost gate.
    # --------------------------------------------------------

    max_bytes = req.get("maxBytes")

    if (
        safe_int(bytes_processed)
        and safe_int(max_bytes)
        and bytes_processed > max_bytes
    ):
        add(codes, "BYTE_LIMIT")

    # --------------------------------------------------------
    # Test rows.
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
            not safe_int(label)
            or label not in (0, 1)
            or not safe_int(prediction)
            or prediction not in (0, 1)
            or not valid_nonempty_string(slice_name)
        ):
            invalid_test_row = True
            break

    # Empty test set or invalid test row means:
    # testMetric = null
    # skip aggregate and slice checks.
    if not rows or invalid_test_row:
        if invalid_test_row:
            add(codes, "INVALID_TEST_ROW")

        response["testMetric"] = None
        response["criticalSlicePass"] = False
        response["decision"] = "reject"
        response["reasonCodes"] = sort_codes(codes)
        return response

    # --------------------------------------------------------
    # Aggregate accuracy.
    # --------------------------------------------------------

    correct = sum(
        row["label"] == row["prediction"]
        for row in rows
    )

    aggregate = r12(correct / len(rows))
    response["testMetric"] = aggregate

    metric_floor = float(req["metricFloor"])

    if aggregate < metric_floor:
        add(codes, "AGGREGATE_FLOOR")

    # --------------------------------------------------------
    # Required slices.
    # --------------------------------------------------------

    by_slice: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        by_slice.setdefault(row["slice"], []).append(row)

    slice_pass = True

    for name in sorted(required.keys(), key=utf8):
        if name not in by_slice:
            add(codes, f"MISSING_SLICE:{name}")
            slice_pass = False
            continue

        slice_rows = by_slice[name]

        slice_correct = sum(
            row["label"] == row["prediction"]
            for row in slice_rows
        )

        slice_metric = r12(
            slice_correct / len(slice_rows)
        )

        floor = float(required[name])

        if slice_metric < floor:
            add(codes, f"SLICE_FLOOR:{name}")
            slice_pass = False

    # criticalSlicePass intentionally does NOT incorporate:
    # - aggregate floor
    # - byte limit
    #
    # It only represents validity/lineage/test-row/required-slice
    # conditions.
    response["criticalSlicePass"] = bool(
        not base_codes
        and stored is not None
        and "INVALID_LINEAGE" not in codes
        and slice_pass
    )

    # --------------------------------------------------------
    # Final decision.
    # --------------------------------------------------------

    admit = (
        not base_codes
        and stored is not None
        and "INVALID_LINEAGE" not in codes
        and not invalid_test_row
        and aggregate >= metric_floor
        and slice_pass
        and safe_int(bytes_processed)
        and safe_int(max_bytes)
        and bytes_processed <= max_bytes
    )

    response["decision"] = "admit" if admit else "reject"
    response["reasonCodes"] = sort_codes(codes)

    return response


# ============================================================
# HTTP ENDPOINT
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

    # Explicit contract for missing/unknown phase.
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

        if valid_nonempty_string(run_id, 128):
            identity = canonical_request(body)

            with LOCK:
                existing = RUNS.get(run_id)

            if existing is not None:
                if existing["requestIdentity"] == identity:
                    return JSONResponse(
                        status_code=200,
                        content=existing["response"],
                    )

                return JSONResponse(
                    status_code=409,
                    content={"error": "RUN_ID_CONFLICT"},
                )

        response = select(body)

        # Only a valid runId can be persisted.
        if valid_nonempty_string(run_id, 128):
            identity = canonical_request(body)

            with LOCK:
                existing = RUNS.get(run_id)

                if existing is not None:
                    if existing["requestIdentity"] == identity:
                        return JSONResponse(
                            status_code=200,
                            content=existing["response"],
                        )

                    return JSONResponse(
                        status_code=409,
                        content={"error": "RUN_ID_CONFLICT"},
                    )

                RUNS[run_id] = {
                    "requestIdentity": identity,
                    "response": response,
                }

        return JSONResponse(
            status_code=200,
            content=response,
        )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    return JSONResponse(
        status_code=200,
        content=evaluate(body),
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True}
