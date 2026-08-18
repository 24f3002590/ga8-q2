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

# runId -> {
#   "request_json": canonical original request,
#   "response": persisted selection response,
# }
RUNS: dict[str, dict[str, Any]] = {}
LOCK = threading.RLock()

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

SAFE_INT_MAX = 9007199254740991
SAFE_INT_MIN = -9007199254740991


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def digest_json(obj: Any) -> str:
    return hashlib.sha256(
        canonical_json(obj).encode("utf-8")
    ).hexdigest()


def add_code(codes: list[str], code: str) -> None:
    if code not in codes:
        codes.append(code)


def sorted_codes(codes: list[str]) -> list[str]:
    return sorted(set(codes), key=utf8_key)


def is_safe_int(x: Any) -> bool:
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and SAFE_INT_MIN <= x <= SAFE_INT_MAX
    )


def is_finite_number(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def parse_instant(value: Any):
    if not isinstance(value, str):
        return None

    if not TIMESTAMP_RE.fullmatch(value):
        return None

    try:
        s = value
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"

        dt = datetime.fromisoformat(s)

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def is_nonempty_string(x: Any, max_len: int | None = None) -> bool:
    if not isinstance(x, str) or not x:
        return False
    if max_len is not None and len(x) > max_len:
        return False
    return True


def valid_fractional_timestamp(value: Any) -> bool:
    return parse_instant(value) is not None


def validate_selection(req: dict[str, Any]):
    codes: list[str] = []

    # Top-level schema
    if req.get("phase") != "select":
        add_code(codes, "INVALID_INPUT")

    run_id = req.get("runId")
    if not is_nonempty_string(run_id, 128):
        add_code(codes, "INVALID_INPUT")

    forbidden = req.get("forbiddenFeatures")
    if not isinstance(forbidden, list):
        add_code(codes, "INVALID_INPUT")
        forbidden = []

    forbidden_seen = set()
    for f in forbidden:
        if not isinstance(f, str):
            add_code(codes, "INVALID_INPUT")
        elif f in forbidden_seen:
            add_code(codes, "INVALID_INPUT")
        else:
            forbidden_seen.add(f)

    limit = req.get("numTrialsLimit")
    if not is_safe_int(limit) or limit <= 0:
        add_code(codes, "INVALID_INPUT")

    rows = req.get("rows")
    if not isinstance(rows, list) or len(rows) == 0:
        add_code(codes, "INVALID_INPUT")
        rows = []

    trials = req.get("trials")
    if not isinstance(trials, list):
        add_code(codes, "INVALID_INPUT")
        trials = []

    # Validate rows
    row_ids = set()

    for row in rows:
        if not isinstance(row, dict):
            add_code(codes, "INVALID_INPUT")
            continue

        rid = row.get("id")
        if not is_nonempty_string(rid):
            add_code(codes, "INVALID_INPUT")
        elif rid in row_ids:
            add_code(codes, "INVALID_INPUT")
        else:
            row_ids.add(rid)

        entity = row.get("entity")
        if not isinstance(entity, str):
            add_code(codes, "INVALID_INPUT")

        split = row.get("split")
        if split not in ("TRAIN", "EVAL"):
            add_code(codes, "INVALID_INPUT")

        version = row.get("version")
        if not is_safe_int(version) or version < 0:
            add_code(codes, "INVALID_INPUT")

        event_time = parse_instant(row.get("eventTime"))
        prediction_time = parse_instant(row.get("predictionTime"))

        if event_time is None:
            add_code(codes, "INVALID_INPUT")

        if prediction_time is None:
            add_code(codes, "INVALID_INPUT")

        features = row.get("features")
        if not isinstance(features, dict):
            add_code(codes, "INVALID_INPUT")
            continue

        for fname, fdata in features.items():
            if not isinstance(fname, str):
                add_code(codes, "INVALID_INPUT")
                continue

            if not isinstance(fdata, dict):
                add_code(codes, "INVALID_INPUT")
                continue

            if "availableAt" not in fdata or "value" not in fdata:
                add_code(codes, "INVALID_INPUT")
                continue

            available_at = parse_instant(fdata.get("availableAt"))
            if available_at is None:
                add_code(codes, "INVALID_INPUT")

    # Validate trials
    trial_ids = set()

    for trial in trials:
        if not isinstance(trial, dict):
            add_code(codes, "INVALID_INPUT")
            continue

        tid = trial.get("trialId")

        if not is_safe_int(tid) or tid < 0:
            add_code(codes, "INVALID_INPUT")
        elif tid in trial_ids:
            add_code(codes, "INVALID_INPUT")
        else:
            trial_ids.add(tid)

        if trial.get("status") not in ("SUCCEEDED", "FAILED"):
            add_code(codes, "INVALID_INPUT")

        metric = trial.get("evalMetric")
        if not is_finite_number(metric):
            add_code(codes, "INVALID_INPUT")

    if isinstance(limit, int) and limit > 0 and len(trials) > limit:
        add_code(codes, "TRIAL_LIMIT_EXCEEDED")

    return sorted_codes(codes)


def build_selection_response(req: dict[str, Any]) -> dict[str, Any]:
    run_id = req.get("runId") if isinstance(req.get("runId"), str) else None

    codes = validate_selection(req)

    # Base response.
    response = {
        "runId": run_id,
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": codes,
    }

    if codes:
        return response

    rows = req["rows"]
    forbidden = set(req["forbiddenFeatures"])

    # ---------------------------------------------------------
    # Deduplicate by [entity, UTC(eventTime)]
    # Highest version wins.
    # Exact version tie -> UTF-8 smallest ID.
    # ---------------------------------------------------------
    retained: dict[tuple[str, datetime], dict[str, Any]] = {}

    for row in rows:
        event_utc = parse_instant(row["eventTime"])
        key = (row["entity"], event_utc)

        old = retained.get(key)

        if old is None:
            retained[key] = row
        else:
            old_version = old["version"]
            new_version = row["version"]

            if new_version > old_version:
                retained[key] = row
            elif new_version == old_version:
                if utf8_key(row["id"]) < utf8_key(old["id"]):
                    retained[key] = row

    retained_rows = list(retained.values())

    # ---------------------------------------------------------
    # Point-in-time feature eligibility.
    #
    # A feature must:
    #   1. exist in every retained row
    #   2. not be forbidden
    #   3. have availableAt <= predictionTime in every row
    # ---------------------------------------------------------
    common_features: set[str] | None = None

    for row in retained_rows:
        names = set(row["features"].keys())

        if common_features is None:
            common_features = names
        else:
            common_features &= names

    if common_features is None:
        common_features = set()

    eligible_features = []

    for fname in common_features:
        if fname in forbidden:
            continue

        eligible = True

        for row in retained_rows:
            available = parse_instant(
                row["features"][fname]["availableAt"]
            )
            prediction = parse_instant(row["predictionTime"])

            if available is None or prediction is None:
                eligible = False
                break

            if available > prediction:
                eligible = False
                break

        if eligible:
            eligible_features.append(fname)

    eligible_features.sort(key=utf8_key)

    # ---------------------------------------------------------
    # Sort IDs independently by UTF-8 bytes.
    # ---------------------------------------------------------
    train_ids = sorted(
        [r["id"] for r in retained_rows if r["split"] == "TRAIN"],
        key=utf8_key,
    )

    eval_ids = sorted(
        [r["id"] for r in retained_rows if r["split"] == "EVAL"],
        key=utf8_key,
    )

    # ---------------------------------------------------------
    # Successful finite trials only.
    # Highest metric wins.
    # Exact tie -> smallest integer trialId.
    # ---------------------------------------------------------
    successful = []

    for trial in req["trials"]:
        if trial["status"] != "SUCCEEDED":
            continue

        metric = float(trial["evalMetric"])

        if not math.isfinite(metric):
            continue

        successful.append(trial)

    if not successful:
        response["trainRowIds"] = train_ids
        response["evalRowIds"] = eval_ids
        response["featureNames"] = eligible_features
        response["reasonCodes"] = ["NO_SUCCESSFUL_TRIAL"]
        return response

    best = successful[0]

    for trial in successful[1:]:
        m1 = float(trial["evalMetric"])
        m2 = float(best["evalMetric"])

        if m1 > m2:
            best = trial
        elif m1 == m2 and trial["trialId"] < best["trialId"]:
            best = trial

    response["selectedTrialId"] = best["trialId"]
    response["trainRowIds"] = train_ids
    response["evalRowIds"] = eval_ids
    response["featureNames"] = eligible_features

    digest_payload = {
        "trainRowIds": train_ids,
        "evalRowIds": eval_ids,
        "featureNames": eligible_features,
    }

    response["datasetDigest"] = digest_json(digest_payload)

    return response


def validate_evaluation(req: dict[str, Any]):
    codes: list[str] = []

    if req.get("phase") != "evaluate":
        add_code(codes, "INVALID_INPUT")

    run_id = req.get("runId")
    if not is_nonempty_string(run_id):
        add_code(codes, "INVALID_INPUT")

    trial_id = req.get("selectedTrialId")
    if not is_safe_int(trial_id) or trial_id < 0:
        add_code(codes, "INVALID_INPUT")

    digest = req.get("datasetDigest")
    if (
        not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        add_code(codes, "INVALID_INPUT")

    floor = req.get("metricFloor")
    if not is_finite_number(floor) or not (0 <= float(floor) <= 1):
        add_code(codes, "INVALID_INPUT")

    slices = req.get("requiredSlices")
    if not isinstance(slices, dict):
        add_code(codes, "INVALID_INPUT")
        slices = {}

    for name, value in slices.items():
        if not isinstance(name, str) or not name:
            add_code(codes, "INVALID_INPUT")
        if not is_finite_number(value) or not (0 <= float(value) <= 1):
            add_code(codes, "INVALID_INPUT")

    rows = req.get("rows")
    if not isinstance(rows, list):
        add_code(codes, "INVALID_INPUT")
        rows = []

    bytes_processed = req.get("bytesProcessed")
    max_bytes = req.get("maxBytes")

    if not is_safe_int(bytes_processed) or bytes_processed < 0:
        add_code(codes, "INVALID_INPUT")

    if not is_safe_int(max_bytes) or max_bytes < 0:
        add_code(codes, "INVALID_INPUT")

    return sorted_codes(codes), rows, slices


def round12(x: float) -> float:
    # Decimal-style 12-place result, avoiding accidental binary
    # representation leaking into JSON.
    return float(f"{x:.12f}")


def evaluate_request(req: dict[str, Any]) -> dict[str, Any]:
    base_codes, rows, required_slices = validate_evaluation(req)

    run_id = req.get("runId") if isinstance(req.get("runId"), str) else None
    trial_id = (
        req.get("selectedTrialId")
        if is_safe_int(req.get("selectedTrialId"))
        else None
    )
    digest = (
        req.get("datasetDigest")
        if isinstance(req.get("datasetDigest"), str)
        else None
    )

    bytes_processed = (
        req.get("bytesProcessed")
        if is_safe_int(req.get("bytesProcessed"))
        else 0
    )

    response = {
        "runId": run_id,
        "selectedTrialId": trial_id,
        "datasetDigest": digest,
        "testMetric": None,
        "criticalSlicePass": False,
        "decision": "reject",
        "bytesProcessed": bytes_processed,
        "reasonCodes": [],
    }

    codes = list(base_codes)

    # ---------------------------------------------------------
    # Lineage check
    # ---------------------------------------------------------
    stored = None

    if not base_codes and run_id is not None:
        with LOCK:
            stored = RUNS.get(run_id)

        if stored is None:
            add_code(codes, "INVALID_LINEAGE")
        else:
            selection = stored["response"]

            if (
                selection.get("selectedTrialId") is None
                or selection.get("datasetDigest") is None
                or selection.get("selectedTrialId") != trial_id
                or selection.get("datasetDigest") != digest
                or selection.get("reasonCodes")
            ):
                add_code(codes, "INVALID_LINEAGE")

    # ---------------------------------------------------------
    # Byte gate still applies even if test rows are invalid.
    # ---------------------------------------------------------
    max_bytes = req.get("maxBytes")

    if (
        is_safe_int(bytes_processed)
        and is_safe_int(max_bytes)
        and bytes_processed > max_bytes
    ):
        add_code(codes, "BYTE_LIMIT")

    # ---------------------------------------------------------
    # Test-row validation
    # ---------------------------------------------------------
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

    if len(rows) == 0 or invalid_test_row:
        if invalid_test_row:
            add_code(codes, "INVALID_TEST_ROW")

        response["testMetric"] = None
        response["criticalSlicePass"] = False
        response["reasonCodes"] = sorted_codes(codes)
        response["decision"] = "reject"

        return response

    # ---------------------------------------------------------
    # Aggregate accuracy
    # ---------------------------------------------------------
    correct = sum(
        1
        for row in rows
        if row["label"] == row["prediction"]
    )

    aggregate = round12(correct / len(rows))
    response["testMetric"] = aggregate

    metric_floor = float(req["metricFloor"])

    if aggregate < metric_floor:
        add_code(codes, "AGGREGATE_FLOOR")

    # ---------------------------------------------------------
    # Required slices
    #
    # "every present required slice meets its floor" and every
    # required slice must exist.
    # ---------------------------------------------------------
    slice_rows: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        slice_rows.setdefault(row["slice"], []).append(row)

    all_required_slices_pass = True

    for name in sorted(required_slices.keys(), key=utf8_key):
        if name not in slice_rows:
            add_code(codes, f"MISSING_SLICE:{name}")
            all_required_slices_pass = False
            continue

        srows = slice_rows[name]
        scorrect = sum(
            1
            for row in srows
            if row["label"] == row["prediction"]
        )

        smetric = round12(scorrect / len(srows))
        floor = float(required_slices[name])

        if smetric < floor:
            add_code(codes, f"SLICE_FLOOR:{name}")
            all_required_slices_pass = False

    # ---------------------------------------------------------
    # criticalSlicePass:
    #   false for invalid input
    #   false for invalid lineage
    #   false for invalid rows
    #   false for missing required slice
    #   false for failed slice floor
    #
    # It intentionally does NOT include aggregate or byte gates.
    # ---------------------------------------------------------
    critical_pass = (
        not base_codes
        and stored is not None
        and "INVALID_LINEAGE" not in codes
        and all_required_slices_pass
    )

    response["criticalSlicePass"] = bool(critical_pass)

    # ---------------------------------------------------------
    # Final admission gate.
    # ---------------------------------------------------------
    decision = (
        not base_codes
        and stored is not None
        and "INVALID_LINEAGE" not in codes
        and not invalid_test_row
        and aggregate >= metric_floor
        and all_required_slices_pass
        and is_safe_int(bytes_processed)
        and is_safe_int(max_bytes)
        and bytes_processed <= max_bytes
    )

    response["decision"] = "admit" if decision else "reject"
    response["reasonCodes"] = sorted_codes(codes)

    return response


@app.post("/bqml")
async def bqml(request: Request):
    # JSON parsing
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

    # Unknown/missing phase has the explicitly specified response.
    if phase not in ("select", "evaluate"):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if phase == "select":
        run_id = body.get("runId")

        # For a valid runId, check state BEFORE processing so that
        # an exact replay returns the original response byte-for-byte
        # semantically, while changed input produces 409.
        if is_nonempty_string(run_id, 128):
            incoming_canonical = canonical_json(body)

            with LOCK:
                existing = RUNS.get(run_id)

                if existing is not None:
                    if existing["request_json"] == incoming_canonical:
                        return JSONResponse(
                            status_code=200,
                            content=existing["response"],
                        )

                    return JSONResponse(
                        status_code=409,
                        content={"error": "RUN_ID_CONFLICT"},
                    )

        response = build_selection_response(body)

        # Persist the complete selection response for valid runIds.
        if is_nonempty_string(run_id, 128):
            with LOCK:
                # Protect against a race between two requests with the
                # same runId.
                existing = RUNS.get(run_id)

                if existing is not None:
                    incoming_canonical = canonical_json(body)

                    if existing["request_json"] == incoming_canonical:
                        return JSONResponse(
                            status_code=200,
                            content=existing["response"],
                        )

                    return JSONResponse(
                        status_code=409,
                        content={"error": "RUN_ID_CONFLICT"},
                    )

                RUNS[run_id] = {
                    "request_json": canonical_json(body),
                    "response": response,
                }

        return JSONResponse(
            status_code=200,
            content=response,
        )

    # ---------------------------------------------------------
    # Evaluation is deliberately NOT persisted as a replacement
    # for the selection. The successful selection remains frozen.
    # ---------------------------------------------------------
    response = evaluate_request(body)

    return JSONResponse(
        status_code=200,
        content=response,
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True}
