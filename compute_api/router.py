"""Routes of the authenticated machine API (``/api/v1``).

Every route runs the same fixed sequence before doing any work:

1. refuse a query string (credentials must never travel in URLs);
2. refuse clients that are not on the loopback interface (the API is meant to
   be reached through an SSH tunnel that terminates on the service host);
3. authenticate the bearer token and check the route's scope;
4. only then read and validate the request body.

``/identity`` and ``/plans`` never contact POWER. ``PUT /attempts/{id}``
prepares, or prepares and submits, one calculation and ``GET /attempts/{id}``
reads its state, both through Compute's existing remote machinery and
server-owned connection profiles (see ``compute_api.execution``). The planner
is injected by the web application so that the machine API uses exactly the
browser's scientific resolution path.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.calculations.capabilities import SCHEMA_VERSION as CAPABILITY_SCHEMA_VERSION
from backend.calculations.registry import CalculationValidationError
from backend.parser import StructureValidationError
from backend.provenance import source_metadata

from compute_api import API_VERSION
from compute_api.auth import (
    SCOPE_PLAN,
    SCOPE_PREPARE,
    SCOPE_READ,
    SCOPE_SUBMIT,
    ApiAuthError,
    Principal,
    authenticate,
)
from compute_api.execution import ExecutionError, execute_attempt, lookup_attempt
from compute_api.ledger import canonical_attempt_id
from compute_api.plan_digest import PLAN_DIGEST_VERSION
from compute_api.projection import (
    PLAN_REQUEST_SCHEMA_VERSION,
    PLAN_RESPONSE_SCHEMA_VERSION,
    plan_response,
)
from compute_api.schemas import (
    MAX_REQUEST_BYTES,
    PlanRequestError,
    parse_execution_request,
    parse_plan_request,
)


LOGGER = logging.getLogger(__name__)
ALLOW_NON_LOOPBACK_ENV = "BMD_API_ALLOW_NON_LOOPBACK"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
MAX_MESSAGE_CHARS = 500
SERVICE_NAME = "bmd-compute"

Planner = Callable[..., Any]


class _ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str, **details: Any):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def _bounded(text: Any) -> str | None:
    if text is None:
        return None
    value = " ".join(str(text).split())
    return value if len(value) <= MAX_MESSAGE_CHARS else value[: MAX_MESSAGE_CHARS - 3] + "..."


def _json(status_code: int, payload: dict, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        payload,
        status_code=status_code,
        headers={"Cache-Control": "no-store", **(headers or {})},
    )


def _error_response(exc: _ApiError | ApiAuthError) -> JSONResponse:
    error = {"code": exc.code, "message": exc.message}
    error.update({key: value for key, value in getattr(exc, "details", {}).items() if value is not None})
    headers = {"WWW-Authenticate": 'Bearer realm="bmd-compute"'} if exc.status_code == 401 else None
    return _json(exc.status_code, {"api_version": API_VERSION, "error": error}, headers)


def _non_loopback_allowed() -> bool:
    return str(os.environ.get(ALLOW_NON_LOOPBACK_ENV, "")).strip().lower() in {"1", "true", "yes"}


def _authorize(request: Request, scope: str | None) -> Principal:
    if request.url.query:
        raise _ApiError(400, "query_not_allowed", "Machine API requests must not carry a query string.")
    client_host = request.client.host if request.client is not None else None
    if client_host not in LOOPBACK_HOSTS and not _non_loopback_allowed():
        raise _ApiError(
            403,
            "loopback_required",
            "The machine API is only served to loopback clients (use the SSH tunnel).",
        )
    principal = authenticate(request.headers.getlist("authorization"))
    if scope is not None:
        principal.require(scope)
    return principal


async def _read_json_body(request: Request) -> Any:
    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise _ApiError(415, "unsupported_media_type", "Send the request body as application/json.")
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_REQUEST_BYTES:
                raise _ApiError(413, "request_too_large", "The request body is too large.")
        except ValueError:
            raise _ApiError(400, "invalid_request", "Invalid Content-Length header.") from None
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REQUEST_BYTES:
            raise _ApiError(413, "request_too_large", "The request body is too large.")
    try:
        return json.loads(bytes(body).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _ApiError(400, "invalid_json", "The request body is not valid UTF-8 JSON.") from None


def build_machine_api_router(*, planner: Planner) -> APIRouter:
    # The machine API is deliberately left out of /openapi.json: it is not
    # advertised on the browser service, and the browser surface stays unchanged.
    router = APIRouter(prefix="/api/v1", include_in_schema=False)

    @router.get("/identity")
    async def identity(request: Request):
        try:
            principal = _authorize(request, scope=None)
        except (_ApiError, ApiAuthError) as exc:
            return _error_response(exc)
        source = source_metadata()
        available = source.get("status") == "available"
        return _json(
            200,
            {
                "api_version": API_VERSION,
                "service": SERVICE_NAME,
                "compute_source": {
                    "git_commit": source.get("git_commit") if available else None,
                    "dirty": source.get("dirty") if available else None,
                },
                "principal": principal.principal,
                "scopes": sorted(principal.scopes),
                "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
                "plan_request_schema_version": PLAN_REQUEST_SCHEMA_VERSION,
                "plan_response_schema_version": PLAN_RESPONSE_SCHEMA_VERSION,
                "plan_digest_version": PLAN_DIGEST_VERSION,
            },
        )

    @router.post("/plans")
    async def plans(request: Request):
        try:
            _authorize(request, scope=SCOPE_PLAN)
            document = await _read_json_body(request)
            try:
                plan_request = parse_plan_request(document)
            except PlanRequestError as exc:
                raise _ApiError(422, "invalid_request", "The plan request is invalid.", fields=exc.errors) from None
        except (_ApiError, ApiAuthError) as exc:
            return _error_response(exc)

        def build_plan() -> dict:
            plan = planner(
                structure_text=plan_request.structure_text,
                fmt=plan_request.structure_format,
                desired_output=plan_request.desired_output,
                custom_workflow=plan_request.custom_workflow,
                resources=plan_request.resources,
            )
            return plan_response(plan, plan_request)

        try:
            # Scientific resolution is CPU-bound; keep it off the event loop.
            response = await run_in_threadpool(build_plan)
        except Exception as exc:  # noqa: BLE001 - classified below, never echoed raw
            return _error_response(_scientific_error(exc, "plan generation", "plan_failed", "BMD Compute could not generate this plan."))
        return _json(200, response)

    @router.put("/attempts/{attempt_id}")
    async def put_attempt(attempt_id: str, request: Request):
        try:
            principal = _authorize(request, scope=None)
            if not principal.scopes & {SCOPE_PREPARE, SCOPE_SUBMIT}:
                raise ApiAuthError(403, "insufficient_scope", "This token cannot prepare or submit calculations.")
            canonical = _canonical_attempt_id(attempt_id)
            document = await _read_json_body(request)
            try:
                execution_request = parse_execution_request(document)
            except PlanRequestError as exc:
                raise _ApiError(422, "invalid_request", "The attempt request is invalid.", fields=exc.errors) from None
            principal.require(SCOPE_SUBMIT if execution_request.submit else SCOPE_PREPARE)
        except (_ApiError, ApiAuthError) as exc:
            return _error_response(exc)

        try:
            response = await run_in_threadpool(
                execute_attempt,
                planner=planner,
                principal=principal.principal,
                attempt_id=canonical,
                request=execution_request,
            )
        except ExecutionError as exc:
            return _error_response(_ApiError(exc.status_code, exc.code, exc.message, **exc.details))
        except Exception as exc:  # noqa: BLE001 - classified below, never echoed raw
            return _error_response(_scientific_error(exc, "attempt execution", "attempt_failed", "BMD Compute could not process this attempt."))
        return _json(200, response)

    @router.get("/attempts/{attempt_id}")
    async def get_attempt(attempt_id: str, request: Request):
        try:
            principal = _authorize(request, scope=SCOPE_READ)
            canonical = _canonical_attempt_id(attempt_id)
        except (_ApiError, ApiAuthError) as exc:
            return _error_response(exc)
        try:
            response = await run_in_threadpool(
                lookup_attempt,
                principal=principal.principal,
                attempt_id=canonical,
            )
        except ExecutionError as exc:
            return _error_response(_ApiError(exc.status_code, exc.code, exc.message, **exc.details))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Machine API attempt lookup failed (%s).", type(exc).__name__, exc_info=True)
            return _error_response(_ApiError(500, "lookup_failed", "BMD Compute could not read this attempt."))
        return _json(200, response)

    return router


def _canonical_attempt_id(value: str) -> str:
    canonical = canonical_attempt_id(value)
    if canonical is None:
        raise _ApiError(422, "invalid_attempt_id", "The attempt ID must be a canonical lowercase UUID.")
    return canonical


def _scientific_error(exc: Exception, context: str, fallback_code: str, fallback_message: str) -> _ApiError:
    if isinstance(exc, StructureValidationError):
        return _ApiError(
            422,
            "structure_invalid",
            _bounded(exc.message) or "The structure could not be read.",
            suggestion=_bounded(exc.suggestion),
        )
    if isinstance(exc, CalculationValidationError):
        diagnostic = exc.diagnostic or {}
        diagnostic_code = diagnostic.get("code")
        return _ApiError(
            422,
            "calculation_invalid",
            _bounded(exc.message) or "The calculation request is not valid.",
            suggestion=_bounded(exc.suggestion),
            diagnostic_code=diagnostic_code if isinstance(diagnostic_code, str) else None,
        )
    LOGGER.error("Machine API %s failed (%s).", context, type(exc).__name__, exc_info=True)
    return _ApiError(500, fallback_code, fallback_message)
