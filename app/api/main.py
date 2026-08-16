from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ImportJobResponse,
    ImportRequest,
    ManualStatus,
    RetrieveRequest,
    RetrieveResponse,
    TraceCreateRequest,
    TraceCreateResponse,
)
from app.generation.mock import MockGenerator
from app.ingestion.service import IngestionService
from app.metrics import HTTP_LATENCY, HTTP_REQUESTS, RAG_ERRORS, observe_timings
from app.settings import PROJECT_ROOT
from app.tracing import InMemoryTraceStore, TraceRecorder, emit_trace, reset_trace, set_trace
from app.wiring import Container, build_container


def create_app(container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or build_container(os.getenv("APP_CONFIG"))
        app.state.import_jobs = {}
        tracing = app.state.container.settings.tracing
        app.state.trace_store = InMemoryTraceStore(
            ttl_seconds=tracing.ttl_seconds,
            max_traces=tracing.max_traces,
            max_events_per_trace=tracing.max_events_per_trace,
        )
        app.state.trace_tasks = set()
        yield

    app = FastAPI(
        title="Li Auto Manual RAG",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        path = str(getattr(route, "path", request.url.path))
        HTTP_REQUESTS.labels(
            method=request.method,
            path=path,
            status=str(response.status_code),
        ).inc()
        HTTP_LATENCY.labels(method=request.method, path=path).observe(time.perf_counter() - started)
        return response

    def current_container(request: Request) -> Container:
        return request.app.state.container

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        selected = current_container(request)
        qdrant_ok = await asyncio.to_thread(selected.vector_store.health)
        return HealthResponse(
            status="ok" if qdrant_ok else "degraded",
            qdrant=qdrant_ok,
            embedding_provider=selected.embedder.component_id,
            embedding_loaded=selected.embedder.is_loaded,
            generator=selected.generator.component_id,
            generator_mock=isinstance(selected.generator, MockGenerator),
            reranker=selected.reranker.component_id,
            reranker_loaded=selected.reranker.is_loaded,
            points_count=await asyncio.to_thread(selected.vector_store.count),
        )

    @app.post("/v1/retrieve", response_model=RetrieveResponse)
    async def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
        selected = current_container(request)
        try:
            outcome = await selected.chat_service.retrieve(
                payload.question,
                payload.top_k,
                payload.filters,
            )
            observe_timings(outcome.timings_ms)
            return RetrieveResponse.model_validate(outcome.model_dump())
        except Exception as exc:  # noqa: BLE001 - convert dependency errors to API status
            RAG_ERRORS.labels(operation="retrieve", exception=type(exc).__name__).inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        selected = current_container(request)
        try:
            answer = await selected.chat_service.answer(payload.question, payload.filters)
            observe_timings(answer.timings_ms)
            return ChatResponse.model_validate(answer.model_dump())
        except Exception as exc:  # noqa: BLE001 - convert dependency errors to API status
            RAG_ERRORS.labels(operation="chat", exception=type(exc).__name__).inc()
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/traces", response_model=TraceCreateResponse, status_code=202)
    async def create_trace(
        payload: TraceCreateRequest,
        request: Request,
    ) -> TraceCreateResponse:
        selected = current_container(request)
        if not selected.settings.tracing.enabled:
            raise HTTPException(status_code=404, detail="RAG tracing is disabled")
        store: InMemoryTraceStore = request.app.state.trace_store
        snapshot = store.create()
        tracing = selected.settings.tracing
        recorder = TraceRecorder(
            snapshot.trace_id,
            store,
            level=payload.trace_level,
            options={
                "vector_preview_dimensions": tracing.vector_preview_dimensions,
                "rerank_pair_sample_limit": tracing.rerank_pair_sample_limit,
                "max_excerpt_chars": tracing.max_excerpt_chars,
            },
        )

        async def run_trace() -> None:
            token = set_trace(recorder)
            try:
                answer = await selected.chat_service.answer(payload.question, payload.filters)
                store.complete(snapshot.trace_id, answer.model_dump(mode="json"))
            except Exception as exc:  # noqa: BLE001 - expose failed pipeline stage in trace
                emit_trace(
                    "request",
                    "request.failed",
                    status="failed",
                    payload={"exception": type(exc).__name__, "message": str(exc)},
                )
                store.fail(snapshot.trace_id, f"{type(exc).__name__}: {exc}")
                RAG_ERRORS.labels(operation="trace", exception=type(exc).__name__).inc()
            finally:
                reset_trace(token)

        task = asyncio.create_task(run_trace())
        request.app.state.trace_tasks.add(task)
        task.add_done_callback(request.app.state.trace_tasks.discard)
        return TraceCreateResponse(
            trace_id=snapshot.trace_id,
            status="running",
            events_url=f"/v1/traces/{snapshot.trace_id}/events",
            snapshot_url=f"/v1/traces/{snapshot.trace_id}",
        )

    @app.get("/v1/traces/{trace_id}")
    async def trace_snapshot(trace_id: str, request: Request):  # type: ignore[no-untyped-def]
        store: InMemoryTraceStore = request.app.state.trace_store
        snapshot = store.get(trace_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return snapshot

    @app.get("/v1/traces/{trace_id}/events")
    async def trace_events(trace_id: str, request: Request) -> StreamingResponse:
        store: InMemoryTraceStore = request.app.state.trace_store
        if store.get(trace_id) is None:
            raise HTTPException(status_code=404, detail="trace not found")
        try:
            last_seq = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            last_seq = 0

        async def event_stream() -> AsyncIterator[str]:
            nonlocal last_seq
            while True:
                events, status = store.events_after(trace_id, last_seq)
                for event in events:
                    last_seq = event.seq
                    yield (
                        f"id: {event.seq}\n"
                        f"event: {event.event}\n"
                        f"data: {event.model_dump_json()}\n\n"
                    )
                if status in {"completed", "failed"} and not events:
                    break
                if await request.is_disconnected():
                    break
                await asyncio.sleep(0.05)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/trace-console", include_in_schema=False)
    async def trace_console() -> FileResponse:
        return FileResponse(PROJECT_ROOT / "app" / "web" / "trace_console.html")

    @app.post("/v1/manuals/import", response_model=ImportJobResponse, status_code=202)
    async def import_manual(payload: ImportRequest, request: Request) -> ImportJobResponse:
        selected = current_container(request)
        running = next(
            (
                job_id
                for job_id, job in request.app.state.import_jobs.items()
                if job["status"] == "running"
            ),
            None,
        )
        if running:
            raise HTTPException(status_code=409, detail=f"import already running: {running}")
        job_id = uuid.uuid4().hex
        request.app.state.import_jobs[job_id] = {"status": "running"}
        previous_incremental = selected.settings.data.incremental
        if payload.force_reembed:
            selected.settings.data.incremental = False

        async def run_import() -> None:
            try:
                report = await IngestionService(selected).run()
                request.app.state.import_jobs[job_id] = {
                    "status": "completed",
                    "report": report,
                }
            except Exception as exc:  # noqa: BLE001 - persist job failure for polling
                RAG_ERRORS.labels(operation="import", exception=type(exc).__name__).inc()
                request.app.state.import_jobs[job_id] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                selected.settings.data.incremental = previous_incremental

        asyncio.create_task(run_import())
        return ImportJobResponse(job_id=job_id, status="running")

    @app.get("/v1/manuals/import/{job_id}", response_model=ImportJobResponse)
    async def import_status(job_id: str, request: Request) -> ImportJobResponse:
        job = request.app.state.import_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="import job not found")
        return ImportJobResponse(job_id=job_id, **job)

    @app.get("/v1/manuals", response_model=list[ManualStatus])
    async def manuals(request: Request) -> list[ManualStatus]:
        selected = current_container(request)
        catalog_path = selected.settings.data.raw_dir / "catalog.json"
        if not catalog_path.exists():
            return []
        catalog = {
            int(item["catalog_id"]): item
            for item in json.loads(catalog_path.read_text(encoding="utf-8"))
        }
        progress_path = selected.settings.experiment.manifest_dir / "batch-latest.json"
        progress: dict[int, dict[str, object]] = {}
        if progress_path.exists():
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            progress = {
                int(item["catalog"]["catalog_id"]): item for item in payload.get("items", [])
            }
        response: list[ManualStatus] = []
        for catalog_id, entry in catalog.items():
            item = progress.get(catalog_id, {})
            report = item.get("report") or {}
            response.append(
                ManualStatus(
                    catalog=entry,
                    status=str(item.get("status", "discovered")),
                    manual_id=report.get("manual_id"),
                    vehicle_model=report.get("vehicle_model"),
                    topics_parsed=int(report.get("topics_parsed", 0)),
                    chunks=int(report.get("chunks", 0)),
                    error=str(item["error"]) if item.get("error") else None,
                )
            )
        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
