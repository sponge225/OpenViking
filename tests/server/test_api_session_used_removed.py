# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Regression coverage for the removed session usage endpoint."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from openviking.server.routers.sessions import router


def test_session_used_endpoint_is_removed():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    path = "/api/v1/sessions/example/used"

    response = client.post(path, json={"contexts": ["viking://resources/example"]})
    schema = client.get("/openapi.json").json()

    assert response.status_code == 404
    assert "/api/v1/sessions/{session_id}/used" not in schema["paths"]
