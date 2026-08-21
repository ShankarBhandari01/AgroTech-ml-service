"""The access/inference log lines a live request must leave behind.

Logging is the only view into a running deployment, so it gets the same treatment as the response:
if a request no longer says what it did, which artifact answered it, and how long it took, that is a
regression. Reuses the e2e `client` fixture, so this exercises the real middleware and pipeline.
"""

from __future__ import annotations

import logging

from tests.test_e2e_predict import _predict, client  # noqa: F401  (fixture import)


def test_a_request_logs_its_arrival_response_and_inference(client, caplog):  # noqa: F811  (pytest fixture, not a redefinition)
    with caplog.at_level(logging.INFO):
        _predict(client)

    text = caplog.text
    assert "--> POST /predict/coldstart" in text
    assert "<-- POST /predict/coldstart 200" in text
    assert "predict/coldstart request:" in text
    assert "predict/coldstart response:" in text
    # Which vegetation-hazard source answered, named in the pipeline line.
    assert "field=coldstart-" in text and "model=" in text


def test_an_error_response_still_logs_its_status(client, caplog):  # noqa: F811  (pytest fixture, not a redefinition)
    with caplog.at_level(logging.INFO):
        client.post("/predict/coldstart", json={"longitude": "not-a-number"})
    assert "<-- POST /predict/coldstart 422" in caplog.text
