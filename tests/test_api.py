from fastapi.testclient import TestClient

from recoveriq.api.app import app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_recover_endpoint_runs_complete_workflow(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    client = TestClient(app)

    response = client.post(
        "/recover",
        json={},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["recovered_amount"] == "500.00"
    assert data["total_recovery_cost"] == "5.00"
    assert data["net_recovery_value"] == "495.00"
    assert data["decision_count"] == 1
    assert data["terminal"] is True

    assert len(data["decisions"]) == 1

    decision = data["decisions"][0]

    assert decision["action"] == "retry_payment"
    assert decision["recovered"] is True
    assert decision["recovered_amount"] == "500.00"
    assert decision["recovery_cost"] == "5.00"
    assert decision["terminal"] is True


def test_recover_endpoint_runs_with_custom_amount(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "amount": "1000.00",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["recovered"] is True
    assert data["recovered_amount"] == "1000.00"
    assert data["total_recovery_cost"] == "5.00"
    assert data["net_recovery_value"] == "995.00"
    assert data["decision_count"] == 1
    assert data["terminal"] is True


def test_recover_endpoint_runs_with_custom_retry_probability(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "retry_success_probability": "0.5",
            "simulator_seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["recovered"] is False
    assert data["recovered_amount"] == "0"
    assert data["total_recovery_cost"] == "5.00"
    assert data["net_recovery_value"] == "-5.00"
    assert data["terminal"] is True


def test_recover_endpoint_rejects_invalid_amount() -> None:
    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "amount": "0",
        },
    )

    assert response.status_code == 422


def test_recover_endpoint_rejects_invalid_retry_probability() -> None:
    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "retry_success_probability": "1.5",
        },
    )

    assert response.status_code == 422


def test_recover_endpoint_runs_payment_method_scenario(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "failure_category": "payment_method",
            "payment_method_update_success_probability": "0.5",
            "simulator_seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["recovered"] is False
    assert data["recovered_amount"] == "0"
    assert data["total_recovery_cost"] == "11.00"
    assert data["net_recovery_value"] == "-15.00"
    assert data["decision_count"] == 3
    assert data["terminal"] is True


def test_recover_endpoint_runs_authentication_scenario(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "failure_category": "authentication",
            "simulator_seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["terminal"] is True
    assert data["decision_count"] >= 1


def test_recover_endpoint_runs_unknown_failure_scenario(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "failure_category": "unknown",
            "simulator_seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["terminal"] is True
    assert data["decision_count"] >= 1


def test_recover_endpoint_runs_with_custom_maximum_steps(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "maximum_steps": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["recovered"] is True
    assert data["decision_count"] == 1
    assert data["terminal"] is True


def test_recover_endpoint_rejects_invalid_maximum_steps() -> None:
    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "maximum_steps": 0,
        },
    )

    assert response.status_code == 422


def test_recover_endpoint_rejects_negative_retry_cost() -> None:
    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "retry_cost": "-1.00",
        },
    )

    assert response.status_code == 422


def test_recover_endpoint_rejects_invalid_payment_method_probability() -> None:
    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "payment_method_update_success_probability": "-0.1",
        },
    )

    assert response.status_code == 422


def test_recover_endpoint_rejects_invalid_failure_category() -> None:
    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "failure_category": "invalid_category",
        },
    )

    assert response.status_code == 422


def test_recover_endpoint_accepts_zero_retry_probability(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "retry_success_probability": "0",
            "simulator_seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["recovered"] is False
    assert data["terminal"] is True


def test_recover_endpoint_accepts_one_retry_probability(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "retry_success_probability": "1",
            "simulator_seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["recovered"] is True
    assert data["recovered_amount"] == "500.00"
    assert data["terminal"] is True


def test_recover_endpoint_accepts_zero_payment_method_probability(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AI_PROVIDER",
        "deterministic",
    )
    monkeypatch.setenv(
        "AI_MODEL",
        "local",
    )

    client = TestClient(app)

    response = client.post(
        "/recover",
        json={
            "failure_category": "payment_method",
            "payment_method_update_success_probability": "0",
            "simulator_seed": 42,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["terminal"] is True
    assert data["recovered"] is False