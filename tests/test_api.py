from fastapi.testclient import TestClient

from ngram_transformer.app.main import app


def test_health_and_models_endpoints() -> None:
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}
    models = client.get("/models").json()
    assert models[0]["name"] == "ngram"
    assert models[0]["ready"] is True


def test_demo_mounts() -> None:
    client = TestClient(app)

    response = client.get("/demo/", follow_redirects=True)

    assert response.status_code == 200
    assert "From N-gram to Transformer" in response.text


def test_root_redirects_to_demo() -> None:
    client = TestClient(app)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/demo/"


def test_generate_without_saving() -> None:
    client = TestClient(app)
    response = client.post(
        "/generate",
        json={
            "model_name": "ngram",
            "prompt": "The old model",
            "max_new_tokens": 8,
            "temperature": 0.9,
            "seed": 42,
            "save": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "ngram"
    assert payload["generated_text"].startswith("The old model")
