from fastapi.testclient import TestClient

from tests.conftest import FakeRAGService


def test_chat_returns_answer_and_sources(client: TestClient, fake_service: FakeRAGService) -> None:
    client.post("/documents", json={"doc_id": "kb-1", "text": "O CEO é a Maria."})

    resp = client.post("/chat", json={"message": "Quem é o CEO?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "Quem é o CEO?" in body["answer"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["text"] == "O CEO é a Maria."


def test_chat_passes_history_and_top_k(client: TestClient, fake_service: FakeRAGService) -> None:
    resp = client.post(
        "/chat",
        json={
            "message": "E sobre férias?",
            "history": [
                {"role": "user", "content": "Olá"},
                {"role": "assistant", "content": "Oi!"},
            ],
            "top_k": 3,
        },
    )
    assert resp.status_code == 200
    call = fake_service.chat_calls[-1]
    assert call["top_k"] == 3
    assert call["history"][0] == {"role": "user", "content": "Olá"}


def test_chat_requires_non_empty_message(client: TestClient) -> None:
    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 422
