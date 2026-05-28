from fastapi.testclient import TestClient

from tests.conftest import FakeRAGService


def test_upsert_text_document(client: TestClient, fake_service: FakeRAGService) -> None:
    resp = client.post(
        "/documents",
        json={"doc_id": "policy-1", "text": "Política de férias da empresa.", "metadata": {"tag": "rh"}},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["doc_ids"] == ["policy-1"]
    assert body["total_chunks"] == 1
    assert fake_service.docs["policy-1"]["metadata"] == {"tag": "rh"}


def test_upsert_is_idempotent_by_doc_id(client: TestClient, fake_service: FakeRAGService) -> None:
    payload = {"doc_id": "doc-x", "text": "versão 1"}
    client.post("/documents", json=payload)
    resp = client.post("/documents", json={"doc_id": "doc-x", "text": "versão 2"})
    assert resp.status_code == 201
    assert fake_service.count() == 1
    assert fake_service.docs["doc-x"]["text"] == "versão 2"


def test_upload_text_file(client: TestClient, fake_service: FakeRAGService) -> None:
    resp = client.post(
        "/documents/upload",
        files={"file": ("notas.txt", b"conteudo de exemplo", "text/plain")},
        data={"doc_id": "notas"},
    )
    assert resp.status_code == 201
    assert resp.json()["doc_ids"] == ["notas"]
    assert "notas" in fake_service.docs


def test_upload_rejects_unsupported_extension(client: TestClient) -> None:
    resp = client.post(
        "/documents/upload",
        files={"file": ("malware.exe", b"\x00\x01", "application/octet-stream")},
    )
    assert resp.status_code == 415


def test_upload_invalid_metadata_json(client: TestClient) -> None:
    resp = client.post(
        "/documents/upload",
        files={"file": ("a.txt", b"x", "text/plain")},
        data={"metadata": "{not-json"},
    )
    assert resp.status_code == 422


def test_delete_document(client: TestClient, fake_service: FakeRAGService) -> None:
    client.post("/documents", json={"doc_id": "tmp", "text": "apagar"})
    resp = client.delete("/documents/tmp")
    assert resp.status_code == 204
    assert "tmp" not in fake_service.docs
