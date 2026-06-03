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


def test_delete_missing_document_returns_404(client: TestClient) -> None:
    resp = client.delete("/documents/inexistente")
    assert resp.status_code == 404


def test_list_documents_empty(client: TestClient) -> None:
    resp = client.get("/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"documents": [], "total_documents": 0, "total_chunks": 0}


def test_list_documents_returns_indexed(client: TestClient) -> None:
    client.post("/documents", json={"doc_id": "a", "text": "alpha"})
    client.post("/documents", json={"doc_id": "b", "text": "beta"})
    body = client.get("/documents").json()
    assert body["total_documents"] == 2
    ids = [doc["doc_id"] for doc in body["documents"]]
    assert ids == ["a", "b"]
    assert body["documents"][0]["modality"] == "text"
    assert body["documents"][0]["content_hash"]


def test_reupsert_same_text_is_unchanged(client: TestClient) -> None:
    payload = {"doc_id": "doc-x", "text": "mesmo conteúdo"}
    first = client.post("/documents", json=payload)
    assert first.json()["status"] == "upserted"
    second = client.post("/documents", json=payload)
    assert second.json()["status"] == "unchanged"


def test_reupsert_changed_text_is_upserted(client: TestClient) -> None:
    client.post("/documents", json={"doc_id": "doc-x", "text": "v1"})
    resp = client.post("/documents", json={"doc_id": "doc-x", "text": "v2"})
    assert resp.json()["status"] == "upserted"


def test_reupload_same_file_is_unchanged(client: TestClient) -> None:
    args = {
        "files": {"file": ("notas.txt", b"conteudo identico", "text/plain")},
        "data": {"doc_id": "notas"},
    }
    assert client.post("/documents/upload", **args).json()["status"] == "upserted"
    assert client.post("/documents/upload", **args).json()["status"] == "unchanged"
