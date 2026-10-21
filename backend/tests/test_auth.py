import pytest


@pytest.fixture
def secured(client, monkeypatch):
    monkeypatch.setenv("POLICY_DESK_PASSWORD", "employee-pass")
    monkeypatch.setenv("POLICY_DESK_ADMIN_PASSWORD", "admin-pass")
    return client


def upload(client, path):
    with path.open("rb") as file:
        return client.post("/ingest", files={"file": (path.name, file)})


def test_open_when_no_passwords_are_set(client):
    assert client.get("/me").json() == {"role": "admin", "auth_required": False}
    assert client.get("/documents").status_code == 200


def test_signed_out_requests_are_rejected(secured, docs_dir):
    assert secured.get("/health").json()["auth_required"] is True
    assert secured.get("/me").json()["role"] is None

    assert secured.get("/documents").status_code == 401
    assert secured.post("/chat", json={"message": "Hi"}).status_code == 401
    assert secured.get("/chat/some-thread").status_code == 401
    assert upload(secured, docs_dir / "company_it_policy.txt").status_code == 401


def test_wrong_password(secured):
    response = secured.post("/login", json={"password": "guess"})

    assert response.status_code == 401
    assert secured.get("/documents").status_code == 401


def test_employee_can_read_but_not_change_documents(secured, docs_dir):
    assert secured.post("/login", json={"password": "employee-pass"}).json() == {"role": "user"}

    assert secured.get("/documents").status_code == 200
    assert upload(secured, docs_dir / "company_it_policy.txt").status_code == 403
    assert secured.delete("/documents/company_it_policy.txt").status_code == 403


def test_admin_can_change_documents_and_sign_out(secured, docs_dir):
    assert secured.post("/login", json={"password": "admin-pass"}).json() == {"role": "admin"}

    assert upload(secured, docs_dir / "company_it_policy.txt").status_code == 200
    assert secured.delete("/documents/company_it_policy.txt").status_code == 200

    secured.post("/logout")
    assert secured.get("/documents").status_code == 401


def test_session_ends_when_password_changes(secured, monkeypatch):
    secured.post("/login", json={"password": "employee-pass"})
    monkeypatch.delenv("POLICY_DESK_PASSWORD")

    # Only the admin password is left, so the old "user" session is invalid.
    assert secured.get("/documents").status_code == 401
