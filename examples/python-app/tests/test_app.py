from app import app


def test_index_route():
    client = app.test_client()
    resp = client.get('/')
    assert resp.status_code == 200
    assert b'NexusZero Protocol' in resp.data
