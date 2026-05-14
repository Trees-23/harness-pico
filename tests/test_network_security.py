import socket

from pico.security.network import (
    configure_ssrf_whitelist,
    contains_internal_url,
    validate_resolved_url,
    validate_url_target,
)


def fake_getaddrinfo(mapping):
    def _fake_getaddrinfo(hostname, port, family=0, type=0):
        del port, family, type
        value = mapping[hostname]
        addresses = value if isinstance(value, list) else [value]
        return [(socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, 0, "", (address, 0)) for address in addresses]

    return _fake_getaddrinfo


def test_validate_url_target_allows_public_http_https(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo({"example.com": "93.184.216.34"}))

    assert validate_url_target("https://example.com/docs") == (True, "")
    assert validate_url_target("http://example.com") == (True, "")


def test_validate_url_target_rejects_unsafe_shapes(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo({"example.com": "93.184.216.34"}))

    assert "Only http/https" in validate_url_target("file:///etc/passwd")[1]
    assert "Missing domain" in validate_url_target("https:///missing-host")[1]
    assert "Credentials" in validate_url_target("https://user:pass@example.com")[1]


def test_validate_url_target_blocks_private_and_metadata_addresses(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake_getaddrinfo(
            {
                "localhost": "127.0.0.1",
                "metadata.local": "169.254.169.254",
                "private.local": "10.0.0.5",
                "mixed.local": ["93.184.216.34", "192.168.1.5"],
            }
        ),
    )

    for url in (
        "http://localhost",
        "http://metadata.local/latest/meta-data",
        "http://private.local",
        "https://mixed.local",
    ):
        ok, error = validate_url_target(url)
        assert ok is False
        assert "private/internal" in error


def test_validate_resolved_url_blocks_redirect_targets(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo({"internal.example": "172.16.0.10"}))

    ok_ip, error_ip = validate_resolved_url("http://127.0.0.1/admin")
    ok_host, error_host = validate_resolved_url("https://internal.example/redirected")

    assert ok_ip is False
    assert "private address" in error_ip
    assert ok_host is False
    assert "private address" in error_host


def test_contains_internal_url_and_whitelist(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake_getaddrinfo(
            {
                "safe.example": "93.184.216.34",
                "tailnet.example": "100.64.0.42",
                "loopback.example": "127.0.0.1",
            }
        ),
    )
    configure_ssrf_whitelist([])

    assert contains_internal_url("curl https://safe.example/path") is False
    assert contains_internal_url("curl http://loopback.example:8080") is True
    assert contains_internal_url("curl http://tailnet.example") is True

    configure_ssrf_whitelist(["100.64.0.0/10"])
    assert validate_url_target("http://tailnet.example") == (True, "")

    configure_ssrf_whitelist([])
