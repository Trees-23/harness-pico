"""Network security utilities for web-capable tools."""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

_BLOCKED_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)

_URL_RE = re.compile(r"https?://[^\s\"'`;|<>]+", re.IGNORECASE)
_allowed_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []


def configure_ssrf_whitelist(cidrs: list[str] | tuple[str, ...]) -> None:
    """Allow explicit CIDR ranges to bypass private-network blocking."""
    global _allowed_networks
    networks = []
    for cidr in cidrs:
        try:
            networks.append(ipaddress.ip_network(str(cidr), strict=False))
        except ValueError:
            continue
    _allowed_networks = networks


def _is_blocked_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if _allowed_networks and any(addr in network for network in _allowed_networks):
        return False
    return any(addr in network for network in _BLOCKED_NETWORKS)


def _resolved_addresses(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    return addresses


def validate_url_target(url: str) -> tuple[bool, str]:
    """Validate that a URL is safe for future external web tools to fetch."""
    try:
        parsed = urlparse(str(url))
    except Exception as exc:
        return False, str(exc)

    if parsed.scheme not in ("http", "https"):
        return False, f"Only http/https allowed, got '{parsed.scheme or 'none'}'"
    if not parsed.netloc:
        return False, "Missing domain"
    if parsed.username or parsed.password:
        return False, "Credentials in URLs are not allowed"
    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname"

    try:
        addresses = _resolved_addresses(hostname)
    except socket.gaierror:
        return False, f"Cannot resolve hostname: {hostname}"

    for address in addresses:
        if _is_blocked_address(address):
            return False, f"Blocked: {hostname} resolves to private/internal address {address}"
    return True, ""


def validate_resolved_url(url: str) -> tuple[bool, str]:
    """Validate a final URL after redirect before consuming the response body."""
    try:
        parsed = urlparse(str(url))
    except Exception:
        return True, ""
    hostname = parsed.hostname
    if not hostname:
        return True, ""
    try:
        address = ipaddress.ip_address(hostname)
        if _is_blocked_address(address):
            return False, f"Redirect target is a private address: {address}"
        return True, ""
    except ValueError:
        pass
    try:
        addresses = _resolved_addresses(hostname)
    except socket.gaierror:
        return True, ""
    for address in addresses:
        if _is_blocked_address(address):
            return False, f"Redirect target {hostname} resolves to private address {address}"
    return True, ""


def contains_internal_url(text: str) -> bool:
    """Return True when text contains any URL targeting an internal address."""
    for match in _URL_RE.finditer(str(text)):
        ok, _ = validate_url_target(match.group(0))
        if not ok:
            return True
    return False
