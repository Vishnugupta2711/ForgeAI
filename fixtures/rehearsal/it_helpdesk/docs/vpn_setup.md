# Corporate VPN & MFA Guide

## Split-Tunnel VPN Configuration
Connect using the WireGuard client on port 51820 UDP to `vpn.internal.example`.
Split-tunnel routes cover `10.20.0.0/16` and `172.16.4.0/24`; public traffic exits locally.

## Multi-Factor Authentication
Hardware FIDO2 keys or TOTP authenticators are mandatory for all remote logins.
SMS one-time codes were retired on January 15, 2026.
If an authenticator is lost, open an urgent ticket so Helpdesk Tier 2 issues a 24-hour bypass code after video ID verification.
