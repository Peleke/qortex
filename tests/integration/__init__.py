"""Integration tests: real PostgreSQL via Docker Compose.

These tests require:
    docker compose -f tests/integration/docker-compose.yml up -d

They are marked @pytest.mark.integration and skip when Docker isn't running.
"""
