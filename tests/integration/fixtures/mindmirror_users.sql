-- MindMirror users database: user accounts and roles
-- Simulates the swae_users PostgreSQL instance (port 5437 in prod)

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    display_name TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- Seed data
-- ============================================================

INSERT INTO users (id, email, display_name) VALUES
    ('01000000-0000-0000-0000-000000000001', 'alice@example.com', 'Alice'),
    ('01000000-0000-0000-0000-000000000002', 'bob@example.com', 'Bob');

INSERT INTO roles (id, name, description) VALUES
    ('d0100000-0000-0000-0000-000000000001', 'user', 'Standard user role'),
    ('d0100000-0000-0000-0000-000000000002', 'coach', 'Coaching privileges');
