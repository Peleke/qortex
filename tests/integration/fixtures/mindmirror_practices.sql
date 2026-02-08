-- MindMirror practices database: workout templates and instances
-- Simulates the swae_practices PostgreSQL instance (port 5436 in prod)
-- Contains cross-DB references to swae_movements (movement_id)

CREATE TABLE practice_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE,
    description TEXT,
    difficulty TEXT CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
    estimated_duration INTEGER,  -- minutes
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE prescription_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    practice_template_id UUID NOT NULL REFERENCES practice_templates(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    sets INTEGER DEFAULT 3,
    reps TEXT DEFAULT '8-12',
    rest_seconds INTEGER DEFAULT 90,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE movement_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prescription_template_id UUID NOT NULL REFERENCES prescription_templates(id) ON DELETE CASCADE,
    movement_id UUID NOT NULL,  -- Cross-DB FK to swae_movements.movements.id (no constraint)
    position INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE practice_instances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    practice_template_id UUID REFERENCES practice_templates(id),
    title TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- Seed data
-- ============================================================

INSERT INTO practice_templates (id, name, slug, description, difficulty, estimated_duration) VALUES
    ('pt100000-0000-0000-0000-000000000001', 'Full Body Strength A', 'full-body-strength-a', 'Squat-focused full body workout', 'intermediate', 60),
    ('pt100000-0000-0000-0000-000000000002', 'Upper Body Push', 'upper-body-push', 'Chest and shoulder pressing day', 'intermediate', 45);

INSERT INTO prescription_templates (id, practice_template_id, position, sets, reps, rest_seconds) VALUES
    ('rx100000-0000-0000-0000-000000000001', 'pt100000-0000-0000-0000-000000000001', 0, 5, '5', 180),
    ('rx100000-0000-0000-0000-000000000002', 'pt100000-0000-0000-0000-000000000001', 1, 3, '8-12', 90),
    ('rx100000-0000-0000-0000-000000000003', 'pt100000-0000-0000-0000-000000000002', 0, 4, '6-8', 120);

-- movement_id values reference movements in swae_movements DB (cross-DB)
INSERT INTO movement_templates (id, prescription_template_id, movement_id, position, notes) VALUES
    ('mt100000-0000-0000-0000-000000000001', 'rx100000-0000-0000-0000-000000000001', 'mv100000-0000-0000-0000-000000000001', 0, 'Barbell Back Squat - warm up with bar first'),
    ('mt100000-0000-0000-0000-000000000002', 'rx100000-0000-0000-0000-000000000002', 'mv100000-0000-0000-0000-000000000002', 0, 'Conventional Deadlift'),
    ('mt100000-0000-0000-0000-000000000003', 'rx100000-0000-0000-0000-000000000003', 'mv100000-0000-0000-0000-000000000003', 0, 'Bench Press');

-- Practice instances (user-scoped)
INSERT INTO practice_instances (user_id, practice_template_id, title, started_at, completed_at, notes) VALUES
    ('u1000000-0000-0000-0000-000000000001', 'pt100000-0000-0000-0000-000000000001', 'Monday Strength', '2026-02-03 07:00:00+00', '2026-02-03 08:05:00+00', 'Felt strong, PR on squats'),
    ('u1000000-0000-0000-0000-000000000001', 'pt100000-0000-0000-0000-000000000002', 'Wednesday Push', '2026-02-05 07:00:00+00', NULL, 'Had to cut short');
