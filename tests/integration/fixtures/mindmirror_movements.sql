-- MindMirror movements database: exercise catalog
-- Simulates the swae_movements PostgreSQL instance (port 5435 in prod)

CREATE TABLE movements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    description TEXT,
    difficulty TEXT CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
    body_region TEXT,
    target_muscle_group TEXT,
    mechanics TEXT CHECK (mechanics IN ('compound', 'isolation')),
    laterality TEXT CHECK (laterality IN ('bilateral', 'unilateral', 'n/a')),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE muscles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    body_region TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE equipment (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    category TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE movement_muscle_links (
    movement_id UUID NOT NULL REFERENCES movements(id) ON DELETE CASCADE,
    muscle_id UUID NOT NULL REFERENCES muscles(id),
    role TEXT NOT NULL CHECK (role IN ('primary', 'secondary', 'tertiary')),
    PRIMARY KEY (movement_id, muscle_id, role)
);

CREATE TABLE movement_equipment_links (
    movement_id UUID NOT NULL REFERENCES movements(id) ON DELETE CASCADE,
    equipment_id UUID NOT NULL REFERENCES equipment(id),
    is_required BOOLEAN DEFAULT true,
    PRIMARY KEY (movement_id, equipment_id)
);

-- ============================================================
-- Seed data (representative subset of 800+ movements)
-- ============================================================

INSERT INTO muscles (id, name, body_region) VALUES
    ('bb100000-0000-0000-0000-000000000001', 'Quadriceps', 'lower body'),
    ('bb100000-0000-0000-0000-000000000002', 'Glutes', 'lower body'),
    ('bb100000-0000-0000-0000-000000000003', 'Pectorals', 'upper body'),
    ('bb100000-0000-0000-0000-000000000004', 'Latissimus Dorsi', 'upper body'),
    ('bb100000-0000-0000-0000-000000000005', 'Deltoids', 'upper body');

INSERT INTO equipment (id, name, category) VALUES
    ('bc100000-0000-0000-0000-000000000001', 'Barbell', 'free weights'),
    ('bc100000-0000-0000-0000-000000000002', 'Dumbbell', 'free weights'),
    ('bc100000-0000-0000-0000-000000000003', 'Pull-up Bar', 'bodyweight'),
    ('bc100000-0000-0000-0000-000000000004', 'Cable Machine', 'machines');

INSERT INTO movements (id, name, slug, description, difficulty, body_region, target_muscle_group, mechanics, laterality) VALUES
    ('ba100000-0000-0000-0000-000000000001', 'Barbell Back Squat', 'barbell-back-squat', 'A compound lower body exercise targeting quads and glutes', 'intermediate', 'lower body', 'quadriceps', 'compound', 'bilateral'),
    ('ba100000-0000-0000-0000-000000000002', 'Conventional Deadlift', 'conventional-deadlift', 'A full-body compound pulling movement from the floor', 'intermediate', 'full body', 'posterior chain', 'compound', 'bilateral'),
    ('ba100000-0000-0000-0000-000000000003', 'Bench Press', 'bench-press', 'A compound upper body pressing exercise', 'intermediate', 'upper body', 'pectorals', 'compound', 'bilateral'),
    ('ba100000-0000-0000-0000-000000000004', 'Pull-up', 'pull-up', 'A bodyweight upper body pulling exercise', 'intermediate', 'upper body', 'latissimus dorsi', 'compound', 'bilateral'),
    ('ba100000-0000-0000-0000-000000000005', 'Overhead Press', 'overhead-press', 'A compound pressing exercise targeting shoulders', 'intermediate', 'upper body', 'deltoids', 'compound', 'bilateral');

-- Movement-muscle links
INSERT INTO movement_muscle_links (movement_id, muscle_id, role) VALUES
    ('ba100000-0000-0000-0000-000000000001', 'bb100000-0000-0000-0000-000000000001', 'primary'),
    ('ba100000-0000-0000-0000-000000000001', 'bb100000-0000-0000-0000-000000000002', 'secondary'),
    ('ba100000-0000-0000-0000-000000000003', 'bb100000-0000-0000-0000-000000000003', 'primary'),
    ('ba100000-0000-0000-0000-000000000004', 'bb100000-0000-0000-0000-000000000004', 'primary');

-- Movement-equipment links
INSERT INTO movement_equipment_links (movement_id, equipment_id, is_required) VALUES
    ('ba100000-0000-0000-0000-000000000001', 'bc100000-0000-0000-0000-000000000001', true),
    ('ba100000-0000-0000-0000-000000000003', 'bc100000-0000-0000-0000-000000000001', true),
    ('ba100000-0000-0000-0000-000000000004', 'bc100000-0000-0000-0000-000000000003', true);
