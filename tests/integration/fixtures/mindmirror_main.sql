-- MindMirror main database: journals, habits, meals
-- Simulates the mindmirror PostgreSQL instance (port 5432 in prod)

-- ============================================================
-- Habits domain
-- ============================================================

CREATE TABLE habit_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    description TEXT,
    category TEXT DEFAULT 'general',
    frequency TEXT DEFAULT 'daily',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE habit_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    habit_template_id UUID NOT NULL REFERENCES habit_templates(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    response TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- Journal domain
-- ============================================================

CREATE TABLE journal_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    entry_type TEXT NOT NULL DEFAULT 'freeform',
    payload JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    modified_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- Meals domain
-- ============================================================

CREATE TABLE food_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE,
    calories NUMERIC NOT NULL CHECK (calories >= 0),
    protein NUMERIC DEFAULT 0 CHECK (protein >= 0),
    carbs NUMERIC DEFAULT 0 CHECK (carbs >= 0),
    fat NUMERIC DEFAULT 0 CHECK (fat >= 0),
    serving_size TEXT DEFAULT '100g',
    source TEXT DEFAULT 'manual',
    user_id UUID,  -- NULL = global catalog, non-NULL = user custom
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE meals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    meal_type TEXT NOT NULL CHECK (meal_type IN ('breakfast', 'lunch', 'dinner', 'snack')),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE meal_food_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meal_id UUID NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
    food_item_id UUID NOT NULL REFERENCES food_items(id),
    servings NUMERIC DEFAULT 1 CHECK (servings > 0),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- Seed data
-- ============================================================

-- Habit templates (catalog)
INSERT INTO habit_templates (id, name, slug, description, category, frequency) VALUES
    ('a1000000-0000-0000-0000-000000000001', 'Morning Meditation', 'morning-meditation', 'Start day with 10 minutes of mindfulness meditation', 'mindfulness', 'daily'),
    ('a1000000-0000-0000-0000-000000000002', 'Evening Journal', 'evening-journal', 'Reflect on the day in writing before bed', 'reflection', 'daily'),
    ('a1000000-0000-0000-0000-000000000003', 'Weekly Meal Prep', 'weekly-meal-prep', 'Prepare meals for the upcoming week', 'nutrition', 'weekly');

-- Habit events (user-scoped)
INSERT INTO habit_events (user_id, habit_template_id, date, response, notes) VALUES
    ('u1000000-0000-0000-0000-000000000001', 'a1000000-0000-0000-0000-000000000001', '2026-02-01', 'completed', 'Great session, felt calm'),
    ('u1000000-0000-0000-0000-000000000001', 'a1000000-0000-0000-0000-000000000001', '2026-02-02', 'skipped', 'Overslept'),
    ('u1000000-0000-0000-0000-000000000001', 'a1000000-0000-0000-0000-000000000002', '2026-02-01', 'completed', NULL);

-- Journal entries
INSERT INTO journal_entries (user_id, entry_type, payload) VALUES
    ('u1000000-0000-0000-0000-000000000001', 'freeform', '{"mood": "good", "themes": ["productivity", "sleep"], "text": "Slept well, productive day."}'),
    ('u1000000-0000-0000-0000-000000000001', 'gratitude', '{"items": ["sunny weather", "good workout", "healthy lunch"]}');

-- Food items (global catalog)
INSERT INTO food_items (id, name, slug, calories, protein, carbs, fat, serving_size, source) VALUES
    ('f1000000-0000-0000-0000-000000000001', 'Chicken Breast', 'chicken-breast', 165, 31, 0, 3.6, '100g', 'usda'),
    ('f1000000-0000-0000-0000-000000000002', 'Brown Rice', 'brown-rice', 216, 5, 45, 1.8, '1 cup cooked', 'usda'),
    ('f1000000-0000-0000-0000-000000000003', 'Salmon Fillet', 'salmon-fillet', 208, 20, 0, 13, '100g', 'usda'),
    ('f1000000-0000-0000-0000-000000000004', 'Sweet Potato', 'sweet-potato', 103, 2.3, 24, 0.1, '1 medium', 'usda'),
    ('f1000000-0000-0000-0000-000000000005', 'Greek Yogurt', 'greek-yogurt', 100, 17, 6, 0.7, '170g', 'usda');

-- Meals (user-scoped)
INSERT INTO meals (id, user_id, meal_type, notes) VALUES
    ('m1000000-0000-0000-0000-000000000001', 'u1000000-0000-0000-0000-000000000001', 'lunch', 'Post-workout meal'),
    ('m1000000-0000-0000-0000-000000000002', 'u1000000-0000-0000-0000-000000000001', 'dinner', NULL);

-- Meal-food links (M2M junction)
INSERT INTO meal_food_items (meal_id, food_item_id, servings) VALUES
    ('m1000000-0000-0000-0000-000000000001', 'f1000000-0000-0000-0000-000000000001', 1.5),
    ('m1000000-0000-0000-0000-000000000001', 'f1000000-0000-0000-0000-000000000002', 1),
    ('m1000000-0000-0000-0000-000000000002', 'f1000000-0000-0000-0000-000000000003', 1),
    ('m1000000-0000-0000-0000-000000000002', 'f1000000-0000-0000-0000-000000000004', 2);
