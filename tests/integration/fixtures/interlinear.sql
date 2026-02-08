-- interlinear database: language learning platform
-- Simulates a Supabase-like single PostgreSQL instance (port 5433 in prod)
-- Features: JSONB columns, multilingual data, CHECK constraints, CASCADE deletes

CREATE TABLE courses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    language_code TEXT NOT NULL CHECK (language_code IN ('es', 'la', 'fr', 'de', 'it', 'pt')),
    level TEXT NOT NULL CHECK (level IN ('a1', 'a2', 'b1', 'b2', 'c1', 'c2')),
    description TEXT,
    is_published BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE lessons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    markdown_content TEXT,
    summary TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE vocabulary (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    word TEXT NOT NULL,
    translation TEXT NOT NULL,
    language_code TEXT NOT NULL,
    part_of_speech TEXT CHECK (part_of_speech IN ('noun', 'verb', 'adjective', 'adverb', 'preposition', 'conjunction', 'pronoun', 'interjection')),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE lesson_vocabulary_items (
    lesson_id UUID NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    vocabulary_id UUID NOT NULL REFERENCES vocabulary(id),
    position INTEGER DEFAULT 0,
    PRIMARY KEY (lesson_id, vocabulary_id)
);

CREATE TABLE grammar_concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    language_code TEXT NOT NULL,
    description TEXT,
    markdown_content TEXT,
    level TEXT CHECK (level IN ('a1', 'a2', 'b1', 'b2', 'c1', 'c2')),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE exercises (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lesson_id UUID NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    exercise_type TEXT NOT NULL CHECK (exercise_type IN ('fill_blank', 'multiple_choice', 'translation', 'conjugation', 'listening')),
    prompt TEXT NOT NULL,
    correct_answer TEXT NOT NULL,
    options JSONB,  -- For multiple choice: ["opt1", "opt2", "opt3"]
    difficulty INTEGER DEFAULT 1 CHECK (difficulty BETWEEN 1 AND 5),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE ai_generation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lesson_id UUID REFERENCES lessons(id),
    model_name TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- Seed data
-- ============================================================

-- Courses
INSERT INTO courses (id, title, slug, language_code, level, description, is_published) VALUES
    ('c1000000-0000-0000-0000-000000000001', 'Spanish for Beginners', 'spanish-a1', 'es', 'a1', 'Learn basic Spanish from scratch', true),
    ('c1000000-0000-0000-0000-000000000002', 'Latin Fundamentals', 'latin-a1', 'la', 'a1', 'Introduction to Classical Latin grammar', true);

-- Lessons
INSERT INTO lessons (id, course_id, title, position, markdown_content, summary) VALUES
    ('e1000000-0000-0000-0000-000000000001', 'c1000000-0000-0000-0000-000000000001', 'Greetings and Introductions', 0, '# Saludos\n\nLearn how to greet people in Spanish.\n\n- Hola = Hello\n- Buenos días = Good morning\n- ¿Cómo estás? = How are you?', 'Basic Spanish greetings'),
    ('e1000000-0000-0000-0000-000000000002', 'c1000000-0000-0000-0000-000000000001', 'Numbers 1-20', 1, '# Los Números\n\nuno, dos, tres, cuatro, cinco...', 'Spanish numbers'),
    ('e1000000-0000-0000-0000-000000000003', 'c1000000-0000-0000-0000-000000000002', 'The Present Tense (Praesens)', 0, '# The Present Tense\n\nLatin verbs are conjugated by adding endings to the verb stem.\n\n## First Conjugation (-āre)\n\namō, amās, amat, amāmus, amātis, amant', 'Latin present tense conjugation');

-- Vocabulary (multilingual)
INSERT INTO vocabulary (id, word, translation, language_code, part_of_speech, notes) VALUES
    ('a0100000-0000-0000-0000-000000000001', 'hola', 'hello', 'es', 'interjection', 'Informal greeting'),
    ('a0100000-0000-0000-0000-000000000002', 'buenos días', 'good morning', 'es', 'noun', 'Formal morning greeting'),
    ('a0100000-0000-0000-0000-000000000003', 'amāre', 'to love', 'la', 'verb', 'First conjugation (-āre)'),
    ('a0100000-0000-0000-0000-000000000004', 'uno', 'one', 'es', 'noun', 'Cardinal number'),
    ('a0100000-0000-0000-0000-000000000005', 'dos', 'two', 'es', 'noun', 'Cardinal number');

-- Lesson-vocabulary links (M2M junction)
INSERT INTO lesson_vocabulary_items (lesson_id, vocabulary_id, position) VALUES
    ('e1000000-0000-0000-0000-000000000001', 'a0100000-0000-0000-0000-000000000001', 0),
    ('e1000000-0000-0000-0000-000000000001', 'a0100000-0000-0000-0000-000000000002', 1),
    ('e1000000-0000-0000-0000-000000000002', 'a0100000-0000-0000-0000-000000000004', 0),
    ('e1000000-0000-0000-0000-000000000002', 'a0100000-0000-0000-0000-000000000005', 1),
    ('e1000000-0000-0000-0000-000000000003', 'a0100000-0000-0000-0000-000000000003', 0);

-- Grammar concepts
INSERT INTO grammar_concepts (id, name, slug, language_code, description, level) VALUES
    ('90100000-0000-0000-0000-000000000001', 'Present Tense', 'present-tense', 'la', 'The praesens tense for ongoing or habitual actions', 'a1'),
    ('90100000-0000-0000-0000-000000000002', 'First Conjugation', 'first-conjugation', 'la', 'Verbs ending in -āre (e.g., amāre, laudāre)', 'a1');

-- Exercises
INSERT INTO exercises (id, lesson_id, exercise_type, prompt, correct_answer, options, difficulty) VALUES
    ('ee100000-0000-0000-0000-000000000001', 'e1000000-0000-0000-0000-000000000001', 'translation', 'Translate: Hello', 'Hola', NULL, 1),
    ('ee100000-0000-0000-0000-000000000002', 'e1000000-0000-0000-0000-000000000001', 'multiple_choice', 'How do you say "good morning"?', 'Buenos días', '["Buenos días", "Buenas noches", "Buenas tardes", "Hola"]', 1),
    ('ee100000-0000-0000-0000-000000000003', 'e1000000-0000-0000-0000-000000000003', 'conjugation', 'Conjugate amāre in 1st person singular present', 'amō', NULL, 2);

-- AI generation logs (metadata domain)
INSERT INTO ai_generation_logs (lesson_id, model_name, prompt_tokens, completion_tokens, metadata) VALUES
    ('e1000000-0000-0000-0000-000000000001', 'claude-sonnet-4-5-20250929', 1200, 800, '{"task": "exercise_generation", "quality_score": 0.92}'),
    ('e1000000-0000-0000-0000-000000000003', 'claude-sonnet-4-5-20250929', 1500, 1100, '{"task": "grammar_explanation", "quality_score": 0.88}');
