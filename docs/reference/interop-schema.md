# Interop Schema Reference

qortex uses JSON Schema to define the contract for consumer integration. Any system in any language can validate against these schemas.

## Schema Versions

| Schema | Version | Description |
|--------|---------|-------------|
| Seed | 1.0 | Universal rule set format |
| Event | 1.0 | Signal log event format |

## Seed Schema

The seed schema defines the universal rule set format used for all projections.

### Full Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://qortex.dev/schemas/seed.v1.schema.json",
  "title": "Qortex Seed",
  "description": "Universal rule set format for agent consumption",
  "type": "object",
  "required": ["persona", "version", "rules", "metadata"],
  "properties": {
    "persona": {
      "type": "string",
      "description": "Flat string identifier",
      "minLength": 1
    },
    "version": {
      "type": "integer",
      "description": "Schema version as integer",
      "minimum": 1
    },
    "rules": {
      "type": "array",
      "items": { "$ref": "#/$defs/rule" }
    },
    "metadata": { "$ref": "#/$defs/metadata" }
  }
}
```

### Example Seed

```yaml
persona: error_handling_rules
version: 1
rules:
  - rule: "Always configure timeouts for external calls"
    category: error_handling
    context: "When making HTTP requests or database queries"
    antipattern: "Calling external services without timeout limits"
    rationale: "Prevents cascading failures and resource exhaustion"
    tags:
      - timeout
      - resilience
      - error_handling
    provenance:
      id: rule:timeout
      domain: error_handling
      derivation: explicit
      confidence: 1.0
      relevance: 0.0
      source_concepts:
        - timeout
        - circuit_breaker
      relation_type: null
      template_id: null
      template_variant: null
      template_severity: null
      graph_version: "2026-02-05T12:00:00Z"

  - rule: "Circuit Breaker requires Timeout to function correctly"
    category: dependency
    context: "When implementing the Circuit Breaker pattern"
    antipattern: "Using Circuit Breaker without timeout configuration"
    rationale: "Circuit Breaker needs timeout to detect failures"
    tags:
      - circuit_breaker
      - timeout
      - derived
    provenance:
      id: derived:circuit_breaker->timeout:imperative
      domain: error_handling
      derivation: derived
      confidence: 0.95
      relevance: 0.0
      source_concepts:
        - circuit_breaker
        - timeout
      relation_type: requires
      template_id: requires:imperative
      template_variant: imperative
      template_severity: null
      graph_version: "2026-02-05T12:00:00Z"

metadata:
  source: qortex
  source_version: "0.1.0"
  projected_at: "2026-02-05T12:00:00Z"
  rule_count: 2
```

### Rule Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `rule` | string | Yes | The rule text itself |
| `category` | string | Yes | Category for filtering |
| `context` | string | No | When this rule applies |
| `antipattern` | string | No | What violating looks like |
| `rationale` | string | No | Why this matters |
| `tags` | array[string] | No | Searchable keywords |
| `provenance` | object | Yes | Origin and derivation metadata |

### Provenance Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique rule identifier |
| `domain` | string | Yes | Knowledge domain |
| `derivation` | enum | Yes | "explicit" or "derived" |
| `confidence` | number | Yes | Confidence score (0-1) |
| `relevance` | number | No | Retrieval relevance score |
| `source_concepts` | array[string] | No | Concept IDs this derives from |
| `relation_type` | string\|null | No | Edge type (for derived rules) |
| `template_id` | string\|null | No | Template ID used |
| `template_variant` | string\|null | No | Template variant |
| `template_severity` | string\|null | No | Template severity |
| `graph_version` | string\|null | No | Graph state timestamp |

### Metadata Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | Origin system ("qortex") |
| `source_version` | string | No | System version |
| `projected_at` | string | No | ISO timestamp of projection |
| `rule_count` | integer | Yes | Number of rules |

## Event Schema

The event schema defines signal log entries.

### Full Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://qortex.dev/schemas/event.v1.schema.json",
  "title": "Qortex Interop Event",
  "description": "Signal log event for the consumer interop protocol",
  "type": "object",
  "required": ["event", "ts", "source"],
  "properties": {
    "event": {
      "type": "string",
      "enum": ["projection_complete", "seed_ingested", "seed_failed"]
    },
    "persona": { "type": "string" },
    "domain": { "type": "string" },
    "path": { "type": "string" },
    "rule_count": { "type": "integer", "minimum": 0 },
    "ts": { "type": "string", "format": "date-time" },
    "source": { "type": "string" },
    "source_version": { "type": "string" },
    "error": { "type": "string" }
  },
  "additionalProperties": true
}
```

### Event Types

| Event | Emitter | Description |
|-------|---------|-------------|
| `projection_complete` | qortex | New seed written to pending |
| `seed_ingested` | consumer | Seed successfully processed |
| `seed_failed` | consumer | Seed validation/processing failed |

### Example Events

```jsonl
{"event":"projection_complete","persona":"error_rules","domain":"error_handling","path":"/home/user/.qortex/seeds/pending/error_rules_2026-02-05T12-00-00.yaml","rule_count":5,"ts":"2026-02-05T12:00:00Z","source":"qortex","source_version":"0.1.0"}
{"event":"seed_ingested","persona":"error_rules","path":"/home/user/.qortex/seeds/processed/error_rules_2026-02-05T12-00-00.yaml","ts":"2026-02-05T12:01:00Z","source":"buildlog","source_version":"1.2.0"}
{"event":"seed_failed","persona":"bad_seed","path":"/home/user/.qortex/seeds/failed/bad_seed.yaml","ts":"2026-02-05T12:02:00Z","source":"buildlog","error":"Missing required field: rules"}
```

## Exporting Schemas

Export schema files for any-language validation:

```bash
qortex interop schema --output ./schemas/
```

Creates:
- `schemas/seed.v1.schema.json`
- `schemas/event.v1.schema.json`

Or programmatically:

```python
from qortex.interop_schemas import export_schemas

seed_path, event_path = export_schemas("./schemas/")
```

## Validation

### Python

```python
from qortex.interop_schemas import validate_seed, validate_event

# Validate a seed
errors = validate_seed(seed_dict)
if errors:
    print(f"Invalid: {errors}")

# Validate an event
errors = validate_event(event_dict)
```

With full JSON Schema validation (requires `jsonschema` package):

```python
import jsonschema
from qortex.interop_schemas import SEED_SCHEMA

validator = jsonschema.Draft202012Validator(SEED_SCHEMA)
errors = list(validator.iter_errors(seed_dict))
```

### JavaScript/TypeScript

```javascript
const Ajv = require("ajv");
const addFormats = require("ajv-formats");

const ajv = new Ajv();
addFormats(ajv);

const schema = require("./seed.v1.schema.json");
const validate = ajv.compile(schema);

if (!validate(seed)) {
  console.log(validate.errors);
}
```

### Go

```go
import "github.com/xeipuuv/gojsonschema"

schemaLoader := gojsonschema.NewReferenceLoader("file:///path/to/seed.v1.schema.json")
documentLoader := gojsonschema.NewGoLoader(seed)

result, _ := gojsonschema.Validate(schemaLoader, documentLoader)
if !result.Valid() {
    for _, err := range result.Errors() {
        fmt.Println(err)
    }
}
```

### CLI

```bash
qortex interop validate seed.yaml
```

## Schema Evolution

Schemas are versioned independently of qortex:

- **Patch**: Bug fixes, clarifications (no version bump)
- **Minor**: Additive changes, new optional fields (1.0 -> 1.1)
- **Major**: Breaking changes (1.x -> 2.0)

Consumers should:
1. Check `version` field in seeds
2. Reject seeds with major version mismatch
3. Accept seeds with higher minor versions (forward compatible)

## Key Design Decisions

### `persona` is flat string

```yaml
# Correct
persona: error_handling_rules

# Wrong (old format)
persona:
  name: error_handling_rules
  description: ...
```

Consumers use persona as filename, so flat string is simpler.

### `version` is integer

```yaml
# Correct
version: 1

# Wrong
version: "1.0.0"
```

Integer comparison is simpler than semver parsing.

### `rule` not `text`

```yaml
# Correct
rules:
  - rule: "Always configure timeouts"

# Wrong (old format)
rules:
  - text: "Always configure timeouts"
```

More explicit and consistent with field naming.

### `provenance` groups metadata

```yaml
# Correct
rules:
  - rule: "..."
    provenance:
      id: rule:timeout
      domain: error_handling

# Wrong (old format)
rules:
  - rule: "..."
    id: rule:timeout
    domain: error_handling
```

Grouping makes it clear what's informational vs operational.

## Security Notes

Schemas validate structure, not content. Consumers must:

1. **Sanitize rule text** before using in prompts
2. **Don't execute** anything from seed files
3. **Validate paths** from events before file operations
4. **Don't trust provenance** for authorization decisions
