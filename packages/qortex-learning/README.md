# qortex-learning

Bandit-based adaptive learning for qortex. Thompson Sampling with Beta-Bernoulli posteriors, persistent state via SQLite, and pluggable reward models.

## Usage

```python
from qortex.learning import Learner, LearnerConfig, Arm, ArmOutcome

learner = await Learner.create(LearnerConfig(name="prompts"))

candidates = [Arm(id="v1", token_cost=10), Arm(id="v2", token_cost=15)]
result = await learner.select(candidates, context={"task": "type-check"}, k=1)

await learner.observe(ArmOutcome(arm_id="v2", outcome="accepted", reward=1.0))
```

Part of the [qortex](https://github.com/Peleke/qortex) workspace.
