# Synthesized Social Identity

Agent-based model of opinion dynamics that combines three mechanisms related to social identity:

- **SIA** (Social Identity Approach): each agent holds an identity content, initialised to its group's mean opinion.
- **IPT** (Identity Protection): when an observed opinion is too far from the identity content (a "threat"), the agent pulls its opinion back toward that content.
- **ICT** (Identity Change): when self-esteem is depleted, the identity content slowly shifts toward observed opinions.

Agents are split into a majority group A (90) and a minority group B (10). Each step, every agent observes a random other agent. Non-threatening opinions are assimilated and self-esteem recharges. Threatening ones trigger defensive opinion change and drain self-esteem. When self-esteem is depleted, identity content adjusts and self-esteem changes. 

## Scenarios

`main.py` runs 100 seeds for four conditions:

| Condition | Output suffix |
|---|---|
| Baseline (all mechanisms) | *(none)* |
| Without SIA | `_noSIA` |
| Without IPT | `_noIPT` |
| Without ICT | `_noIT` |

## Setup

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
uv run main.py
```

This creates a `.venv` and installs the dependencies from `pyproject.toml` (pinned in `uv.lock`):

| Package | Used for |
|---|---|
| `numpy`, `pandas` | simulation and data handling |
| `matplotlib` | plots |
| `ipykernel` | running the `#%%` cells interactively |


## Output

- `data/baseline*.csv`: opinions, identity standards, self-esteem and threat per agent, recorded every 10 steps.
- `figs/baseline*.png`: ensemble mean over all seeds.
- `figs/baseline*_exampleSeed48.png`: a single example run.

Parameters (group sizes, `alpha`, `gamma`, `beta`, etc.) are defined at the top of `main.py`.

## Files

- `main.py`: vectorised simulation and plotting.


## Contributions

- Clara Rademacher (Model design and implementation)
- Peter Steiglechner (Model design and implementation)