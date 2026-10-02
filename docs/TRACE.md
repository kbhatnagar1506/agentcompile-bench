# Trace format (v1)

One JSON object per line in `runs/<run>/trace.jsonl`, written as events happen. The scorer
reads only this file and `manifest.json`, so a run can be re-scored at any time, and anything
a metric needs has to be recorded here.

Every event carries `v` (format version), `ts` (unix seconds), `kind`, and the conversation's
context:

| Field | Meaning |
| --- | --- |
| `run` | run id |
| `conversation` | `run:suite/task:arm:trial`, also the id the SDK sends to AgentCompile |
| `suite`, `task`, `split` | which task; split is `train`, `dev` or `heldout` |
| `job`, `tags` | the task's job label and capability tags |
| `arm` | `baseline`, `sdk` or `shadow` |
| `agent_model` | `provider/model` |
| `trial` | 0-based trial index |

## Kinds

**`agent_call`**: one model call by the agent, through the (possibly wrapped) client. Carries the
live SDK trail's fields unchanged, so a bench run reads like production:

| Field | Meaning |
| --- | --- |
| `step`, `turn` | position in the conversation (turn = customer messages answered so far) |
| `route` | `baseline` (no SDK), or the SDK's `compiled`, `forwarded`, `fail-open`, `shadow`, `no-conversation` |
| `action`, `tool`, `reason` | the decision, as the SDK reported it |
| `events` | AgentCompile's decision events (job, questions, hand-off reasons) |
| `job` | the job named in `events`, if any |
| `decide_ms` | time the decision took, from the SDK |
| `total_ms` | wall time of the whole call, measured by the bench |
| `input_tokens`, `output_tokens`, `cost_usd` | agent tokens and list-price cost; 0 when compiled |
| `ac_cost_usd` | AgentCompile's own model cost for this call, summed from `events[].cost_usd` |

**`customer`**: one customer message (`turn`, `text`), with the customer model's tokens and cost
(0 for scripted customers). Kept apart from agent cost.

**`tool`**: one tool call: `tool`, `args`, `by` (`agent` or `compiled`: who chose it), `write`
(changes the database), `ok` (the tool accepted it), `result` (first 2,000 characters).

**`reply`**: one message to the customer: `by`, `text`, `turn_ms` (from the customer's message to
this reply, tool calls included).

**`conversation`**: written once at the end: `status` (`done`, `error`, `budget`), `ended`
(`customer_stop`, `stop_tool`, `max_steps`, ...), `success` (end state and outputs match;
`null` when the budget stopped it), `detail`, `turns`, `wall_ms`, `error`, and totals:
`agent_calls` (calls that reached the agent's model), `compiled_calls`, tokens, `agent_cost_usd`,
`customer_cost_usd`, `ac_cost_usd`, and `writes` as `{by: [sent, accepted]}`.

## Counting rules

- **An agent call** is a call that reached the agent's model. Compiled calls are counted apart.
- **Runtime writes** are writes chosen by a compiled answer; "sent" and "accepted" are counted
  separately (the first runtime bench sent 2 and had 0 accepted).
- **Conversations stopped by the budget** are kept in the trace but left out of every metric.
  **Errors** count as failures.
