# AgentCompile Bench

One benchmark run that tests everything an agent compiler claims: same answers, fewer agent calls, fewer wrong actions, correct hand-offs, latency, consistency, fail-open safety, and every provider. It unifies existing agent benchmarks under one runner and one scorecard, and adds the suites none of them have.

Status: design. The full spec and build plan are in [docs/SPEC.md](docs/SPEC.md).

```mermaid
flowchart LR
  S["Suites and adapters<br/>τ, τ², τ³, T1-Bench, AppWorld,<br/>AgentBench, BFCL + 10 added suites"] --> R["Runner<br/>suite × agent × model × arm<br/>budget cap, cached customer"]
  R --> A["Arms<br/>baseline · with SDK · chaos<br/>6 providers, 4 frameworks"]
  A --> T["Trace store<br/>one format"]
  T --> SC["Scorer"] --> C["Scorecard<br/>15 capabilities, bounds, pass/fail"]
```
