"""Generate a small synthetic domain corpus for the continued-pretraining demo.

Real continued pretraining runs on billions of tokens of raw domain text.
Here we generate a tiny (~25-30k word) synthetic corpus covering PostgreSQL
internals, Java, distributed systems, and ServiceNow/Glide-style platform
docs -- just enough to demonstrate the *mechanics* (raw text -> next-token
prediction) end-to-end on Colab's free tier, not to teach a model real
domain knowledge.
"""

import random
from pathlib import Path

SEED = 42
OUTPUT_PATH = Path(__file__).parent / "data" / "domain_corpus.txt"
EVAL_OUTPUT_PATH = Path(__file__).parent / "data" / "domain_eval.json"

PARAGRAPHS_PER_DOMAIN = 25

# --- PostgreSQL internals --------------------------------------------------
PG_TEMPLATES = [
    "PostgreSQL uses {mechanism} to implement {feature}. {detail}",
    "In {feature}, the {component} is responsible for {action}. {detail}",
    "A common cause of {problem} in PostgreSQL is {cause}. The usual fix is {fix}.",
]
PG_SLOTS = {
    "mechanism": ["multi-version concurrency control (MVCC)", "write-ahead logging (WAL)",
                  "the buffer cache", "autovacuum", "the planner's cost model", "table partitioning"],
    "feature": ["transaction isolation", "crash recovery", "index-only scans",
                "vacuum and freezing", "replication", "query planning"],
    "component": ["background writer", "WAL writer", "autovacuum launcher",
                  "checkpointer", "query planner", "buffer manager"],
    "action": ["flushing dirty pages to disk", "reclaiming dead tuples",
               "writing WAL records before data pages", "estimating row counts",
               "choosing between a sequential scan and an index scan"],
    "detail": [
        "This avoids readers blocking writers and writers blocking readers.",
        "This ensures durability even if the server crashes mid-transaction.",
        "This keeps table and index bloat under control over time.",
        "This trades some write amplification for faster crash recovery.",
        "This relies on accurate table statistics gathered by ANALYZE.",
    ],
    "problem": ["table bloat", "replication lag", "long-running transaction blocking vacuum",
                "query plan regressions", "connection pool exhaustion"],
    "cause": ["autovacuum not keeping up with the update/delete rate",
              "a long-running transaction holding back the vacuum horizon",
              "stale table statistics after a bulk load",
              "too many idle connections holding resources"],
    "fix": ["tuning autovacuum thresholds per table", "running a manual VACUUM (or VACUUM FULL)",
            "running ANALYZE after large data changes", "adding a connection pooler like PgBouncer"],
}

# --- Java ------------------------------------------------------------------
JAVA_TEMPLATES = [
    "In Java, {concept} is used to {purpose}. {detail}",
    "The {component} in the JVM is responsible for {action}. {detail}",
    "A common mistake with {concept} is {mistake}, which leads to {consequence}.",
]
JAVA_SLOTS = {
    "concept": ["garbage collection", "the ConcurrentHashMap", "try-with-resources",
                "generics", "the volatile keyword", "CompletableFuture"],
    "purpose": ["automatically reclaim memory from unreachable objects",
                "allow safe concurrent reads and writes without a global lock",
                "guarantee a resource is closed even if an exception is thrown",
                "catch type errors at compile time instead of at runtime",
                "ensure visibility of a variable's latest value across threads",
                "compose asynchronous operations without blocking a thread"],
    "detail": [
        "Different garbage collectors (G1, ZGC, Shenandoah) trade pause time against throughput.",
        "This scales much better under contention than a synchronized HashMap.",
        "This is equivalent to an implicit finally block closing the resource.",
        "Type erasure means generic type information isn't available at runtime.",
        "It does not make compound operations atomic on its own.",
        "Exceptions inside the chain propagate through the returned future.",
    ],
    "component": ["garbage collector", "JIT compiler", "classloader", "thread scheduler"],
    "action": ["reclaiming memory from objects with no remaining references",
               "compiling hot methods to native code at runtime",
               "loading class bytecode into the running JVM",
               "deciding which runnable thread executes next"],
    "mistake": ["assuming += on a shared counter is atomic", "forgetting to override hashCode when overriding equals",
                "catching Exception too broadly", "blocking inside a reactive/async callback"],
    "consequence": ["lost updates under concurrent access", "broken behavior in hash-based collections",
                    "swallowing bugs that should have failed loudly", "thread pool starvation"],
}

# --- Distributed systems ---------------------------------------------------
DIST_TEMPLATES = [
    "In distributed systems, {concept} addresses the problem of {problem}. {detail}",
    "{algorithm} achieves {guarantee} by {mechanism}.",
    "A key trade-off in distributed systems is between {tradeoff_a} and {tradeoff_b}. {detail}",
]
DIST_SLOTS = {
    "concept": ["consensus", "leader election", "vector clocks", "quorum reads/writes",
                "idempotent request handling", "backpressure"],
    "problem": ["multiple nodes agreeing on a single value despite failures",
                "ensuring exactly one node coordinates a given task",
                "ordering events across nodes without a shared clock",
                "guaranteeing a majority of replicas have seen a write"],
    "detail": [
        "Protocols like Raft and Paxos solve this with majority voting over terms/rounds.",
        "This typically uses lease-based mechanisms with periodic renewal.",
        "This provides a partial order, not a total order, across events.",
        "Reading and writing to a majority of replicas tolerates minority failures.",
    ],
    "algorithm": ["Raft", "Paxos", "Two-Phase Commit", "Gossip protocols"],
    "guarantee": ["strong consistency across replicas", "atomic commitment across participants",
                  "eventual convergence of cluster state"],
    "mechanism": ["electing a leader and replicating a log with majority acknowledgment",
                  "a coordinator collecting votes before committing or aborting",
                  "nodes periodically exchanging state with random peers"],
    "tradeoff_a": ["consistency", "latency", "availability during a partition"],
    "tradeoff_b": ["availability", "throughput", "strict ordering guarantees"],
}

# --- ServiceNow / Glide-style platform docs --------------------------------
SN_TEMPLATES = [
    "On the ServiceNow platform, {feature} is implemented by {component}. {detail}",
    "When {scenario}, the platform relies on {mechanism} to {outcome}.",
    "A common operational concern with {feature} is {problem}, mitigated by {fix}.",
]
SN_SLOTS = {
    "feature": ["table archiving", "attachment storage partitioning", "ACL enforcement",
                "business rule execution order", "the Unreferenced Record Cleaner job"],
    "component": ["a scheduled background job that moves qualifying records to archive tables",
                  "a provider abstraction that routes attachments to the active storage partition",
                  "the security manager evaluating role and condition checks per operation",
                  "the business rule engine ordering rules by order field within each trigger phase"],
    "detail": [
        "This keeps the live table smaller while still letting archived data be queried via ar_ tables.",
        "This allows storage to scale horizontally as attachment volume grows.",
        "This runs on every record access, not just on save.",
        "Rules in the same order value execute in an implementation-defined but stable sequence.",
    ],
    "scenario": ["a record is dot-walked across tables on an archived form",
                 "a large table accumulates orphaned child records",
                 "an instance needs to add a new storage partition without downtime"],
    "mechanism": ["the ArchiveResolver fallback path", "a cleanup job that walks reference fields",
                  "a monitor job that triggers partition creation above a size threshold"],
    "outcome": ["resolve the correct archived parent record", "safely delete unreferenced rows",
                "route new uploads to the newly created partition"],
    "problem": ["forgetting to account for archived data in integration tests",
                "job contention when many large tables archive at the same time",
                "stale cached state after a schema change"],
    "fix": ["adding ar_ table coverage explicitly to IT suites",
            "staggering archive job schedules across tables",
            "invalidating the relevant caches on schema change"],
}


def render(templates, slots, rng, n):
    paragraphs = []
    for _ in range(n):
        template = rng.choice(templates)
        filled = template
        for key, options in slots.items():
            placeholder = "{" + key + "}"
            while placeholder in filled:
                filled = filled.replace(placeholder, rng.choice(options), 1)
        paragraphs.append(filled)
    return paragraphs


def main():
    rng = random.Random(SEED)

    domains = {
        "postgresql_internals": render(PG_TEMPLATES, PG_SLOTS, rng, PARAGRAPHS_PER_DOMAIN),
        "java": render(JAVA_TEMPLATES, JAVA_SLOTS, rng, PARAGRAPHS_PER_DOMAIN),
        "distributed_systems": render(DIST_TEMPLATES, DIST_SLOTS, rng, PARAGRAPHS_PER_DOMAIN),
        "servicenow_platform": render(SN_TEMPLATES, SN_SLOTS, rng, PARAGRAPHS_PER_DOMAIN),
    }

    all_paragraphs = []
    for domain, paras in domains.items():
        for p in paras:
            all_paragraphs.append(p)
    rng.shuffle(all_paragraphs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n\n".join(all_paragraphs))

    word_count = sum(len(p.split()) for p in all_paragraphs)
    print(f"Wrote {len(all_paragraphs)} paragraphs ({word_count} words) to {OUTPUT_PATH}")
    for domain, paras in domains.items():
        print(f"  {domain}: {len(paras)} paragraphs")

    # Small domain eval set: terminology, Q&A, summarization, reasoning.
    eval_set = [
        {"type": "terminology", "prompt": "What does MVCC stand for and why does PostgreSQL use it?"},
        {"type": "terminology", "prompt": "What is the purpose of Write-Ahead Logging (WAL)?"},
        {"type": "qa", "prompt": "Why can a long-running transaction block autovacuum in PostgreSQL?"},
        {"type": "qa", "prompt": "What problem does Raft's leader election solve?"},
        {"type": "summarization", "prompt": "Summarize in one sentence why ServiceNow uses table archiving."},
        {"type": "summarization", "prompt": "Summarize the trade-off between consistency and availability in distributed systems."},
        {"type": "reasoning", "prompt": "If a ServiceNow instance archives a table but doesn't update integration tests, what's likely to break?"},
        {"type": "reasoning", "prompt": "Why might using CompletableFuture incorrectly cause thread pool starvation?"},
    ]
    with open(EVAL_OUTPUT_PATH, "w") as f:
        import json
        json.dump(eval_set, f, indent=2)
    print(f"Wrote {len(eval_set)} domain eval prompts to {EVAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
