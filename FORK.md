# samnolak/Strata — linux-multigpu

A fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) tuned for one box: **4× RTX 4080 (sm_89, PCIe, no
NVLink), Linux, a large MoE model at UD-Q6 with experts offloaded to RAM and NVMe**. Everything here sits on top of
upstream **v0.2.9** and is rebased onto each new upstream release.

The `main` branch mirrors upstream. The fork's work lives in `linux-multigpu`.

**Every change is opt-in.** With none of the variables below set, the engine behaves as upstream v0.2.9.

## Own changes

| Area | Switch | What it does |
|---|---|---|
| SSD offload | `STRATA_REFILL_PAR` | parallel pinned refill of streamed experts |
| | `STRATA_WALK_PREFETCH` | prefetch along the staged layer walk (falls back to #4695's path) |
| | `STRATA_SSD_AHEAD` | read-ahead for the next pipeline stage |
| | `STRATA_SSD_TRACE` | per-read SSD trace |
| KV memory | `STRATA_ROPE_SHARE=1` | one RoPE table per device, shared across sessions |
| | `STRATA_KV_HOST_LAZY=1` | host K/V of a streamed state is pinned lazily as the context grows, with a RAM floor |
| CPU experts | `STRATA_CPU_AVX2_MT=1` | multi-token AVX-2 rows for Q6_K gate/up and Q8_0 (bit-exact, parity test included) |
| Swarm / batch | `STRATA_SWARM_SPEC=1\|2` | pipelined slot groups verify the proposals one global schedule picks (SwarmSchedule, CPU-tested) |
| | `STRATA_BATCH_CPU_SHARE` | batch admissions' prompt reads give the idle CPU pool its expert share |
| | `STRATA_KEEP_LOAN_MS`, `STRATA_KEEP_LOAN_MAX` | a batch admission keeps its prompt loan for the next one |
| Serving | `STRATA_SLOT_AFFINITY` and related | admission by uncached prompt length, cache-aware slot choice |
| Drafter | opt-in Q8_0 / Q4_0 routed experts for the MTP drafter (issue #3728) | |
| Prompt path | — | MMQ row-id table carries the column tile's slack (fix) |
| | `STRATA_MMQ_DIRECT=1`, `STRATA_Q6K_SPLIT` | Q6_K in the pointer-table MMQ |
| | `STRATA_DENSE_MMQ` | Q8_0 dense prompt projections as int8 MMQ (not bit-exact, off in the configs) |
| Stage copies | `STRATA_STAGE_COPY_PARALLEL=1` | staged conversation-cache copies in parallel (with PR #4002's merge side) |
| | `STRATA_STAGE_PREFETCH=1`, `STRATA_PF_ROUTE_EVENT=1` | prefetch for the next stage, routed by an event |
| | `STRATA_IDLE_PARK` | park an idle slot's conversation (not used in the configs) |
| VRAM | `--vram-reserve-per-dev` / `STRATA_VRAM_RESERVE_PER_DEV` | VRAM reserve per card (e.g. more on the display GPU) |
| Checkpoints | `STRATA_BRANCH_CHECKPOINT_ALIGN=1` | exact branch checkpoints on the super-chunk grid |
| Control server | `STRATA_FAIR_USERS=1` | batch slots shared fairly between users, not requests |
| | `STRATA_TOOLCALL_RETRY=N` | re-generate a broken tool call (off in the configs: costs speed) |
| | `STRATA_REASONING_CAP_FRAC` | thinking capped at a share of max_tokens |
| | — | tool names escaped on the wire (`_read`, `_bash`) match their unprefixed form |

Details for the batch and swarm switches: [docs/BATCHING.md](docs/BATCHING.md).

## Upstream PRs carried here

This branch also merges several upstream pull requests that are not in upstream `main` yet. Their commits keep their
original authors: slot placement policy and checkpoints (andyhrc), the pipelined decode loop and the 11-output draft
chain (ruibeikaa), `STRATA_SLOT_EXPIRE_S` (SuperDistance), `STRATA_PW_WITH_BATCH` (Vop3p), `STRATA_HC_Q8` on layer
splits (havok2-htwo, Markus Paltian), `STRATA_SELECT_TWO_PASS` (Stefano Sinden), parallel staged cache copies
(shangfeng-liu), the lost-experts fix after the pipelined loop (uvlad23).

## Configs

[configs/4x4080/](configs/4x4080/) holds the run configs this branch is used with (solo, agents, multi-model), with
the measured numbers and what each switch needs.

## Build

Same as upstream (see [docs/INSTALL.md](docs/INSTALL.md)). The configuration used here for sm_89:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="89-real;89-virtual"
cmake --build build -j
```

## License

MIT, as upstream. See [LICENSE](LICENSE).
