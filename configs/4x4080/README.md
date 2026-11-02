# Run configs: Qwen3.8 Flash Next UD-Q6_K_XL on 4× RTX 4080

Hardware: 4× RTX 4080 with 32 GB each (modded boards), AMD EPYC 7Y43 (48 cores, AVX2), 128 GB RAM, NVMe RAID0 for the
model. Linux. Model: Unsloth `Qwen3.8-Flash-Next-UD-Q6_K_XL` (6 shards), packed by Strata's setup. Context 384K (yarn
×1.5), KV `k8v4`. GPU 3 also drives the desktop, hence the bigger VRAM reserve on the last card.

| File | Mode | Engine | Measured |
|---|---|---|---|
| `solo-stock-strata.json` | one request at a time | **stock Strata** works | ru prose ~100 tok/s (MTP 78% accepted), new code ~114, code edit ~182, after 27K ~123; prefill 27K ~2160 tok/s (temperature 0.6) |
| `agents.json` | 8 slots, all experts resident in RAM | this fork | 1/2/4 agents: ~13/18/25 tok/s in total |
| `multi.json` | 4 slots, 40 GiB of experts pinned, rest streamed from NVMe | this fork | 1/2/4 agents: ~10/13/16 tok/s in total while ComfyUI runs on the same box |

Swarm numbers are the sum over all agents, from long agent turns (40–50K-token prompts, mostly from cache). The 78%
MTP acceptance also holds on the live server for single requests (78–81%).

## Placeholders

Replace before use:

| Placeholder | What |
|---|---|
| `<MODEL_DIR>` | folder with the six `Qwen3.8-Flash-Next-UD-Q6_K_XL-0000N-of-00006.gguf` shards |
| `<STRATA_DATA>` | Strata's data dir (`--data-dir`): `packs/`, `mtp/`, `models/` (mmproj) |
| `<STRATA_CONTROL>` | the Control package dir (`expert-profile.bin` lives in its `data/`); for the fork: your checkout |
| `<STRATA_ENGINE_PKG>` | the installed CUDA engine package (`strata-vision`, CUDA runtime libs) |
| `<FORK_ENGINE_BUILD>` | your build of this branch (folder with the `strata` binary) |

## The MTP drafter

- Head: stock, `tools/mtp_fetch.py` → `tools/mtp_pack.py`, `rt` format, Q2_0. A Q8_0 head raised acceptance only on long
  contexts and costs VRAM, so Q2_0 stays.
- `--mtp-draft-vocab draft_vocab_cyrillic.bin`: the stock English+code subset plus all Cyrillic, without CJK
  (`tools/draft_vocab.py --base data/draft_vocab.bin --add cyrillic`). Russian prose: 53% → 69% accepted, 68 → 83 tok/s.
- `--spec 8 --mtp-max-t 4 --spec-min-p 0.70`: the drafter only offers tokens it is ≥70% sure of. Acceptance goes up a
  lot, speed stays about the same.
- `--pipeline-windows 2`: +20% decode with `--parallel 1`; in batch mode the engine drops it unless
  `STRATA_PW_WITH_BATCH=1` (this fork).
- Batch only: `--batch-mtp`, `STRATA_PIPED_MTP`, `STRATA_SPLIT_MTP_BATCH=1` (stock Strata), `STRATA_SWARM_SPEC=1`
  (this fork).

## Serving the fork configs

`agents.json` / `multi.json` are served by this branch's control server:

```bash
cd <your checkout> && python -m serve.server --engine strata --config configs/4x4080/agents.json \
    --host 127.0.0.1 --port 8080 --parallel 8 --gpu 0,1,2,3
```

(`--parallel 4` for `multi.json`.) `STRATA_FAIR_USERS=1` goes in the server's environment. A setting that only exists
in this fork is ignored by stock Strata, but the speed numbers above need the fork. See [../../FORK.md](../../FORK.md).

`solo-stock-strata.json` is a stock Strata run config: put it where Strata's setup writes its own
(`strata-<pack>.json`) or pass it with `--config`.
