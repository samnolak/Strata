"""tools/mtp_rt.py - plan v0.3 P6: the MTP draft layer's runtime files, from the packed MTP GGUF.

    python tools/mtp_rt.py --gguf <Strata>/mtp-bf16/mtp-q2_0.gguf --out <Strata>/mtp-bf16/rt

Writes
  experts.bin   512 routed experts in the engine's blob layout (`include/strata/kernels/cpu/expert.hpp`): gate/up
                rows interleaved (2r = gate r, 2r+1 = up r), then down rows; the Q2_0 codes in one plane and the fp16
                scales in another.  A lossless relayout of the GGUF's Q2_0 blocks (same bytes, `cpu_expert_fixture.py`).
  dense.bin     every other tensor: the large projections quantized to Q8_0 (ggml's reference rounding) so the
                engine's multi-column MMVQ can run them; the hyper-connection and router weights kept BF16; the
                RMSNorm weights as F32 with the Gemma "+1" applied (vLLM GemmaRMSNorm scales by 1 + w).
  dense.txt     one line per tensor: name kind rows cols offset bytes   (kind = q8_0 | bf16 | f32)

A GGUF packed with `mtp_pack.py --experts q8_0` (or q4_0) gives instead
  experts.bin   512 blobs in the native expert layout (`NativeExpertLayout`, `include/strata/kernels/iq_kernels.hpp`):
                [gate rows | up rows | down rows], the GGUF's own blocks unchanged (Q8_0: 5,222,400 B per expert)
  experts.txt   the format marker the engine reads: `<format> <gu_type> <d_type> <n_embd> <n_ff> <n_expert> <blob_bytes>`
                (e.g. `q8_0 8 8 2560 640 512 5222400`).  No experts.txt = the Q2_0 planar layout above, so a Q2_0
                rt directory is written exactly as before.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402

H, FF, NE = 2560, 640, 512
BLOB = 3 * (H * FF * 18 // 64)
Q8 = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
      "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
      "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"}


def q8_0(x: np.ndarray) -> bytes:
    """ggml quantize_row_q8_0_ref over rows of a 2-D float32 array: 32-value blocks, fp16 d = amax / 127."""
    b = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def blob_of(gu: np.ndarray, dn: np.ndarray) -> bytes:
    """gu: (1280, 720) Q2_0 rows (gate rows 0..639, up rows 640..1279); dn: (2560, 180)."""
    gub = gu.reshape(2 * FF, H // 64, 18)
    inter = np.empty_like(gub)
    inter[0::2] = gub[:FF]
    inter[1::2] = gub[FF:]
    dnb = dn.reshape(H, FF // 64, 18)
    parts = [inter[:, :, 2:].tobytes(), dnb[:, :, 2:].tobytes(), inter[:, :, :2].tobytes(), dnb[:, :, :2].tobytes()]
    out = b"".join(parts)
    assert len(out) == BLOB, len(out)
    return out


# formats with a native-layout runtime blob: name -> (ggml type id, values per block, bytes per block)
NATIVE = {"q8_0": (8, 32, 34), "q4_0": (2, 32, 18)}


def native_blob_of(gu: np.ndarray, dn: np.ndarray, rb_gu: int, rb_dn: int) -> bytes:
    """gu: (1280, rb_gu) rows of GGUF blocks (gate rows 0..639, up rows 640..1279); dn: (2560, rb_dn).  The blob is
    the rows as they are: gate rows, then up rows (up_off = 640 * rb_gu), then down rows (down_off = 2 * up_off)."""
    out = np.ascontiguousarray(gu).tobytes() + np.ascontiguousarray(dn).tobytes()
    assert len(out) == 2 * FF * rb_gu + H * rb_dn, len(out)
    return out


def expert_format(r, gu_t, dn_t) -> str:
    """strata.mtp.expert_format from mtp_pack.py, checked against the tensors' ggml types."""
    fmt = None
    f = r.fields.get("strata.mtp.expert_format")
    if f is not None:
        fmt = bytes(f.parts[f.data[0]]).decode("utf-8")
    by_type = {42: "q2_0", 8: "q8_0", 2: "q4_0"}
    t_gu, t_dn = int(gu_t.tensor_type), int(dn_t.tensor_type)
    if t_gu != t_dn or t_gu not in by_type:
        raise SystemExit(f"mtp_rt: expert tensor types {t_gu}/{t_dn} are not one supported format (q2_0, q8_0, q4_0)")
    if fmt is not None and fmt != by_type[t_gu]:
        raise SystemExit(f"mtp_rt: strata.mtp.expert_format {fmt!r} but the expert tensors are {by_type[t_gu]}")
    return by_type[t_gu]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = gguf.GGUFReader(a.gguf)
    tens = {t.name: t for t in r.tensors}
    gu_t, dn_t = tens["mtp.layers.0.mlp.experts.gate_up_proj"], tens["mtp.layers.0.mlp.experts.down_proj"]
    fmt = expert_format(r, gu_t, dn_t)
    gu = np.asarray(gu_t.data)
    dn = np.asarray(dn_t.data)
    marker = out / "experts.txt"
    if fmt == "q2_0":
        assert gu.shape == (NE, 2 * FF, 720) and dn.shape == (NE, H, 180), (gu.shape, dn.shape)
        if marker.exists():   # a directory reused for Q2_0: the engine must not read a stale native marker
            marker.unlink()
        with open(out / "experts.bin", "wb") as f:
            for e in range(NE):
                f.write(blob_of(gu[e], dn[e]))
        expert_bytes = NE * BLOB
    else:
        ty, qk, bb = NATIVE[fmt]
        rb_gu, rb_dn = H // qk * bb, FF // qk * bb
        assert gu.shape == (NE, 2 * FF, rb_gu) and dn.shape == (NE, H, rb_dn), (gu.shape, dn.shape)
        blob = 2 * FF * rb_gu + H * rb_dn
        with open(out / "experts.bin", "wb") as f:
            for e in range(NE):
                f.write(native_blob_of(gu[e], dn[e], rb_gu, rb_dn))
        expert_bytes = NE * blob
        marker.write_text(f"{fmt} {ty} {ty} {H} {FF} {NE} {blob}\n", encoding="utf-8")
    lines = []
    off = 0
    with open(out / "dense.bin", "wb") as f:
        for name, t in tens.items():
            if "experts." in name:
                continue
            short = name[len("mtp."):]
            short = short[len("layers.0."):] if short.startswith("layers.0.") else short
            data = np.asarray(t.data)
            if int(t.tensor_type) == 0:        # F32 norm weights: raw GemmaRMSNorm w -> 1 + w
                arr = (data.astype(np.float32) + 1.0)
                raw, kind, rows, cols = arr.tobytes(), "f32", 1, arr.size
            else:                              # BF16
                u16 = data.view(np.uint16) if data.dtype != np.uint16 else data
                u16 = u16.reshape(data.shape[0], -1)
                rows, cols = u16.shape
                if short in Q8:
                    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                    raw, kind = q8_0(f32), "q8_0"
                else:
                    raw, kind = u16.tobytes(), "bf16"
            pad = (-off) % 256
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append(f"{short} {kind} {rows} {cols} {off} {len(raw)}")
            off += len(raw)
    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"experts.bin {expert_bytes} B ({fmt}), dense.bin {off} B, {len(lines)} tensors -> {out}")
    for l in lines:
        print("  " + l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
