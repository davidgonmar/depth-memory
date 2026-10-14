"""Triton kernels of the flexible backend: one fused write/read step and its reverse.

One program owns a tile of token/group rows and their K x D memory states. The forward handles the
memory as K row vectors so that its contractions over K are short static loops and the D axis stays
elementwise; the reverse keeps the K x D tile. Tile height and warp count are autotuned per (K, D,
rule). All arithmetic is FP32 in registers; keys, values, reads and the residual tape use the
model's working dtype, the memory M is stored in BF16, the covariance P in its configured storage
dtype, and gains, denominators, forgetting factors and the P cotangent are FP32. RLS forgets through P; the delta rule and the outer product decay M by the
forgetting factor before each write. M is rounded to its storage dtype after every write. Every
write also keeps the covariance it started from on a tape, so the reverse restores P exactly and
unwinds only M in place; unwinding a rounded M is approximate, and the first write of a segment is
restored from the exact constant M = 0 instead.
"""

import torch
import triton
import triton.language as tl

RLS, DELTA_RULE, OUTER_PRODUCT = 0, 1, 2


def _configs():
    return [triton.Config({"BN": rows}, num_warps=warps)
            for rows in (8, 16, 32, 64, 128) for warps in (2, 4, 8)]


def _prune(configs, named_args, **kwargs):
    # Keep the per-program tile within 16384 elements: wider states get fewer rows.
    args = {**named_args, **kwargs}
    tile = args["BD"] * args["K"]
    kept = [c for c in configs if c.kwargs["BN"] * tile <= 16384]
    return kept or configs[:3]


_AUTOTUNE = dict(key=["K", "D", "RULE"], prune_configs_by={"early_config_prune": _prune})


@triton.autotune(configs=_configs(), restore_value=["M", "P", "CovTape", "Cursor"], **_AUTOTUNE)
@triton.jit
def _forward(
    M,
    P,
    CovTape,
    Key,
    Value,
    Query,
    Lam,
    Out,
    Error,
    Gain,
    Den,
    LamOut,
    Cursor,
    Slot,
    N,
    K: tl.constexpr,
    D: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    RULE: tl.constexpr,
):
    # One program owns BN token/group rows. The memory is handled as K row vectors of D, so every
    # contraction over K is a short static loop and the D axis stays elementwise.
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    nm = n < N
    d = tl.arange(0, BD)
    dm = d < D
    rm = nm[:, None] & dm[None, :]
    vo = n[:, None] * D + d[None, :]
    lam = tl.load(Lam + n, nm, 1.0)
    v = tl.load(Value + vo, rm, 0).to(tl.float32)
    if RULE == 0:
        j = tl.arange(0, BK)
        jm = j < K
        km = nm[:, None] & jm[None, :]
        kvec = tl.load(Key + n[:, None] * K + j[None, :], km, 0).to(tl.float32)
        # u = P k, den = lambda + k.u, gain = u / den, P <- (P - u u^T / den) / lambda
        den = lam
        uvec = tl.zeros([BN, BK], tl.float32)
        for i in tl.static_range(K):
            po = n[:, None] * K * K + i * K + j[None, :]
            prow = tl.load(P + po, km, 0).to(tl.float32)
            tl.store(CovTape + po, prow.to(CovTape.dtype.element_ty), km)
            ui = tl.sum(prow * kvec, 1)
            ki = tl.load(Key + n * K + i, nm, 0).to(tl.float32)
            den += ki * ui
            uvec = tl.where(j[None, :] == i, ui[:, None], uvec)
        for i in tl.static_range(K):
            po = n[:, None] * K * K + i * K + j[None, :]
            prow = tl.load(P + po, km, 0).to(tl.float32)
            ui = tl.sum(tl.where(j[None, :] == i, uvec, 0.0), 1)
            pnew = (prow - ui[:, None] * uvec / den[:, None]) / lam[:, None]
            tl.store(P + po, pnew.to(P.dtype.element_ty), km)
    else:
        den = tl.full([BN], 1.0, tl.float32)
    # error = v - M^T k for RLS, v - (lambda M)^T k for the delta rule, v for the outer product
    error = v
    if RULE != 2:
        for i in tl.static_range(K):
            mrow = tl.load(M + n[:, None] * K * D + i * D + d[None, :], rm, 0).to(tl.float32)
            if RULE != 0:
                mrow = mrow * lam[:, None]
            ki = tl.load(Key + n * K + i, nm, 0).to(tl.float32)
            error -= ki[:, None] * mrow
    out = tl.zeros([BN, BD], tl.float32)
    for i in tl.static_range(K):
        mo = n[:, None] * K * D + i * D + d[None, :]
        mrow = tl.load(M + mo, rm, 0).to(tl.float32)
        ki = tl.load(Key + n * K + i, nm, 0).to(tl.float32)
        if RULE == 0:
            gi = tl.sum(tl.where(j[None, :] == i, uvec, 0.0), 1) / den
        else:
            mrow = mrow * lam[:, None]
            gi = ki
        updated = (mrow + gi[:, None] * error).to(M.dtype.element_ty)
        tl.store(M + mo, updated, rm)
        qi = tl.load(Query + n * K + i, nm, 0).to(tl.float32)
        out += updated.to(tl.float32) * qi[:, None]
        tl.store(Gain + n * K + i, gi, nm)
    tl.store(Out + vo, out.to(Out.dtype.element_ty), rm)
    tl.store(Error + vo, error.to(Error.dtype.element_ty), rm)
    tl.store(Den + n, den, nm)
    tl.store(LamOut + n, lam, nm)
    if tl.program_id(0) == 0:
        tl.store(Cursor, tl.load(Slot) + 1)


def _reverse_configs():
    # The reverse keeps the K x D tile of every row live; few rows per program measures fastest.
    return [triton.Config({"BN": rows}, num_warps=warps)
            for rows in (1, 2, 4, 8) for warps in (1, 2, 4)]


@triton.autotune(configs=_reverse_configs(), restore_value=["M", "P", "GM", "GP", "Cursor"], **_AUTOTUNE)
@triton.jit
def _reverse(
    M,
    P,
    CovTape,
    GM,
    GP,
    Key,
    Query,
    Lam,
    Error,
    Gain,
    Den,
    DY,
    DK,
    DV,
    DQ,
    DL,
    Cursor,
    Slot,
    N,
    K: tl.constexpr,
    D: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    RULE: tl.constexpr,
):
    slot = tl.load(Slot)
    first = slot == 0
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    i = tl.arange(0, BK)
    j = tl.arange(0, BD)
    km = i < K
    dm = j < D
    nm = n < N
    mm = nm[:, None, None] & km[None, :, None] & dm[None, None, :]
    pm = nm[:, None, None] & km[None, :, None] & km[None, None, :]
    mo = n[:, None, None] * K * D + i[None, :, None] * D + j[None, None, :]
    po = n[:, None, None] * K * K + i[None, :, None] * K + i[None, None, :]
    ko = n[:, None] * K + i[None, :]
    vo = n[:, None] * D + j[None, :]
    after = tl.load(M + mo, mm, 0).to(tl.float32)
    if RULE == 0:
        pa = tl.load(P + po, pm, 0).to(tl.float32)
        # The covariance this write started from, exactly as the forward used it.
        p = tl.load(CovTape + po, pm, 0).to(tl.float32)
    k = tl.load(Key + ko, nm[:, None] & km[None, :], 0).to(tl.float32)
    error = tl.load(Error + vo, nm[:, None] & dm[None, :], 0).to(tl.float32)
    gain = tl.load(Gain + ko, nm[:, None] & km[None, :], 0)
    lam = tl.load(Lam + n, nm, 1.0)
    den = tl.load(Den + n, nm, 1.0)
    # Predecessor memory in its storage dtype; the first write starts from M = 0.
    # The plain rules wrote onto the decayed memory lambda * M, so the unwind divides by lambda.
    before = after - gain[:, :, None] * error[:, None, :]
    if RULE != 0:
        before = before / lam[:, None, None]
    before = before.to(M.dtype.element_ty).to(tl.float32)
    before = tl.where(first, 0.0, before)
    decayed = before
    if RULE != 0:
        decayed = before * lam[:, None, None]
    q = tl.load(Query + ko, nm[:, None] & km[None, :], 0).to(tl.float32)
    dy = tl.load(DY + vo, nm[:, None] & dm[None, :], 0).to(tl.float32)
    # The first write's memory output is recreated from the exact initial state rather than from
    # the unwound one, so the accumulated unwind error does not enter its derivatives.
    first_after = (gain[:, :, None] * error[:, None, :]).to(M.dtype.element_ty).to(tl.float32)
    after = tl.where(first, first_after, after)
    # M cotangent: the read's contribution is rounded to M's storage dtype and accumulated there,
    # as autograd would through the output cast of the write.
    read_grad = (q[:, :, None] * dy[:, None, :]).to(M.dtype.element_ty)
    h = tl.load(GM + mo, mm, 0).to(tl.float32) + read_grad.to(tl.float32)
    h = h.to(M.dtype.element_ty).to(tl.float32)
    dq = tl.sum(after * dy[:, None, :], 2)
    dv = tl.sum(h * gain[:, :, None], 1)
    dg = tl.sum(h * error[:, None, :], 2)
    if RULE == 0:
        jp = tl.load(GP + po, pm, 0).to(tl.float32)
        u = gain * den[:, None]
        ju = tl.sum(jp * u[:, None, :], 2)
        jtu = tl.sum(jp * u[:, :, None], 1)
        du = dg / den[:, None] - (ju + jtu) / (lam * den)[:, None]
        dd = -tl.sum(dg * gain, 1) / den + tl.sum(u * ju, 1) / (lam * den * den)
        du = du + dd[:, None] * k
        dk = (
            -tl.sum(before * dv[:, None, :], 2)
            + dd[:, None] * u
            + tl.sum(p * du[:, :, None], 1)
        )
        gp = jp / lam[:, None, None] + du[:, :, None] * k[:, None, :]
        dl = dd - tl.sum(tl.sum(jp * pa, 2), 1) / lam
        tl.store(GP + po, gp, pm)
    elif RULE == 1:
        dk = dg - tl.sum(decayed * dv[:, None, :], 2)
    else:
        dk = dg
    gm = h
    if RULE != 2:
        gm = h - k[:, :, None] * dv[:, None, :]
    if RULE != 0:
        # The cotangent reaches the undecayed memory through lambda; lambda's own gradient is
        # the overlap of that cotangent with the memory it scaled.
        dl = tl.sum(tl.sum(gm * before, 2), 1)
        gm = gm * lam[:, None, None]
    tl.store(GM + mo, gm, mm)
    tl.store(DK + ko, dk.to(DK.dtype.element_ty), nm[:, None] & km[None, :])
    tl.store(DQ + ko, dq.to(DQ.dtype.element_ty), nm[:, None] & km[None, :])
    tl.store(DV + vo, dv.to(DV.dtype.element_ty), nm[:, None] & dm[None, :])
    tl.store(DL + n, dl, nm)
    tl.store(M + mo, before, mm)
    if RULE == 0:
        tl.store(P + po, p, pm)
    if tl.program_id(0) == 0:
        tl.store(Cursor, slot)


def _sizes(memory):
    """Rows (token x group), key and value dims of the memory, and their power-of-two tile sizes."""
    k, d = memory.shape[-2:]
    return memory.numel() // (k * d), k, d, max(2, triton.next_power_of_2(k)), max(2, triton.next_power_of_2(d))


def _grid(meta_rows):
    return lambda meta: (triton.cdiv(meta_rows, meta["BN"]),)


def launch_forward(
    memory,
    covariance,
    cov_tape,
    key,
    value,
    query,
    forgetting,
    out,
    error,
    gain,
    den,
    lam_out,
    cursor,
    slot,
    rule=RLS,
):
    """Apply one write and read in place and fill its tape rows (cov_tape, error, gain, den, lam_out).

    ``cov_tape`` receives the covariance the write starts from (RLS only). ``slot`` is a [1] int32
    device tensor with this write's index; the kernel records ``slot + 1`` in ``cursor``.
    """
    n, k, d, bk, bd = _sizes(memory)
    _forward[_grid(n)](
        memory, covariance, cov_tape, key, value, query, forgetting, out, error, gain, den, lam_out,
        cursor, slot, n, K=k, D=d, BK=bk, BD=bd, RULE=rule,
    )


def launch_reverse(
    memory,
    covariance,
    cov_tape,
    grad_memory,
    grad_covariance,
    key,
    query,
    forgetting,
    error,
    gain,
    den,
    grad_out,
    cursor,
    slot,
    rule=RLS,
):
    """Unwind one write in place (M unwound, P restored from ``cov_tape``), accumulate the state
    gradients and return (dk, dv, dq, dlambda)."""
    dk, dv, dq, dl = (torch.empty_like(t) for t in (key, error, query, forgetting))
    n, k, d, bk, bd = _sizes(memory)
    _reverse[_grid(n)](
        memory, covariance, cov_tape, grad_memory, grad_covariance, key, query, forgetting, error,
        gain, den, grad_out, dk, dv, dq, dl, cursor, slot, n, K=k, D=d, BK=bk, BD=bd, RULE=rule,
    )
    return dk, dv, dq, dl


# Standalone ops; the model path goes through state.py, which owns the tape.
@torch.library.custom_op("ldm_flexible::write_read", mutates_args=("memory", "covariance"))
def write_read(
    memory: torch.Tensor,
    covariance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    forgetting: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One write and read; returns (out, error, gain, den, cov) with ``cov`` the covariance the
    write started from."""
    out, error = torch.empty_like(value), torch.empty_like(value)
    gain = torch.empty(key.shape, device=key.device, dtype=torch.float32)
    den = torch.empty_like(forgetting)
    cov = torch.empty_like(covariance)
    cursor = torch.zeros(1, device=memory.device, dtype=torch.int32)
    slot = torch.zeros(1, device=memory.device, dtype=torch.int32)
    launch_forward(memory, covariance, cov, key, value, query, forgetting, out, error, gain, den,
                   torch.empty_like(forgetting), cursor, slot)
    return out, error, gain, den, cov


@write_read.register_fake
def _fake_write_read(memory, covariance, key, value, query, forgetting):
    return (
        torch.empty_like(value),
        torch.empty_like(value),
        torch.empty(key.shape, device=key.device, dtype=torch.float32),
        torch.empty_like(forgetting),
        torch.empty_like(covariance),
    )


@torch.library.custom_op(
    "ldm_flexible::reverse", mutates_args=("memory", "covariance", "grad_memory", "grad_covariance")
)
def reverse(
    memory: torch.Tensor,
    covariance: torch.Tensor,
    cov: torch.Tensor,
    grad_memory: torch.Tensor,
    grad_covariance: torch.Tensor,
    key: torch.Tensor,
    query: torch.Tensor,
    forgetting: torch.Tensor,
    error: torch.Tensor,
    gain: torch.Tensor,
    den: torch.Tensor,
    grad_out: torch.Tensor,
    first: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    slot = torch.full((1,), 0 if first else 1, device=memory.device, dtype=torch.int32)
    cursor = slot + 1
    return launch_reverse(memory, covariance, cov, grad_memory, grad_covariance, key, query,
                          forgetting, error, gain, den, grad_out, cursor, slot)


@reverse.register_fake
def _fake_reverse(
    memory,
    covariance,
    cov,
    grad_memory,
    grad_covariance,
    key,
    query,
    forgetting,
    error,
    gain,
    den,
    grad_out,
    first,
):
    return tuple(torch.empty_like(t) for t in (key, error, query, forgetting))
