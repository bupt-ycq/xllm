# Parallel Chunk KDA: why chunk 16 can be more accurate than chunk 64

Date: 2026-09-01

## Bottom line

For a **parallel** KDA/WY implementation, chunk 16 can be more accurate than a
naive chunk 64 implementation for a precise mathematical reason: with the
bounded log gate $g\in[-5,0]$, a 16-token local prefix has

\[
s_r=\sum_{i=1}^{r}g_i\in[-80,0],
\]

so both factors used by the chunk algorithm,
$\exp(s_r)$ and $\exp(-s_r)$, remain in the BF16/FP32 exponent range.  At
chunk 64 the worst interval is $[-320,0]$; the same direct factorization can
produce zero on one side and infinity on the other.  Moonshot gives this as
the first reason FlashKDA v1 fixes its internal chunk at 16.
[FlashKDA v1 deep dive](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/docs/20260420-flashkda-v1-deep-dive.md#L11-L43)

That does **not** prove that every chunk-16 implementation is more accurate
than every chunk-64 implementation.  Current FLA's chunk-64 kernel already
uses 16-token secondary blocks, FP32 diagonal blocks, anchored exponent
differences, and a blockwise inverse merge to control the range.  Chunk 16
also creates four times as many cross-chunk state updates over the same prompt,
which can add rounding error.  Therefore the implementer's observation is
plausible, but identifying its cause requires comparing the actual MLU stages,
not the two constants alone.

## Three different meanings of “16”

1. **FlashKDA v1 internal compute chunk = 16.**  It is a true parallel KDA:
   K1 prepares every 16-token tile in parallel and K2 advances the recurrent
   state tile by tile.  The C++ entry point fixes `CHUNK = 16`.
   [FlashKDA entry point](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/flash_kda.cpp#L255-L292)
2. **FLA/vLLM Triton compute chunk = 64 by default.**  The public FLA KDA API
   defaults to 64 and currently accepts 32 or 64; internally, however, its
   chunk-64 solve is split into four 16-token subchunks.
   [FLA public API](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk.py#L393-L401)
   [FLA secondary chunking](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk_intra.py#L801-L864)
   Current upstream vLLM can also select the FlashKDA backend when its hardware,
   dtype, shape, and safe-gate constraints are satisfied; otherwise it uses
   the vendored Triton path.  Thus “vLLM community always uses 64” is no longer
   accurate.
   [vLLM backend selection](https://github.com/vllm-project/vllm/blob/55aa766dc8c1a7d739ef90c5580ca4b91050b35f/vllm/models/kimi_k3/nvidia/kda.py#L300-L317)
   [vLLM prefill dispatch](https://github.com/vllm-project/vllm/blob/55aa766dc8c1a7d739ef90c5580ca4b91050b35f/vllm/models/kimi_k3/nvidia/kda.py#L821-L971)
3. **vLLM `--block-size 16` is cache/scheduler granularity.**  It says how
   many token positions form a cache block; it does not select the internal
   KDA algebraic chunk.  vLLM defines the default cache block size separately
   in `CacheConfig`.
   [vLLM cache configuration](https://github.com/vllm-project/vllm/blob/55aa766dc8c1a7d739ef90c5580ca4b91050b35f/vllm/config/cache.py#L117-L151)

The previously inspected MR's pure-Torch, token-by-token recurrence is a
fourth case: it does not construct a 16-by-16 or 64-by-64 parallel WY system,
so the range and triangular-solve argument below does not explain that
fallback.  This report instead analyzes the non-serial, FlashKDA-like path
asked about here.

## 1. Exact arithmetic: 16 and 64 compute the same KDA

For one head, KDA updates the matrix state as

\[
S_t=(I-\beta_t k_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}
       +\beta_tk_tv_t^\top,
\qquad 0<\alpha_t\le 1.
\]

The Kimi Linear paper derives a chunkwise WY/UT form of exactly this
recurrence.  It defines the cumulative decay $\gamma$, forms a unit
lower-triangular system, and then advances the state once per chunk.  Changing
$C$ only changes parenthesization; in real arithmetic, $C=16$, $C=64$,
and tokenwise recurrence agree.
[KDA recurrence and chunk derivation](https://arxiv.org/html/2510.26692#S3.SS1)

Therefore any observed accuracy difference is a **finite-precision and
implementation** effect, not a different model formula.

## 2. The strongest mechanism: cumulative gate range

Let the per-channel log gate inside a chunk be $g_i=\log\alpha_i\le0$, and

\[
s_r=\sum_{i=1}^{r}g_i,\qquad \Gamma_r=\exp(s_r).
\]

The paper's UT form contains both

\[
X=\Gamma\odot K,\qquad Y=K/\Gamma,
\]

whose product generates the causal decay ratios.  Mathematically the large
and small factors cancel:

\[
X_iY_j^\top
=\sum_d k_{i,d}k_{j,d}\exp(s_{i,d}-s_{j,d}).
\]

In floating point, computing (X) and (Y) separately first can lose that
cancellation.  The Kimi paper explicitly identifies division by fine-grained
decay as a numerical issue and says prior implementations use log-domain and
secondary full-precision chunks to control it.
[Kimi Linear efficiency analysis](https://arxiv.org/html/2510.26692#S3.SS2)

For FlashKDA's bounded gate, $g_i\in[-5,0]$.  BF16 has the same 8-bit
exponent field and finite exponent range as FP32.  The relevant values are:

| compute chunk | worst $s_C$ | $\exp(s_C)$ | $\exp(-s_C)$ | direct-factor status |
|---:|---:|---:|---:|---|
| 16 | -80 | $1.80\times10^{-35}$ | $5.54\times10^{34}$ | finite and normal |
| 64 | -320 | $1.06\times10^{-139}$ | $9.42\times10^{138}$ | underflow / overflow |

For reference, the BF16/FP32 largest finite magnitude is about
$3.39\times10^{38}$, and the smallest positive normal is about
$1.18\times10^{-38}$.  FlashKDA uses a flush-to-zero FP32 base-2 exponential,
so normal range is the relevant lower boundary.  Direct overflow/downflow can
already occur once $|s|$ is around 88, far before the worst chunk-64 value.
[FlashKDA exponent helper](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/smxx/utils.cuh#L38-L42)

The implementation matches the algebra: it accumulates the activated gate in
FP32, then materializes `exp(cumsum)`, `exp(-cumsum)`, and the restored-key
factor in BF16.  With (C=16), the ([-80,80]) guarantee keeps those direct
factors finite.
[FP32 gate cumsum](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/smxx/fwd_kernel1.cuh#L306-L330)
[decay factors](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/smxx/fwd_kernel1.cuh#L423-L471)

This is a **proved range guarantee**, conditional on all of the following:

- the gate is the bounded/safe form with lower bound no smaller than -5;
- the implementation directly materializes both positive and reciprocal
  prefix-decay factors;
- the exponential/destination formats have BF16/FP32 exponent range.

It is not a guarantee for the unbounded gate
$-\exp(A)\operatorname{softplus}(x)$, where a single token can be below -5.

### Why current FLA chunk 64 does not simply overflow

Current FLA avoids the naive direct-(C=64) calculation.  It keeps the gate
cumsum in FP32, divides the 64-token system into 16-token blocks, and forms
off-diagonal terms with exponent differences anchored at a subchunk boundary;
the displayed exponent arguments are non-positive decay distances rather than
an unbounded reciprocal prefix.  It separately stores the 16-by-16 diagonal
blocks in FP32.
[FLA fused gate cumsum](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/gate.py#L407-L509)
[anchored off-diagonal factors](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk_intra.py#L132-L218)
[FP32 diagonal buffer](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk_intra.py#L801-L864)

Thus, if an MLU chunk-64 port simply enlarges a direct FlashKDA-like chunk-16
formula without porting FLA's rescaling/subchunk machinery, degraded precision
is expected.  If it faithfully implements those safeguards, the exponent-range
argument alone is insufficient to predict that 16 will win.

## 3. Gate-cumsum rounding and exponent sensitivity

Suppose the computed prefix is $\widehat{s}_r=s_r+\varepsilon_r$.  Then

\[
\frac{\widehat\Gamma_r-\Gamma_r}{\Gamma_r}
=\exp(\varepsilon_r)-1\approx\varepsilon_r.
\]

Longer local scans contain more additions, so their worst-case accumulation
error is larger.  Because every $g_i\le0$, there is no beneficial sign
cancellation.  Resetting the local prefix every 16 tokens bounds both the
number of additions and the exponent interval.  In practice, however, both
FlashKDA and FLA calculate the gate activation/cumsum in FP32, so ordinary
FP32 addition error is usually secondary to exponent underflow and BF16
materialization.

This mechanism is mathematically sound, but attributing a measured model-level
gap to it rather than to the range failure requires per-stage measurements.

## 4. The triangular solve is shorter and usually easier at 16

The UT transform solves a unit lower-triangular system

\[
T=I+L,\qquad M=T^{-1}\operatorname{Diag}(\beta).
\]

Since (L) is strictly lower triangular, (L^C=0), hence

\[
T^{-1}=I-L+L^2-\cdots+(-L)^{C-1}.
\]

At 16 the dependency chain has at most 15 off-diagonal steps; at 64 it has at
most 63.  A perturbation $\Delta L$ from rounded decay factors and GEMMs has
the first-order effect

\[
\Delta(T^{-1})\approx -T^{-1}(\Delta L)T^{-1}.
\]

Therefore a poorly conditioned (T) can amplify input error twice.  Larger
(C) gives a longer substitution chain and admits higher-degree products of
(L), so it has more opportunity to accumulate or amplify rounding.  This is
especially relevant for nearly collinear keys and $\beta$ near one.  The
FlashKDA source explicitly notes that its earlier low-precision inverse lost
accuracy when $|L|$ approached one for near-collinear keys, and now uses FP32
forward substitution for the diagonal 8-by-8 blocks plus BF16-HMMA block
merges with FP32 accumulation.
[FlashKDA inverse implementation](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/smxx/utils.cuh#L194-L211)

This is a **risk argument, not a monotonic theorem**: a particular 64-by-64
unit-triangular matrix can be better conditioned than a particular 16-by-16
one.  The relevant proof for a failing case is the measured condition number
or inverse residual $\lVert I-T\widehat T^{-1}\rVert$.

FLA again mitigates the risk: it solves FP32 16-by-16 diagonal blocks and then
merges them with TF32 or IEEE dot products before writing the inverse in the
input dtype.
[FLA solve precision and merge](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk_intra.py#L20-L23)
[FLA block inverse merge](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk_intra.py#L265-L379)

## 5. Low-precision GEMMs and reassociation

Chunking replaces tokenwise updates with dense products such as (MV),
(W S), and the intra-chunk output product.  For a conventional rounded dot
product of length (n), the accumulation factor is commonly summarized as

\[
\gamma_n=\frac{nu}{1-nu}.
\]

Holding the arithmetic format fixed, increasing the relevant inner dimension
from 16 to 64 makes this bound roughly four times larger.  Chunk 64 also
creates 16 times as many entries in each $C\times C$ intermediate as chunk
16, increasing the number of places where an extreme case can occur.

Actual Tensor Core error is implementation-specific because both kernels use
BF16 operands with FP32 accumulation and insert BF16 casts at different
boundaries.  FlashKDA stores decayed Q/K, the inverse, and the intra-chunk
matrix as BF16 workspaces, while retaining selected accumulations in FP32.
[FlashKDA workspace dtypes](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/smxx/fwd_kernel1.cuh#L44-L82)
[FlashKDA matrix construction](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/smxx/fwd_kernel1.cuh#L478-L511)

Therefore “shorter GEMMs reduce rounding” is a reasonable contributor, but it
is a **hypothesis** until both candidate kernels' exact casts and MMA modes are
matched.

## 6. Counter-effect: chunk 16 has more state boundaries

For a fixed prompt length (L), chunk 16 performs four times as many
cross-chunk updates as chunk 64.  If

\[
\widehat S_{m+1}=P_m\widehat S_m+B_m+\delta_m,
\]

then the accumulated error is

\[
e_M=\sum_{m=0}^{M-1}P_{M-1}\cdots P_{m+1}\delta_m.
\]

More chunks mean more local rounding terms $\delta_m$.  In exact KDA with
L2-normalized keys, $0\le\beta\le1$, and $0<\alpha\le1$, the one-token
transition is non-expansive in the 2-norm: both
$I-\beta kk^\top$ and $\operatorname{Diag}(\alpha)$ have operator norm at
most one.  Gates therefore tend to damp old error rather than amplify it, but
they do not eliminate the extra rounding sites.

This counter-effect is concrete in FlashKDA v1: its on-chip recurrent state is
BF16, and the state is rounded back to BF16 after each 16-token update even
when a public FP32 initial/final buffer is requested.  Moonshot reports no
measurable benchmark loss when the update uses FP32 FMA, but that is an
empirical statement, not an algebraic guarantee.
[FlashKDA precision notes](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/docs/20260420-flashkda-v1-deep-dive.md#L45-L68)
[FlashKDA BF16 state update](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/smxx/fwd_kernel2.cuh#L659-L724)

FLA's state kernel instead holds the running state in FP32 registers across
its chunk loop and casts only stored intermediate checkpoints to the K/V dtype;
the final state is allocated FP32.  This difference can favor FLA chunk 64
even when FlashKDA's shorter local solve favors chunk 16.
[FLA running-state recurrence](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/common/chunk_delta_h.py#L140-L184)
[FLA state update and FP32 final state](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/common/chunk_delta_h.py#L244-L324)

## 7. What is proved, and what remains a hypothesis

| Claim | Status |
|---|---|
| Chunk 16 and 64 are algebraically equivalent | Proved by the recurrent/chunk derivation in exact arithmetic |
| With safe gate ([-5,0]), direct 16-token positive/reciprocal decay factors fit BF16/FP32 exponent range | Proved by the bounds above and confirmed as FlashKDA's design rationale |
| The same direct factorization is unsafe at chunk 64 | Proved; the worst-case factors exceed the format range |
| Current FLA chunk 64 is a naive direct 64-token factorization | False; source shows 16-token secondary blocks and FP32/blockwise safeguards |
| A shorter triangular solve has fewer rounding dependencies | Proved structurally |
| The observed MLU accuracy gain is primarily caused by triangular conditioning | Hypothesis until inverse residuals/condition numbers are measured |
| Chunk 16 is always more accurate | Not proved and generally false as a universal statement; extra state boundaries can reverse the result |
| “Fused gate” itself explains the gain | Not proved; fusion matters only if it changes dtype, cast, approximation, or intermediate storage |

## 8. Minimal experiment that can identify the cause

Use the same Q/K/V/raw-gate/beta/state tensors and compare a FP64 or FP32
tokenwise recurrent oracle against parallel $C\in\{16,32,64\}$.  For each
case, record errors after each stage, not only final logits:

1. activated gate and local FP32 cumsum;
2. `exp(cumsum)`, reciprocal/restored decay factors, and counts of zero/Inf/NaN;
3. (L), (T^{-1}), its condition number, and
   $\lVert I-T\widehat T^{-1}\rVert$;
4. (W/U), per-chunk output, and final recurrent state;
5. state held in FP32 versus BF16 between chunks;
6. chunk-64 with and without FLA-style 16-token secondary rescaling.

The decisive gate sweep is a constant per-channel log gate near
$-0.5,-1.0,-1.4,-2.5,-5.0$, plus the model's real gate distribution.
Around average magnitude 1.4, a direct 64-token reciprocal prefix is already
near the FP32/BF16 range boundary, while chunk 16 remains safely inside it.
Separately use nearly collinear normalized keys and $\beta$ near one to
stress the triangular solve.

If zeros/Infs first appear in the chunk-64 decay factors, the range mechanism
is confirmed.  If decay factors remain finite but the inverse residual jumps,
conditioning/solve precision is the cause.  If both stages match and drift
grows only at chunk boundaries, state dtype and cross-chunk propagation are
the cause.

## Assessment for the reported result

The most likely mathematical explanation for “chunk 16 tested best” is that
the MLU parallel implementation uses the direct cumulative-decay
factorization (or an incomplete version of FLA's rescaling).  Under the
bounded $-5$ gate, chunk 16 is the largest common power-of-two choice that
gives a clean normal-range guarantee:
$5\times16=80<87.3$.  Chunk 64 requires
secondary scaling/subchunking and a more careful block solve.

But this should be reported as a likely diagnosis, not as proof that the MLU
kernel is “more accurate than community FLA.”  The community chunk-64 kernel
contains safeguards that a fair comparison must preserve, and FlashKDA v1
also makes different state-dtype and GEMM-association choices.  The stagewise
experiment above is the shortest path from the observed final error to a
verified cause.
