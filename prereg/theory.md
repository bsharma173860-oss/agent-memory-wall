# Theory: why the numbers are what they are (KV-cache energy study)

Private until the paper is out. Companion to `2026-11-kv-energy-plan.md`. Each result below is derived from first
principles, then checked against a measurement; where the measurement is still running, the prediction is written
here first. "Energy" means GPU board energy from NVML unless stated.

Model constants (Qwen3-32B, BF16): 64 layers, hidden 5120, 64 query heads and 8 KV heads of 128, MLP 25600.
Card: H200 NVL, 989 TFLOP/s dense BF16 peak, PCIe Gen5 x16 host link.

## 1. Energy of re-reading a prompt (recompute)

Prefill is compute-bound, so energy per token = FLOPs per token x joules per FLOP.

- Linear layers: 2 x 31.2B weights = **62.4 GFLOP per token**, independent of context.
- Attention: QK^T and AV cost 2 x 2 x 64 heads x 128 x 64 layers = **2.10 MFLOP per earlier token**. Causal prefill of
  L tokens averages L/2 earlier tokens, so

      F(L) = 62.4 GFLOP + 2.10 MFLOP x L/2          (64.5 GFLOP at 2k, 72.9 at 10k, 79.2 at 16k)

- Measured under thrashing: 7,858 prompt tokens/s at ~590 W with ~12k-token contexts = 0.589 PFLOP/s, **60% of
  peak**, so **1.00 pJ per FLOP gross (0.85 above idle)**. Hence

      e_r(L) = F(L) x 1.0 pJ = 0.065 J/token at 2k ... 0.079 J/token at 16k

- Checks: (a) the direct estimate 590 W / 7,858 tok/s = 0.075 J/token; (b) E2's marginal re-read cost, about 0.065 at
  2k, 0.08 at 10k, 0.085 at 16k. Theory predicts the 16k/2k ratio at 1.23; measured about 1.31. The extra is expected:
  attention kernels run below the GEMM efficiency assumed here. Cross-check: Librarian (2605.27787) fits 0.034 J/token
  for a 27B model in FP8 on B200, consistent with fewer FLOPs per token, FP8 and a newer card.
- **Consequence:** recompute gets dearer per token as the context grows, so the case for keeping or moving KV gets
  stronger with length (answers llm-d RQ2, "the crossover shifts with prefix length").

## 2. Energy of restoring a prompt from host memory (transfer)

KV per token = 2 (K,V) x 64 layers x 8 heads x 128 x 2 bytes = **262,144 bytes**. Over PCIe Gen5 at ~50 GB/s that is
5.2 microseconds per token. At 100-250 W of GPU draw during the copy:

      e_h = P x bytes / bandwidth = 0.5 to 1.3 mJ per token

**Prediction for H10:** restore costs 1/50 to 1/150 of recompute per token (the pre-registered bound, 1/10, is
deliberately loose). A GPU hit costs only the scheduling and the one output token, near zero per prompt token.
Host CPU and DRAM add little: HotCarbon measured the host difference under 1% of the GPU saving.

**The llm-d decision rule (2609.05565, eq. 8)** for one host on one card becomes

      move (restore)  iff  e_h x U + holding cost  <  e_r(L) x U,   with e_r / e_h ~ 50-150

so restoring wins whenever the KV is still held. The real question is holding it: section 4.

**Measured (H10, 6 Nov):** restore 0.3-0.7 mJ per token above idle (inside or just below the predicted 0.5-1.3),
recompute 56-73 mJ: restore is ~110x cheaper at ~5k tokens, ~160x at ~10k, ~220x at ~20k (measured prompt lengths
4,962 / 10,066 / 20,336; the test labels said 4k/8k/16k). The ratio grows with length because
recompute grows with L (attention) while restore stays per-byte. GPU hit: ~0.

## 3. Why thrashing costs exactly what no cache costs

Under LRU, a session's KV survives only if the KV touched since its last turn fits in capacity C. When sessions
return in near-cyclic order and N x s > C, every session is evicted just before it returns: hits -> 0 (classic loop
result, LIRS 2002). With zero hits the uncached tokens U are identical to running with no cache, and turn energy is

      E_turn = e_r x U + E_decode + P_idle x t

so E_thrash = E_nocache. Measured: 8 overload cells on two engines, within 0.05-2% (SGLang 0.01-0.15%, identical
turn counts). The small vLLM gap (+0.6 to +1.7% for no cache) is the residual 0.04% of hits plus run noise.

## 4. Where the cliff is

A session hits if the distinct other sessions touched during its gap, times s, plus its own s, fit in C.

- Fixed cycle: all N-1 others are touched, so the cliff is at **N* = C/s**.
- Random order, gap of one mean cycle: each other session is touched with probability 1 - 1/e = 0.63, so
  **N* = 1 + C/(0.63 s)**, a factor of 1.58.
- Published: Metastability (2611.02543) fits N_c ~ 1.9 C/c in disaggregated SGLang; the robot-fleet test (2611.01404)
  shows a factor near 1 for strict cycles.

With C = 244,928 tokens and s about 10.5k: fixed cycle 23, random 38, published 44. H8 measures 24/32/40 to tell
them apart. This is the colocated case Metastability sec. 9 says is not modeled.

**Offload tiers.** With an inclusive host tier (write_through: the host also holds what is on the GPU), usable
capacity is max(C_gpu, C_host), not the sum; with an exclusive tier it is the sum. SGLang at 80 GB: C_host = 305,176,
so N* moves only from about 23-38 to 29-47; 48 sessions sit at the edge, hence 12.6% hits. At 160 GB (~610k tokens):
N* = 58-93, so 48 sessions should fit (H9) and 96 should not.

Energy follows from hits with one parameter: E = E_thrash x (1 - f x dh), f the prefill share of turn energy under
thrashing and dh the change in hit rate (restore is ~0, sec. 2). Fitted on the 160 GB point (dh 0.754, 449.6 J):
f = 0.76, i.e. ~253 J of the 1,058 J turn is decode. It then predicts SGLang 80 GB 957 J (measured 953), vLLM 70 GiB
1,033 J (1,035) and 973 J (973): within 0.5%. It fails at SGLang 160 GB, 48 sessions, 30 s pauses: 534 J predicted vs
614 J measured (-12.9%); every other offload cell is within 0.4% (analysis/analysis.md). (A first guess f = 0.87 gave 942 vs 953 and 364 vs 450; withdrawn.)

**Measured cliff (H8):** collapse near W/C 1.2 with 5 s pauses and 1.9-2.2 with 30 s pauses: neither the fixed-cycle
(1) nor the random (1.58) model alone; the factor grows as returns become less cyclic.

## 5. Admission gate (K slots, yield after q turns)

With K x s <= C, a session's first turn after admission is cold and the next q-1 are warm:

      E(q) = [E_cold + (q - 1) E_warm] / q

With E_cold ~ 1,030 J and E_warm ~ 400 J (measured regimes): q=4 gives ~560 J, q=8 ~480 J, against ~1,047 J
thrashing. Waiting cost by Little's law: a session waits about (N/K - 1) x q x T_turn for its next slot, so energy
falls as 1/q while waiting grows as q: a Pareto curve, not a free lunch. H6 measures q = 4 and 8.

## 6. Batching (part 1)

Decode is memory-bound: power is nearly flat while tokens per second grow with the batch b, so energy per token
falls as ~1/b until compute saturates. Measured: 8.73 J/token at 1 user to 0.22 at 64 (40x), with power rising only
~20%.

## Questions a reviewer or reader will ask, and where the answer is

| Question | Answer | Status |
|---|---|---|
| Is the cliff just LRU? | Yes, the loop result (sec. 3); we cite it and measure its energy | done |
| Is it engine-specific? | Same in vLLM and SGLang, 0.05-2% | done |
| Why is thrash energy equal to no cache? | Zero hits means identical U (sec. 3) | done |
| Where is the cliff for my server? | N* = C/s to 1.6 C/s (sec. 4) | H8 running |
| Is restoring greener than recomputing? By how much? | e_r/e_h ~ 50-150 predicted (sec. 2) | H10 queued |
| Does the crossover depend on length? | Yes, e_r grows with L (sec. 1) | E2 done; H10 adds |
| Does offload save energy? When? | Only when the tier holds the working set; inclusive tier = max, not sum (sec. 4) | 80 GB done; H9 queued |
| What does offload cost on the host? | Bounded: engine CPU time + bytes; HotCarbon measured <1% | bound; bare metal later |
| Do LFU or SLRU fix it? | Prediction H7 | queued |
| Does admission fix it, and at what latency? | E(q) and Little's law (sec. 5) | H6 queued |
| GPU energy only? | Yes, labeled; agents' host share is 41-45% (2609.29707); RAPL on bare metal later | open |
| Synthetic sessions? | Yes; real agent traces are phase 2 | open |
| One model, one card? | Yes; other models and AMD MI300X are phase 3 | open |
| Repeatable? | 2 rounds, alternating order: SGLang no cache repeats within 0.06% in every cell, same turn counts | done (rest running) |
| Is that just the same random sessions twice? | Rounds share seeds, so they show measurement noise (~0.1%); two new seeds (s1, s2) rerun vLLM cache vs no cache at 16 and 48 sessions | E3g queued |
| Is the cached-token count trustworthy? | Engine-reported cached tokens agree with TTFT (hits 0.2-0.3 s, misses tens of seconds) in every cell | done |
| Is the energy reading trustworthy? | NVML energy counter, checked against AMD's own tool at 0.11% on MI300X; idle measured after a 1 °C settle | done |
