import random
import numba as nb
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import math
from scipy.stats import nbinom, gamma  # negative‑binomial
from types import MethodType
import time
import os, json, hashlib

# --- faster fast_tx_core with prange --------------------------------
# at top of population.py
TEST_PROB_SYMP = 0.10  # legacy daily P(test) once symptomatic; used only if test_delay_mode='daily_prob'
LAB_TURNAROUND_DAYS = 2  # default lab/result turnaround after specimen collection
ISOLATION_DAYS = 21

# CDC-aligned traced-contact management:
# asymptomatic close contacts enter 21-day symptom monitoring and risk reduction,
# not mandatory full quarantine. The legacy QUARANTINE_DAYS name is kept as an
# alias for backward compatibility.
MONITORING_DAYS = 21
MONITOR_SEXUAL_CONTACT_MULT = 0.30  # retain 30% of sexual/high-contact event activity
MONITOR_SOCIAL_CONTACT_MULT = 0.70  # retain 70% of routine social/household close activity
QUARANTINE_DAYS = MONITORING_DAYS
TRACE_PROB_EDGE = 0.10

# ============================================================================
# CDC MMWR reference-model activity stratification (`riskg`, 6 levels).
# Source: Spicknall IH et al. "Modeling the Impact of Sexual Networks in the
# Transmission of Monkeypox Virus Among Gay, Bisexual, and Other MSM —
# United States, 2022." MMWR 2022;71(35):1133-1137. The 6 strata follow
# Weiss RE et al. (2020) for STI sexual-network activity groups.
#
#   CDC_RISKG_POP_WEIGHTS : population share of each stratum (5x19% + 1x5%)
#   CDC_RISKG_ONETIME_7D  : one-time partner formation rate per 7 days, by
#                           stratum (level 6 = 2/7 ≈ 0.286/day ≈ 100/yr)
#   CDC_RISKG_PARTNERS_3W : mean number of partners in the past 3 weeks
#   CDC_RISKG_PARTNERS_1Y : mean number of partners in the past year
#
# These are LITERATURE-FIXED anchors used by both the one-off layer and the
# sex_party heavy-tail. Do not tune them during calibration.
# ============================================================================
CDC_RISKG_POP_WEIGHTS = np.array([0.19, 0.19, 0.19, 0.19, 0.19, 0.05], dtype=np.float64)
CDC_RISKG_ONETIME_7D = np.array([0.000, 0.007, 0.038, 0.071, 0.221, 2.000], dtype=np.float64)
CDC_RISKG_PARTNERS_3W = np.array([0.8, 0.8, 0.9, 1.0, 1.5, 6.6], dtype=np.float64)
CDC_RISKG_PARTNERS_1Y = np.array([1.8, 1.8, 4.0, 4.0, 14.7, 124.7], dtype=np.float64)

# ============================================================================
# sex_party partner-degree E[k] by CDC riskg stratum.
#
# The CDC MMWR partner counts are *cumulative* (past-3-weeks / past-1-year),
# so they cannot be mapped 1:1 onto a single party night. We instead use the
# RELATIVE gradient of the past-3-weeks partner counts (short-term partner
# turnover is the correct analogue for a single party night) and anchor the
# top stratum to `party_ht_top_mean` (calibrated to 20 in the 10M multi-seed
# sweep). With top=20 this yields E[k] = [2.42, 2.42, 2.73, 3.03, 4.55, 20.0],
# i.e. the bulk (~2.4) matches the previous flat NB mean (~2.7) while the top
# 5% retains the heavy tail.
#
# The 1-year gradient is NOT used: it is diluted by long-term monogamous
# partnerships (bottom strata -> 0.29 partners), which is not the right
# quantity for a single-night partner draw.
# ============================================================================
def _cpc_sexparty_ek(top_mean):
    """Return 6-level E[k] scaled from CDC past-3-week partner ratios."""
    rel = CDC_RISKG_PARTNERS_3W / CDC_RISKG_PARTNERS_3W[-1]
    return float(top_mean) * rel


# --- imperfect test performance (toy defaults; tweak as needed) --------
TEST_SENS = 0.95  # P(test+ | truly infected)  #Evaluation of a Zoonotic Orthopoxvirus PCR Assay for the Detection of Mpox Virus Infection doi: 10.1016/j.jmoldx.2023.06.010
TEST_SPEC = 0.999  # P(test- | truly uninfected) #Evaluation of a Zoonotic Orthopoxvirus PCR Assay for the Detection of Mpox Virus Infection doi: 10.1016/j.jmoldx.2023.06.010

Case_data = [0] * 18 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 1, 2, 1, 4, 0, 1, 6, 3, 1, 2,
                        4, 1, 1, 1, 2, 3, 2, 2, 6, 4, 9, 3, 1, 7, 4, 7, 6, 6, 8, 11, 10, 11, 12, 11, 29, 11, 17,
                        17, 20, 20, 20, 30, 28, 29, 24, 29, 25, 20, 39, 28, 35, 26, 30, 43, 26, 58, 31, 41, 36,
                        36, 36, 36, 42, 37, 48, 25, 37, 36, 35, 53, 32, 29, 30, 33, 41, 31, 33, 25, 16, 30, 30,
                        23, 23, 25, 23, 22, 18, 19, 19, 18, 18, 15, 19, 26, 15, 15, 10, 20, 16, 9, 17, 16, 12,
                        6, 11, 10, 9, 8, 8, 2, 6, 8, 10, 7, 3, 5, 4, 4, 6, 4, 7, 5, 7, 2, 4, 3, 3, 6, 2, 3, 6,
                        1, 7, 2, 3, 4, 3, 3, 3, 2, 7, 1, 3, 1, 0, 4, 3, 1, 1, 1, 1, 1, 0, 6, 0, 1, 1, 0, 0, 1,
                        1, 2, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 4, 3, 0, 3, 2, 2, 0, 2, 0, 1, 0, 2, 0, 0, 2,
                        1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                        0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                        0, 0, 1, 0, 0, 2, 0, 1, 0, 3, 0, 4, 0, 2, 1, 0, 1, 0, 0, 0, 2, 2, 2, 1, 1, 0, 0, 1, 0,
                        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0,
                        0, 0]
# Initialize a list to store the probabilities of each edge
probability_log = []


# ============================================================================
#  Mpox/STI-focused tuning hooks (added for transparency & resubmission)
#  - sexual_active_frac: fraction of static MCP edges active per day (default 0.10)
#  - sexual_edge_mult : extra thinning multiplier for daily sexual edges (default 1.0)
#  - oneoff_active_frac: daily active fraction among MCP pool for one-off edges (default 0.01)
#  - sex_party_attend_prob: daily attendance prob among eligible (default 3/7)
#  - sex_party_k_p: negbinom p controlling partner slots in party (default 0.33)
#  - sex_party_k_clip: cap of partner slots per attendee (default 10)
#  - sex_party_touch_frac: fraction of party sexual pairs expanded to close-touch edges (default 0.10)
#  These are intentionally simple multipliers so you can calibrate edge volumes
#  (reduce sexual edges, increase party edges) without rewriting network logic.
# ============================================================================


def _hash_params(d: dict) -> str:
    """Stable short hash for dict params."""
    s = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(s, digest_size=16).hexdigest()


def _make_cache_key(n: int, seed: int, params: dict) -> str:
    return _hash_params({"n": int(n), "seed": int(seed), **params})


def save_npz(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


@nb.njit(cache=True)
def _cum_offsets(k_vec):
    """Return cumulative‑sum offsets as **int32**.

    Examples
    --------
    >>> _cum_offsets(np.array([2,3,1]))
    array([0,2,5,6], dtype=int32)
    """
    n = k_vec.size
    off = np.empty(n + 1, np.int32)
    total = 0
    for i in range(n):  # explicit loop avoids np.cumsum(out=...)
        off[i] = total
        total += int(k_vec[i])
    off[n] = total
    return off


@nb.njit(nogil=True, fastmath=True, parallel=True)
def fast_tx_core(src: np.ndarray, dst: np.ndarray, p_uv: np.ndarray, p_vu: np.ndarray,
                 infectious: np.ndarray, susceptible: np.ndarray,
                 dead: np.ndarray, rel_sus: np.ndarray,
                 inactive_msk: np.ndarray, roll_uv: np.ndarray, roll_vu: np.ndarray
                 ) -> np.ndarray:
    """Two‑way transmission; returns unique newly infected IDs."""

    new_inf = np.zeros(infectious.size, dtype=nb.boolean)

    for i in nb.prange(src.size):
        u = src[i];
        v = dst[i]

        # 任一方不活跃（隔离/检疫）直接跳过
        if inactive_msk[u] or inactive_msk[v]:
            continue

        # u → v
        if infectious[u] and susceptible[v] and (not dead[v]):
            if roll_uv[i] < p_uv[i] * rel_sus[v]:
                new_inf[v] = True

        # v → u（同一个 roll）
        if infectious[v] and susceptible[u] and (not dead[u]):
            if roll_vu[i] < p_vu[i] * rel_sus[u]:
                new_inf[u] = True

    return np.where(new_inf)[0].astype(np.int64)


@nb.njit(nogil=True, fastmath=True, parallel=True, cache=True)
def fast_tx_flags(src: np.ndarray, dst: np.ndarray,
                  p_uv: np.ndarray, p_vu: np.ndarray,
                  infectious: np.ndarray, susceptible: np.ndarray,
                  dead: np.ndarray, rel_sus: np.ndarray,
                  inactive_msk: np.ndarray, roll_uv: np.ndarray, roll_vu: np.ndarray):
    """
    对每一条边返回两个布尔标志：u->v 是否成功，v->u 是否成功。
    注意：不在内核里做“按人去重”与 O(n) 全量扫描。
    """
    m = src.size
    succ_uv = np.zeros(m, dtype=nb.boolean)
    succ_vu = np.zeros(m, dtype=nb.boolean)

    for i in nb.prange(m):
        u = src[i];
        v = dst[i]
        if inactive_msk[u] or inactive_msk[v]:
            continue

        r_uv = roll_uv[i];
        r_vu = roll_vu[i]

        # u -> v
        if infectious[u] and susceptible[v] and (not dead[v]):
            if r_uv < p_uv[i] * rel_sus[v]:
                succ_uv[i] = True

        # v -> u
        if infectious[v] and susceptible[u] and (not dead[u]):
            if r_vu < p_vu[i] * rel_sus[u]:
                succ_vu[i] = True

    return succ_uv, succ_vu


@nb.njit(nogil=True, fastmath=True, parallel=True, cache=True)
def _party_edges(pool: np.ndarray, offsets: np.ndarray, base_p, same_bias,
                 genders, sexcat, compat_flat):
    """Build directed edges for *one whole day* of one party‑layer.

    Parameters
    ----------
    pool : 1‑D int64
        Attendee IDs in arbitrary order (first group, then next group…)
    offsets : 1‑D int32 (len = n_groups+1)
        Cumulative sizes → offsets[g]…offsets[g+1] is one party.
        We pass pre‑computed offsets so that *_update_*() can choose the
        size distribution (10‑30 vs 100‑500 vs 3‑60) **without** any
        Python loops inside this njit function.
    base_p : float
        Per‑slot Bernoulli probability that a chosen slot becomes an edge.
    same_bias : float
        Multiplier for same‑gender pairs (≥1); captures observed F‑F / M‑M
        clustering at bars / crowds.
    genders : 1‑D uint8 view  (0 = male , 1 = female)
    sexcat  : 1‑D uint8   (0…4 encoding  F‑het, M‑het, MSM, WSW, Bi)
    compat_flat : 1‑D length‑25 bool  –  compat.ravel(); avoids 2‑D index.

    Returns
    -------
    src, dst : int64 – one *directed* edge per element (both directions
                are already created where compatible & Bernoulli‑kept).
    """
    # --------------------------------------------------------------
    #  Estimate maximum edges  (4 × n  ≈ generous upper bound)
    #  → avoids expensive dynamic resize in njit. "4" is empirical:
    #    Poisson(2) draws × Bernoulli(≤1) ⇒ ~2 realised edges/pax.
    #    Even dense small parties rarely exceed 4×n.
    # --------------------------------------------------------------
    est = pool.size * 4
    src = np.empty(est, dtype=np.int64)
    dst = np.empty(est, dtype=np.int64)
    pos = 0

    # ----- iterate over each party (offsets define the slices) -----
    for g in range(offsets.size - 1):
        a = offsets[g]
        b = offsets[g + 1]
        grp = pool[a:b]
        n = grp.size
        if n < 2:
            continue

        # partner slots ~ Poisson(λ ≈2.2).  Draw ONCE for this group

        k = np.random.poisson(2.2, n)  # mean 2.2 , var 2.2  (fast!)
        # k = np.random.negative_binomial(1, 0.2,n)
        tot = k.sum()
        if tot == 0:
            continue

        # repeat each attendee ID k_i times → source list
        rep_src = np.repeat(grp, k)

        # choose random partners inside the *same* group
        # pick indices (0…n‑1) then map to IDs
        partners = np.random.randint(0, n, size=tot)
        rep_dst = grp[partners]

        # --- mask: remove self‑loops & enforce orientation compat ----
        mask = rep_src != rep_dst
        if mask.any():
            src_masked = rep_src[mask]
            dst_masked = rep_dst[mask]

            # orientation compatibility  (flattened 5×5 table)
            mask2 = compat_flat[sexcat[src_masked] * 5 + sexcat[dst_masked]]
            if mask2.any():
                src_masked = src_masked[mask2]
                dst_masked = dst_masked[mask2]

                # Bernoulli keep (same‑gender bias for bar / crowd layer)
                if base_p < 1.0 or same_bias != 1.0:
                    prob = base_p * np.where(genders[src_masked] == genders[dst_masked],
                                             same_bias, 1.0)
                    keep = np.random.random(src_masked.size) < prob
                    src_masked = src_masked[keep]
                    dst_masked = dst_masked[keep]

                m = src_masked.size
                if m:
                    src[pos:pos + m] = src_masked
                    dst[pos:pos + m] = dst_masked
                    pos += m

    return src[:pos], dst[:pos]


@nb.njit(nogil=True, fastmath=True, cache=True)
def _party_edges_from_k(pool: np.ndarray,  # int64 – attendees
                        offsets: np.ndarray,  # int32  – party slices
                        k_vec: np.ndarray):  # int32  – slot counts
    """
    Build directed edges for a party layer when each attendee’s number of
    partner-slots *k_i* is supplied by the caller.

    • NO Bernoulli thinning
    • NO gender bias
    • NO orientation compatibility filter
    --------------------------------------------------------------------
    Returns
    -------
    src, dst : int64 arrays (one *directed* edge per element)
    """
    # maximum edges = total partner-slots (every slot kept)
    max_edges = k_vec.sum()
    src = np.empty(max_edges, dtype=np.int64)
    dst = np.empty(max_edges, dtype=np.int64)
    pos = 0

    # iterate over each individual party
    for g in nb.prange(offsets.size - 1):
        a = offsets[g]
        b = offsets[g + 1]

        grp = pool[a:b]
        if grp.size < 2:  # solo attendee?  skip
            continue

        k = k_vec[a:b]
        tot = k.sum()
        if tot == 0:
            continue

        # repeat each attendee ID k_i times  →  source list
        rep_src = np.repeat(grp, k)  # shape (tot,)

        # choose random partners **within the same party**
        partners = np.random.randint(0, grp.size, tot)
        rep_dst = grp[partners]

        # remove self-loops only
        mask = rep_src != rep_dst
        if not mask.any():
            continue

        m = mask.sum()
        src[pos:pos + m] = rep_src[mask]
        dst[pos:pos + m] = rep_dst[mask]
        pos += m

    return src[:pos], dst[:pos]


@nb.njit(nogil=True, fastmath=True, cache=True)
def _party_edges_m2f_from_k(pool, offsets, k_vec, genders_uint8):
    max_edges = k_vec.sum()
    src = np.empty(max_edges, dtype=np.int64)
    dst = np.empty(max_edges, dtype=np.int64)
    pos = 0

    for g in nb.prange(offsets.size - 1):
        a = offsets[g];
        b = offsets[g + 1]
        grp = pool[a:b]
        if grp.size < 2:
            continue

        k = k_vec[a:b]
        tot = k.sum()
        if tot == 0:
            continue

        rep_src = np.repeat(grp, k)
        partners = np.random.randint(0, grp.size, tot)
        rep_dst = grp[partners]

        # keep only male-source & female-target, and no self-loops
        mask = (rep_src != rep_dst) & \
               (genders_uint8[rep_src] == 0) & \
               (genders_uint8[rep_dst] == 1)

        if not mask.any():
            continue

        m = mask.sum()
        src[pos:pos + m] = rep_src[mask]
        dst[pos:pos + m] = rep_dst[mask]
        pos += m

    return src[:pos], dst[:pos]


# NOTE: serial (not parallel) so that np.random.seed() inside is reproducible.
@nb.njit(nogil=True, fastmath=True, cache=True)
def build_mcp_edges(mcp_idx, ages, genders, sexcat, sex_role,
                    compat, k_part,
                    beta_anal_receptive, beta_anal_insertive, beta_vag, beta_oral, beta_skin,
                    rng_seed: int = 0):
    """Build static sexual-contact network for the MCP layer.
    +
    Stores each selected partner-pair once (undirected storage) but returns
    *directional* betas (beta_uv, beta_vu) for u→v and v→u.
    """
    # Seed numba's internal RNG so static network construction is reproducible
    # per (seed); numba's legacy np.random does NOT follow host np.random.seed().
    np.random.seed(rng_seed)
    n_agents = mcp_idx.size

    # ---------- Pass-1: per-agent desired counts ----------
    counts_need = k_part.astype(np.int64)
    # ---------- Prefix sum offsets ----------
    offsets = np.empty(n_agents + 1, dtype=np.int64)
    total = 0
    for i in range(n_agents):
        offsets[i] = total
        total += counts_need[i]
    offsets[n_agents] = total
    if total == 0:
        return (np.empty(0, np.int64),
                np.empty(0, np.int64),
                np.empty(0, np.float32),
                np.empty(0, np.float32))
    src = np.empty(total, dtype=np.int64)
    dst = np.empty(total, dtype=np.int64)
    beta_uv = np.empty(total, dtype=np.float32)
    beta_vu = np.empty(total, dtype=np.float32)
    counts_filled = np.zeros(n_agents, dtype=np.int64)
    # ---------- Pass-2: fill (may fill fewer than requested) ----------

    for n in range(n_agents):  # serial: reproducible np.random.seed() ordering
        uid = mcp_idx[n]
        need = counts_need[n]

        if need <= 0:
            continue
        cand = mcp_idx[(np.abs(ages[mcp_idx] - ages[uid]) <= 10) & (mcp_idx != uid)]
        if cand.size == 0:
            continue

        ok = compat[sexcat[uid], sexcat[cand]]
        cand = cand[ok]
        if cand.size == 0:
            continue
        draw = np.random.choice(cand, min(need, cand.size), replace=False)
        filled = draw.size
        if filled == 0:
            continue

        base = offsets[n]
        for j in range(filled):
            v = draw[j]
            idx = base + j
            src[idx] = uid
            dst[idx] = v
            if genders[uid] == genders[v]:  # same sex
                if genders[uid] == 0:  # MSM
                    r1 = sex_role[uid]
                    r2 = sex_role[v]
                    if r1 == 1 and r2 == 0:
                        beta_uv[idx] = beta_anal_receptive
                        beta_vu[idx] = beta_anal_insertive
                    elif r1 == 0 and r2 == 1:
                        beta_uv[idx] = beta_anal_insertive
                        beta_vu[idx] = beta_anal_receptive
                    elif r1 == 2 and r2 == 0:
                        beta_uv[idx] = beta_anal_receptive
                        beta_vu[idx] = beta_anal_insertive
                    elif r1 == 0 and r2 == 2:
                        beta_uv[idx] = beta_anal_insertive
                        beta_vu[idx] = beta_anal_receptive
                    elif r1 == 1 and r2 == 2:
                        beta_uv[idx] = beta_anal_receptive
                        beta_vu[idx] = beta_anal_insertive
                    elif r1 == 2 and r2 == 1:
                        beta_uv[idx] = beta_anal_insertive
                        beta_vu[idx] = beta_anal_receptive
                    elif r1 == 2 or r2 == 2:
                        beta_uv[idx] = 0.5 * (beta_anal_receptive + beta_anal_insertive)
                        beta_vu[idx] = beta_uv[idx]
                    else:
                        beta_uv[idx] = beta_skin
                        beta_vu[idx] = beta_skin
                else:  # WSW
                    beta_uv[idx] = beta_oral
                    beta_vu[idx] = beta_oral
            else:

                beta_uv[idx] = beta_vag
                beta_vu[idx] = beta_vag

        counts_filled[n] = filled

    # ---------- Compact ----------
    filled_total = 0

    for i in range(n_agents):
        filled_total += counts_filled[i]

    if filled_total == total:
        return src, dst, beta_uv, beta_vu

    src_c = np.empty(filled_total, dtype=np.int64)
    dst_c = np.empty(filled_total, dtype=np.int64)
    beta_uv_c = np.empty(filled_total, dtype=np.float32)
    beta_vu_c = np.empty(filled_total, dtype=np.float32)

    w = 0

    for i in range(n_agents):
        k = counts_filled[i]
        if k <= 0:
            continue
        base = offsets[i]
        for t in range(k):
            src_c[w + t] = src[base + t]
            dst_c[w + t] = dst[base + t]
            beta_uv_c[w + t] = beta_uv[base + t]
            beta_vu_c[w + t] = beta_vu[base + t]
        w += k

    return src_c, dst_c, beta_uv_c, beta_vu_c


@nb.njit(cache=True)
def dedup_by_day(ids, seen_day, day):
    out = np.empty(ids.size, dtype=np.int64)
    k = 0
    for i in range(ids.size):
        x = ids[i]
        if seen_day[x] != day:
            seen_day[x] = day
            out[k] = x
            k += 1
    return out[:k]


# 先串行稳定，确认无崩溃后把 parallel=True 打开
# 先串行稳定，确认无崩溃后把 parallel=True 打开
# NOTE: serial (not parallel) so that np.random.seed() inside is reproducible.
# Parallel numba RNG ordering is non-deterministic, which breaks seed-to-seed
# reproducibility of the one-off layer.
@nb.njit(nogil=True, fastmath=True, cache=True, boundscheck=True)
def build_oneoff_edges_fast(active: np.ndarray,  # int64 – 今日参与 one-off 的 uid
                            ages: np.ndarray,  # int16
                            genders: np.ndarray,  # 0/1 (int8/int32 都可)
                            sexcat: np.ndarray,  # uint8 (0..5)
                            sex_role: np.ndarray,  # int8 (-1 或 0/1/2 for MSM)
                            compat: np.ndarray,  # (6,6) bool
                            p_anal_rec: float,
                            p_anal_ins: float,
                            p_vag: float,
                            p_oral: float,
                            p_skin: float,
                            mf_anal_frac: float = 0.10,
                            age_gap: int = 20,
                            oneoff_p: float = 0.15,  # NegBin(n=1, p) 的 p
                            keep_scale: float = 1.0,  # 行为缩放，0~1
                            rng_seed: int = 0  # deterministic RNG seed (seeded inside njit)
                            ):
    """
    稳定版 one-off 并行生成：
      - Pass-1 统计目标 need_i
      - Prefix 分配容量
      - Pass-2 实际写入且记录 filled_i
      - 若总 filled < total，做一次紧凑压缩，绝不返回未初始化槽位
    返回：
      src, dst    : int64
      p_uv, p_vu  : float32
    """
    # Seed numba's internal RNG so edge generation is reproducible per (seed, day).
    # Numba's legacy np.random does NOT follow the host np.random.seed(), so it
    # must be seeded explicitly here; rng_seed is derived from (self._seed, day).
    np.random.seed(rng_seed)
    m = active.size
    counts_need = np.zeros(m, dtype=np.int64)

    # -------- Pass-1: 计算每个 uid 期望配对数 need_i ------------
    for i in range(m):  # serial: reproducible np.random.seed() ordering
        # NegBin(1, p)；注意 numpy 支持 n 为整数（这里=1）
        need = np.random.negative_binomial(1, oneoff_p)
        if need < 0:
            need = 0
        counts_need[i] = need

    # -------- Prefix: 前缀和 -> 总容量 total ---------------------
    offsets = np.empty(m + 1, dtype=np.int64)
    total = 0
    for i in range(m):
        offsets[i] = total
        total += counts_need[i]
    offsets[m] = total

    # 若 total=0，直接返回空
    if total == 0:
        return (np.empty(0, np.int64),
                np.empty(0, np.int64),
                np.empty(0, np.float32),
                np.empty(0, np.float32))

    # -------- 预分配最大容量（可能部分未被填满） -------------------
    src = np.empty(total, dtype=np.int64)
    dst = np.empty(total, dtype=np.int64)
    p_uv = np.empty(total, dtype=np.float32)
    p_vu = np.empty(total, dtype=np.float32)

    # 每个 i 实际成功条数
    counts_filled = np.zeros(m, dtype=np.int64)

    # -------- Pass-2: 实际填充 & 记录 filled_i -------------------
    for i in range(m):  # 稳定后可换 nb.prange
        uid = active[i]
        need = counts_need[i]
        if need <= 0:
            continue

        base = offsets[i]
        filled = 0
        tries = 0
        max_tries = need * 10 + 20  # 限制尝试次数，避免极端死循环

        # 局部“去重”缓存，避免 uid 对同一 v 重复
        chosen = np.empty(need, dtype=np.int64)
        chosen_size = 0

        while (filled < need) and (tries < max_tries):
            tries += 1
            j = np.random.randint(0, m)
            v = active[j]
            if v == uid:
                continue
            # 年龄差
            if abs(int(ages[uid]) - int(ages[v])) > age_gap:
                continue
            # 取向兼容（6×6）
            if not compat[int(sexcat[uid]), int(sexcat[v])]:
                continue
            # 行为缩放（Bernoulli）
            if keep_scale < 1.0 and (np.random.random() >= keep_scale):
                continue
            # 局部去重
            dup = False
            for t in range(chosen_size):
                if chosen[t] == v:
                    dup = True
                    break
            if dup:
                continue

            # 写入一条边
            idx = base + filled
            src[idx] = uid
            dst[idx] = v

            gu = genders[uid];
            gv = genders[v]
            if (gu == 0) and (gv == 0):  # MSM
                r1 = sex_role[uid];
                r2 = sex_role[v]
                if r1 == 1 and r2 == 0:  # uid insertive, v receptive
                    p_uv[idx] = p_anal_rec;
                    p_vu[idx] = p_anal_ins
                elif r1 == 0 and r2 == 1:  # uid receptive, v insertive
                    p_uv[idx] = p_anal_ins;
                    p_vu[idx] = p_anal_rec
                elif r1 == 2 and r2 == 0:  # uid versatile, v receptive
                    p_uv[idx] = p_anal_rec;
                    p_vu[idx] = p_anal_ins
                elif r1 == 0 and r2 == 2:  # uid receptive, v versatile
                    p_uv[idx] = p_anal_ins;
                    p_vu[idx] = p_anal_rec
                elif r1 == 1 and r2 == 2:  # uid insertive, v versatile
                    p_uv[idx] = p_anal_rec;
                    p_vu[idx] = p_anal_ins
                elif r1 == 2 and r2 == 1:  # uid versatile, v insertive
                    p_uv[idx] = p_anal_ins;
                    p_vu[idx] = p_anal_rec
                elif (r1 == 2) or (r2 == 2):  # 任一 versatile：平均
                    avg = 0.5 * (p_anal_rec + p_anal_ins)
                    p_uv[idx] = avg;
                    p_vu[idx] = avg
                else:  # 角色冲突或未知 → 皮肤
                    p_uv[idx] = p_skin;
                    p_vu[idx] = p_skin
            elif (gu == 1) and (gv == 1):  # WSW
                p_uv[idx] = p_oral;
                p_vu[idx] = p_oral
            else:  # 异性
                if np.random.random() < mf_anal_frac:  # oneoff_mf_anal_frac
                    # treat as anal sex: receptive acquisition vs insertive acquisition
                    p_uv[idx] = p_anal_rec;
                    p_vu[idx] = p_anal_ins
                else:
                    p_uv[idx] = p_vag;
                    p_vu[idx] = p_vag

            chosen[chosen_size] = v
            chosen_size += 1
            filled += 1

        counts_filled[i] = filled  # 记录实际写入条数

    # -------- 压缩：仅当总 filled < total 时 -----------------------
    filled_total = 0
    for i in range(m):
        filled_total += counts_filled[i]

    if filled_total == total:
        # 恰好都写满，无需压缩
        return src, dst, p_uv, p_vu

    # 否则压缩到紧凑数组（避免未初始化槽位泄露）
    src_c = np.empty(filled_total, dtype=np.int64)
    dst_c = np.empty(filled_total, dtype=np.int64)
    p_uv_c = np.empty(filled_total, dtype=np.float32)
    p_vu_c = np.empty(filled_total, dtype=np.float32)

    write_ptr = 0
    for i in range(m):
        base = offsets[i]
        k = counts_filled[i]
        if k > 0:
            end = base + k
            # 复制已写入的有效段
            for t in range(k):
                src_c[write_ptr + t] = src[base + t]
                dst_c[write_ptr + t] = dst[base + t]
                p_uv_c[write_ptr + t] = p_uv[base + t]
                p_vu_c[write_ptr + t] = p_vu[base + t]
            write_ptr += k

    return src_c, dst_c, p_uv_c, p_vu_c


@nb.njit(nogil=True, fastmath=True, cache=True)
def build_edge_risks(src, dst, genders, sex_role,
                     p_anal_rec, p_anal_ins, p_vag, p_oral, p_skin):
    """Compute per-edge directional transmission probabilities.

    Returns p_uv, p_vu with same length as src/dst.
    For male-male edges, uses sex roles to assign receptive/insertive risks.
        If p_mf_anal_frac>0, a fraction of MF/FM edges are treated as anal sex:
      - MF (male->female): dst is receptive
      - FM (female->male): src is receptive (female), dst is insertive (male)
    """
    m = src.size
    p_uv = np.empty(m, dtype=np.float32)
    p_vu = np.empty(m, dtype=np.float32)

    for i in range(m):
        u = src[i]
        v = dst[i]
        gu = genders[u]
        gv = genders[v]

        if (gu == 0) and (gv == 0):
            r1 = sex_role[u]
            r2 = sex_role[v]
            # mapping matches build_oneoff_edges_fast / build_mcp_edges conventions
            if r1 == 1 and r2 == 0:  # u insertive, v receptive
                p_uv[i] = p_anal_rec
                p_vu[i] = p_anal_ins
            elif r1 == 0 and r2 == 1:  # u receptive, v insertive
                p_uv[i] = p_anal_ins
                p_vu[i] = p_anal_rec
            elif r1 == 2 and r2 == 0:  # u versatile, v receptive
                p_uv[i] = p_anal_rec
                p_vu[i] = p_anal_ins
            elif r1 == 0 and r2 == 2:  # u receptive, v versatile
                p_uv[i] = p_anal_ins
                p_vu[i] = p_anal_rec
            elif r1 == 1 and r2 == 2:  # u insertive, v versatile
                p_uv[i] = p_anal_rec
                p_vu[i] = p_anal_ins
            elif r1 == 2 and r2 == 1:  # u versatile, v insertive
                p_uv[i] = p_anal_ins
                p_vu[i] = p_anal_rec
            elif (r1 == 2) or (r2 == 2):
                avg = 0.5 * (p_anal_rec + p_anal_ins)
                p_uv[i] = avg
                p_vu[i] = avg
            else:
                p_uv[i] = p_skin
                p_vu[i] = p_skin

        elif (gu == 1) and (gv == 1):
            p_uv[i] = p_oral
            p_vu[i] = p_oral

        else:
            p_uv[i] = p_vag
            p_vu[i] = p_vag

    return p_uv, p_vu


class Population:
    """Synthetic population with multi‑layer contact networks and
    simple mpox natural‑history & transmission model.

    The implementation is purposely compact so you can easily read,
    tweak and extend it.  All heavy lifting (loops over 100 k agents)
    still sits in fast NumPy code.

    Version: May‑2025 – orientation‑aware sexual/party layers,
    small/large “no‑sex” events, one‑off edges and negative‑binomial
    partner counts.  Feel free to adjust any constants! """

    # ------------------------------------------------------------------
    #  Helper: draw from a NegBin(r=1, p)
    #  mean  = (1-p)/p,    var = (1-p)/p^2      ⇒ heavy right tail
    # ------------------------------------------------------------------

    #    @staticmethod
    # def _draw_nb(size:int, mean:float,disp:float) -> np.ndarray:
    #     """Return an array of random counts with given mean and
    #     an over‑dispersed negative‑binomial (r=1) distribution."""
    #     p = 1 / (1 + mean * disp)
    #     r = mean * p / (1 - p)
    #     return nbinom(r, p).rvs(size=size)

    @staticmethod
    def _draw_nb_rng(rng, size: int, mean: float, disp: float) -> np.ndarray:
        """
        mean, disp -> (r, p) 的一个常用映射；用 numpy Generator 直接抽。
        """
        # 令 r = 1/disp，p = r / (r + mean)
        r = 1.0 / max(disp, 1e-8)
        p = r / (r + mean)
        return rng.negative_binomial(r, p, size=size).astype(np.int32)

    def _load_or_build_demographics(self):
        # ---- 把你人口学生成规则写入 key（你以后改了这里也要改）----
        demo_params = dict(
            # orientation probs
            male_ori=(0.910, 0.039, 0.051),
            female_ori=(0.897, 0.076, 0.027),
            # msm role probs
            msm_role=(0.199, 0.312, 0.489),  # insertive/ receptive/ versatile
            # mcp flags probs
            mcp_prob=(
                0.019, 0.03, 0.063,  # male hetero/bi/homo
                0.02, 0.07, 0.15,  # female hetero/bi/homo
            ),
            # sex party eligible probs
            party_prob=(
                # 0.35, 0.25, 0.85,  # male homo/bi/hetero
                # 0.25, 0.15, 0.85,  # male homo/bi/hetero
                0.35, 0.25, 0.15,  # male homo/bi/hetero
                0.25, 0.20, 0.15,  # female homo/bi/hetero
            ),
            # 如果你未来改年龄分布，也应该在这里加一个版本号/参数
            age_dist="uniform_0_90_v1",
            gender_dist="50_50_v1",
            compat_version="v1",
            activity_level_version="v4",  # bump: sex_party activity level -> 6-level riskg gradient
        )

        key = _make_cache_key(self.n, self._seed, {"demo": demo_params})
        cache_path = f"cache/demo/demo_{key}.npz"

        if os.path.exists(cache_path):
            print("Loading demographics from cache...")
            d = load_npz(cache_path)

            # ---- 直接恢复数组（人口学标签）----
            self.ages = d["ages"].astype(np.int16, copy=False)
            self.genders = d["genders"].astype(np.int8, copy=False)

            # 这里 orientations 是字符串数组：np.savez 可以存，但会占空间；
            # 更快更省建议存 sexcat(uint8) + 不再需要 orientations。
            # 你目前代码很多地方用 orientations 字符串，所以这里仍然恢复它：
            self.orientations = d["orientations"]  # dtype '<U6'

            self.sex_role = d["sex_role"].astype(np.int8, copy=False)
            self.sexcat = d["sexcat"].astype(np.uint8, copy=False)

            self.mcp_flags = d["mcp_flags"].astype(bool, copy=False)
            self.mcp_idx = d["mcp_idx"].astype(np.int64, copy=False)

            # Activity-level stratification (heavy-tail). Backward-compatible:
            # older caches lack this field, so regenerate deterministically from
            # the same RNG stream used at creation time is not possible here;
            # instead re-derive from the stored mcp_idx with a fresh rng seeded
            # by the cache key for reproducibility.
            if "activity_level" in d:
                self.activity_level = d["activity_level"].astype(np.int8, copy=False)
            else:
                self.activity_level = np.full(self.n, -1, dtype=np.int8)
                activity_weights = np.array([0.19, 0.19, 0.19, 0.19, 0.19, 0.05], dtype=np.float64)
                activity_weights /= activity_weights.sum()
                # Construction-phase RNG: independent of self._rng so that
                # demographics/network building never perturbs the runtime
                # transmission RNG stream (seed reproducibility).
                _constr_rng = np.random.default_rng(self._seed)
                self.activity_level[self.mcp_idx] = _constr_rng.choice(
                    np.arange(6, dtype=np.int8), size=self.mcp_idx.size, p=activity_weights
                )

            self.sex_party_eligible = d["sex_party_eligible"].astype(bool, copy=False)

            # High-activity 6-level riskg stratification for the sex_party layer.
            # Stored in the cache since v4; older caches migrate the binary
            # `sex_party_high_active` flag -> stratum 5, or rebuild from
            # sex_party_eligible with a fresh rng seeded by self._seed.
            if "sex_party_activity_level" in d:
                self.sex_party_activity_level = d["sex_party_activity_level"].astype(np.int8, copy=False)
            elif "sex_party_high_active" in d:
                # migrate old binary flag -> 6-level (top 5% -> stratum 5)
                self.sex_party_activity_level = np.full(self.n, -1, dtype=np.int8)
                self.sex_party_activity_level[d["sex_party_high_active"].astype(bool)] = 5
            else:
                self._build_sex_party_activity_level()

            # compat 也恢复（或者你也可以固定用 _build_sex_categories 生成）
            self.compat = d["compat"].astype(bool, copy=False)
            self._compat_flat = self.compat.ravel()

            # allow all paire types for SOCIAL (no orientation restriction)
            self._compat_all_true = np.ones_like(self._compat_flat, dtype=np.bool_)
            # 一些派生量也顺手恢复/重建
            self._gender_uint8 = (self.genders == 1).astype(np.uint8)

            return key

        # ------------------ 不命中：按你原逻辑生成 ------------------
        print("Creating demographics...")
        n = self.n
        # Construction-phase RNG: independent of self._rng so demographics
        # generation never perturbs the runtime transmission RNG stream.
        rng = np.random.default_rng(self._seed)

        self.ages = rng.integers(0, 90, size=n, dtype=np.int16)
        self.genders = rng.choice(np.array([0, 1], dtype=np.int8), size=n)

        self.orientations = np.empty(n, dtype="<U6")
        male_mask = self.genders == 0
        female_mask = ~male_mask
        self.orientations[male_mask] = rng.choice(["hetero", "bi", "homo"], size=male_mask.sum(),
                                                  p=demo_params["male_ori"])
        self.orientations[female_mask] = rng.choice(["hetero", "bi", "homo"], size=female_mask.sum(),
                                                    p=demo_params["female_ori"])

        self.sex_role = np.full(n, -1, dtype=np.int8)
        sex_role_mask = male_mask & np.isin(self.orientations, ["homo", "bi"])
        self.sex_role[sex_role_mask] = rng.choice(np.array([0, 1, 2], dtype=np.int8),
                                                  size=sex_role_mask.sum(),
                                                  p=demo_params["msm_role"])

        # sexcat + compat
        self._build_sex_categories()  # 会设定 self.sexcat / self.compat / self._compat_flat
        self._compat_all_true = np.ones_like(self._compat_flat, dtype=np.bool_)
        self._gender_uint8 = (self.genders == 1).astype(np.uint8)

        # mcp_flags
        self.mcp_flags = np.zeros(n, dtype=bool)
        hetero = self.orientations == "hetero"
        bi = self.orientations == "bi"
        homo = self.orientations == "homo"
        male = male_mask
        female = ~male

        self.mcp_flags[(male & hetero)] = rng.random((male & hetero).sum()) < demo_params["mcp_prob"][0]
        self.mcp_flags[(male & bi)] = rng.random((male & bi).sum()) < demo_params["mcp_prob"][1]
        self.mcp_flags[(male & homo)] = rng.random((male & homo).sum()) < demo_params["mcp_prob"][2]
        self.mcp_flags[(female & hetero)] = rng.random((female & hetero).sum()) < demo_params["mcp_prob"][3]
        self.mcp_flags[(female & bi)] = rng.random((female & bi).sum()) < demo_params["mcp_prob"][4]
        self.mcp_flags[(female & homo)] = rng.random((female & homo).sum()) < demo_params["mcp_prob"][5]
        self.mcp_idx = np.where(self.mcp_flags)[0].astype(np.int64)

        # Activity-level stratification for the one-off layer (heavy-tail).
        # Aligned with the CDC MMWR model's 6-level `riskg` activity strata:
        #   5 levels x 19% + 1 top level x 5% (Weiss et al. 2020; Spicknall
        #   MMWR 2022). Weights come from the module-level CDC_RISKG_POP_WEIGHTS
        #   constant (single source of truth).
        # Level 5 (top 5%) carries the heavy tail of one-off partner formation
        # (~2/7 partners/day ≈ 100/year), matching the CDC qnts vector.
        # activity_level is 0..5 for MCP agents and -1 for non-MCP agents.
        self.activity_level = np.full(n, -1, dtype=np.int8)
        activity_weights = CDC_RISKG_POP_WEIGHTS / CDC_RISKG_POP_WEIGHTS.sum()
        mcp_uids = self.mcp_idx
        self.activity_level[mcp_uids] = rng.choice(
            np.arange(6, dtype=np.int8), size=mcp_uids.size, p=activity_weights
        )

        # sex_party_eligible
        party_prob = {
            (0, "homo"): demo_params["party_prob"][0],
            (0, "bi"): demo_params["party_prob"][1],
            (0, "hetero"): demo_params["party_prob"][2],
            (1, "homo"): demo_params["party_prob"][3],
            (1, "bi"): demo_params["party_prob"][4],
            (1, "hetero"): demo_params["party_prob"][5],
        }
        self.sex_party_eligible = np.zeros(n, dtype=bool)
        for (g, o), prob in party_prob.items():
            mask = (self.genders == g) & (self.orientations == o) & self.mcp_flags
            self.sex_party_eligible[mask] = rng.random(mask.sum()) < prob

        # High-activity 6-level riskg stratification for the sex_party layer.
        # Restricted to the MSM core pool (gay men eligible for sex parties), so
        # the heavy partner tail drives intra-MM amplification without widening
        # hetero spill.
        self._build_sex_party_activity_level()

        # -------------- 写入人口学缓存 --------------
        save_npz(cache_path,
                 ages=self.ages,
                 genders=self.genders,
                 orientations=self.orientations,
                 sex_role=self.sex_role,
                 sexcat=self.sexcat,
                 mcp_flags=self.mcp_flags,
                 mcp_idx=self.mcp_idx,
                 sex_party_eligible=self.sex_party_eligible,
                 sex_party_activity_level=self.sex_party_activity_level,
                 activity_level=self.activity_level,
                 compat=self.compat)

        return key

    def _build_sex_party_activity_level(self):
        """Assign a 6-level CDC `riskg` activity stratum (0..5) to the MSM core
        pool (gay men eligible for sex parties), driving the heavy partner tail
        in the sex_party layer.

        The stratification is restricted to the MSM core so the heavy tail
        drives intra-MM amplification without widening hetero spill. Weights
        follow CDC_RISKG_POP_WEIGHTS (5x19% + 1x5%, Weiss et al. 2020;
        Spicknall MMWR 2022). Non-core individuals keep -1.

        Replaces the previous binary `sex_party_high_active` flag with a full
        6-level gradient so that partner degree varies smoothly by stratum
        (see CDC_RISKG_PARTNERS_3W), rather than a two-tier split.
        """
        n = self.n
        self.sex_party_activity_level = np.full(n, -1, dtype=np.int8)
        msm_eligible = (self.genders == 0) & (self.orientations == "homo") & self.sex_party_eligible
        msm_eligible_uids = np.where(msm_eligible)[0]
        if msm_eligible_uids.size == 0:
            return
        weights = CDC_RISKG_POP_WEIGHTS / CDC_RISKG_POP_WEIGHTS.sum()
        # Deterministic given the same self._seed for cache reproducibility.
        _rng = np.random.default_rng(self._seed)
        self.sex_party_activity_level[msm_eligible_uids] = _rng.choice(
            np.arange(6, dtype=np.int8), size=msm_eligible_uids.size, p=weights
        )

    def _load_or_build_household_partners(self, demo_key: str):
        """Build (or load) a *stable* one-to-one household/regular-partner network.

        We store partners as an int32 array partner_of[i] = j (or -1 if none).
        This is O(N) memory and scales to very large N (e.g., 10M) without storing edges.
        """
        hh_params = dict(
            hh_partner_rate=float(self.params.get("hh_partner_rate", 0.55)),  # fraction of adults with a stable partner
            hh_assort=float(self.params.get("hh_assort", 0.25)),
            # assortativity by activity group (party-eligible vs not)
            bi_same_sex_frac=float(self.params.get("bi_same_sex_frac", 0.10)),
            bi_same_sex_frac_male=float(self.params.get("bi_same_sex_frac", 0.10)),
            build_household_version="v1",
        )
        key = _hash_params({"demo_key": demo_key, "household": hh_params})
        cache_path = f"cache/household/household_{key}.npz"
        os.makedirs("cache/household", exist_ok=True)

        if os.path.exists(cache_path):
            if self.verbose:
                print("Loading household partner network from cache...")
            d = load_npz(cache_path)
            self.partner_of = d["partner_of"].astype(np.int32, copy=False)
            # household act type per person (0=MM anal, 1=MF vaginal, 2=FF oral)
            if "hh_type" in d:
                self.hh_type = d["hh_type"].astype(np.uint8, copy=False)
            else:
                # fallback: derive from genders (fast, no RNG)
                self.hh_type = np.full(self.n, 255, dtype=np.uint8)
                has = self.partner_of >= 0
                i = np.where(has)[0]
                j = self.partner_of[has]
                gi = self.genders[i]
                gj = self.genders[j]
                # note: we only need type for i (and we will mirror below)
                t = np.where((gi == 0) & (gj == 0), 0, np.where(gi != gj, 1, 2)).astype(np.uint8)
                self.hh_type[i] = t
                self.hh_type[j] = t
            # male-male role flag per person: 1 if this person is the receptive partner in MM pair, else 0; -1 for non-MM
            if "hh_mm_receptive" in d:
                self.hh_mm_receptive = d["hh_mm_receptive"].astype(np.int8, copy=False)
            else:
                self.hh_mm_receptive = np.full(self.n, -1, dtype=np.int8)

            # build a compact list of unique pairs (i<j) for fast daily activation
            self.hh_u, self.hh_v = np.where(self.partner_of >= 0)[0], self.partner_of[self.partner_of >= 0]
            # keep only i<j to avoid duplicates
            mask = self.hh_u < self.hh_v
            self.hh_u = self.hh_u[mask].astype(np.int64, copy=False)
            self.hh_v = self.hh_v[mask].astype(np.int64, copy=False)
            return

        if self.verbose:
            print("Creating household partner network...")

        self.partner_of, self.hh_type, self.hh_mm_receptive = self._create_household_partners(hh_params)
        self.partner_of = self.partner_of.astype(np.int32, copy=False)
        self.hh_type = self.hh_type.astype(np.uint8, copy=False)
        self.hh_mm_receptive = self.hh_mm_receptive.astype(np.int8, copy=False)

        save_npz(cache_path, partner_of=self.partner_of, hh_type=self.hh_type, hh_mm_receptive=self.hh_mm_receptive)

        # precompute unique pairs for fast daily activation
        self.hh_u = np.where(self.partner_of >= 0)[0].astype(np.int64, copy=False)
        self.hh_v = self.partner_of[self.partner_of >= 0].astype(np.int64, copy=False)
        mask = self.hh_u < self.hh_v
        self.hh_u = self.hh_u[mask]
        self.hh_v = self.hh_v[mask]

    def _create_household_partners(self, hh_params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create stable partners for a subset of adults.

        Pairing rules (simple, fast, and explainable):
        - Adults only (age>=18).
        - MSM paired with men; WSW paired with women.
        - Heterosexuals paired male-female.
        - Bisexuals are split into same-sex vs opposite-sex pools by bi_same_sex_frac_*.
        - Activity assortativity: with probability hh_assort, choose partner from same activity group
          (activity group proxy = sex_party_eligible); otherwise allow bridging (core<->non-core).
        """
        # Construction-phase RNG: independent of self._rng so network building
        # never perturbs the runtime transmission RNG stream.
        rng = np.random.default_rng(self._seed)
        n = self.n
        partner_of = np.full(n, -1, dtype=np.int32)
        # act type per person in household pair: 0=MM anal, 1=MF vaginal, 2=FF oral, 255=none
        hh_type = np.full(n, 255, dtype=np.uint8)
        # for MM pairs only: 1 if this person is the receptive partner, 0 if insertive; -1 for non-MM / none
        hh_mm_receptive = np.full(n, -1, dtype=np.int8)

        adult = self.ages >= 18

        # activity group proxy: party-eligible
        core = self.mcp_flags & adult
        noncore = (~self.mcp_flags) & adult

        # who gets a stable partner
        hh_partner_rate = float(hh_params["hh_partner_rate"])
        has_partner = adult & (rng.random(n) < hh_partner_rate)

        # helper to pick pool with assortativity
        def _split_by_core(ids: np.ndarray):
            c = ids[core[ids]]
            nc = ids[noncore[ids]]
            rng.shuffle(c);
            rng.shuffle(nc)
            return c, nc

        hh_assort = float(hh_params["hh_assort"])

        # --- same-sex pools ---
        msm = np.where(has_partner & (self.sexcat == 2))[0]
        wsw = np.where(has_partner & (self.sexcat == 3))[0]

        # bisexual split
        bi_m = np.where(has_partner & (self.sexcat == 4))[0]
        bi_f = np.where(has_partner & (self.sexcat == 5))[0]
        bi_m_same = bi_m[rng.random(bi_m.size) < float(hh_params["bi_same_sex_frac"])]
        bi_m_opp = np.setdiff1d(bi_m, bi_m_same, assume_unique=False)
        bi_f_same = bi_f[rng.random(bi_f.size) < float(hh_params["bi_same_sex_frac"])]
        bi_f_opp = np.setdiff1d(bi_f, bi_f_same, assume_unique=False)

        msm_pool = np.concatenate([msm, bi_m_same]).astype(np.int64, copy=False)
        wsw_pool = np.concatenate([wsw, bi_f_same]).astype(np.int64, copy=False)

        def _match_with_assort(pool: np.ndarray):
            if pool.size < 2:
                return
            c, nc = _split_by_core(pool)
            # within-core pairs
            m = (c.size // 2) * 2
            for i in range(0, m, 2):
                u, v = int(c[i]), int(c[i + 1])
                partner_of[u] = v;
                partner_of[v] = u
            # within-noncore pairs
            m2 = (nc.size // 2) * 2
            for i in range(0, m2, 2):
                u, v = int(nc[i]), int(nc[i + 1])
                partner_of[u] = v;
                partner_of[v] = u
            # bridging leftovers (or if hh_assort low)
            # collect leftovers
            c_left = c[m:]
            nc_left = nc[m2:]
            # with probability (1-hh_assort), allow more cross-group matching by moving some within pairs to bridging
            # (cheap approximation)
            # Always bridge leftovers:
            k = min(c_left.size, nc_left.size)
            for i in range(k):
                u, v = int(c_left[i]), int(nc_left[i])
                if partner_of[u] == -1 and partner_of[v] == -1:
                    partner_of[u] = v;
                    partner_of[v] = u

        # Apply assortativity by optionally shuffling some people between core/noncore before matching
        # We approximate assortativity by allowing a fraction (1-hh_assort) of each pool to be swapped.
        def _apply_assort_pool(pool: np.ndarray):
            if pool.size == 0:
                return pool
            if hh_assort >= 0.999:
                return pool
            c, nc = _split_by_core(pool)
            # move a slice from core to noncore and vice versa to create bridging opportunities
            swap_c = int((1.0 - hh_assort) * c.size)
            swap_nc = int((1.0 - hh_assort) * nc.size)
            if swap_c > 0 and swap_nc > 0:
                c_sw = c[:swap_c]
                nc_sw = nc[:swap_nc]
                c = np.concatenate([c[swap_c:], nc_sw])
                nc = np.concatenate([nc[swap_nc:], c_sw])
            rng.shuffle(c);
            rng.shuffle(nc)
            return np.concatenate([c, nc])

        _match_with_assort(_apply_assort_pool(msm_pool))
        _match_with_assort(_apply_assort_pool(wsw_pool))

        # --- opposite-sex pools ---
        male_os = np.where(has_partner & (self.sexcat == 1))[0]  # male hetero
        female_os = np.where(has_partner & (self.sexcat == 0))[0]  # female hetero
        male_os = np.concatenate([male_os, bi_m_opp]).astype(np.int64, copy=False)
        female_os = np.concatenate([female_os, bi_f_opp]).astype(np.int64, copy=False)

        # remove anyone already paired in same-sex step (shouldn't happen but safe)
        male_os = male_os[partner_of[male_os] == -1]
        female_os = female_os[partner_of[female_os] == -1]

        # split by core/noncore for assortativity
        m_core, m_non = _split_by_core(male_os)
        f_core, f_non = _split_by_core(female_os)

        # within-group pairing first
        k1 = min(m_core.size, f_core.size)
        for i in range(k1):
            u, v = int(m_core[i]), int(f_core[i])
            partner_of[u] = v;
            partner_of[v] = u
        k2 = min(m_non.size, f_non.size)
        for i in range(k2):
            u, v = int(m_non[i]), int(f_non[i])
            partner_of[u] = v;
            partner_of[v] = u

        # bridging pairs (core male with noncore female, etc.)
        m_left = m_core[k1:]
        f_left = f_non[k2:]
        k3 = min(m_left.size, f_left.size)
        for i in range(k3):
            u, v = int(m_left[i]), int(f_left[i])
            if rng.random() < (1.0 - hh_assort):
                partner_of[u] = v;
                partner_of[v] = u

        m_left2 = m_non[k2:]
        f_left2 = f_core[k1:]
        k4 = min(m_left2.size, f_left2.size)
        for i in range(k4):
            u, v = int(m_left2[i]), int(f_left2[i])
            if rng.random() < (1.0 - hh_assort):
                partner_of[u] = v;
                partner_of[v] = u

        # derive act types for all paired individuals (vectorized)
        has = partner_of >= 0
        i = np.where(has)[0]
        j = partner_of[has]
        gi = self.genders[i]
        gj = self.genders[j]
        t = np.where((gi == 0) & (gj == 0), 0, np.where(gi != gj, 1, 2)).astype(np.uint8)
        hh_type[i] = t
        hh_type[j] = t

        # assign fixed receptive/insertive roles for male-male pairs (fast, one-time)
        mm = (t == 0)
        if np.any(mm):
            iim = i[mm]
            jim = j[mm]
            # to avoid double-assigning both directions, only assign where i<j
            mask = iim < jim
            u = iim[mask]
            v = jim[mask]
            # random choose which partner is receptive
            u_receptive = rng.random(u.size) < 0.5
            hh_mm_receptive[u] = u_receptive.astype(np.int8)
            hh_mm_receptive[v] = (~u_receptive).astype(np.int8)
        return partner_of, hh_type, hh_mm_receptive

    def _update_household_network(self):
        """Daily active household sex edges from stable partners."""
        if not hasattr(self, "hh_u") or self.hh_u.size == 0:
            return (np.empty(0, np.int64),
                    np.empty(0, np.int64),
                    np.empty(0, np.float32),
                    np.empty(0, np.float32))

        rng = self._rng

        # ------------------------------------------------------------------
        # Stratified stable/regular-partner activation by MCP composition
        # ------------------------------------------------------------------
        # The stable partner network is generated among adults, not only among
        # MCP agents. Therefore, applying one uniform high value such as
        # p_household_sex_per_day=0.50 to all adult stable edges would
        # over-activate general-population stable partnerships.
        #
        # We instead classify each stable edge by whether neither, exactly one,
        # or both endpoints are MCP agents:
        #   1) neither endpoint is MCP: general adult stable edge
        #      p_day ~= 0.10, equivalent to about 3 contacts/month
        #   2) exactly one endpoint is MCP: MCP-bridging stable edge
        #      p_day ~= 0.26, equivalent to about 9 contacts/month
        #   3) both endpoints are MCP: MCP-core stable edge
        #      p_day ~= 0.39, equivalent to about 15 contacts/month
        #
        # Conversion from monthly frequency f_month:
        #      p_day = 1 - exp(-f_month / 30)
        # The old p_household_sex_per_day is retained only as a fallback for
        # backward compatibility if the new stratified parameters are absent.
        p_general = float(self.params.get(
            "p_stable_sex_general_per_day",
            self.params.get("p_household_sex_per_day", 0.10),
        ))
        p_one_mcp = float(self.params.get("p_stable_sex_one_mcp_per_day", 0.26))
        p_both_mcp = float(self.params.get("p_stable_sex_both_mcp_per_day", 0.39))

        p_general = min(1.0, max(0.0, p_general))
        p_one_mcp = min(1.0, max(0.0, p_one_mcp))
        p_both_mcp = min(1.0, max(0.0, p_both_mcp))

        u_all = self.hh_u.astype(np.int64, copy=False)
        v_all = self.hh_v.astype(np.int64, copy=False)

        # Default: neither endpoint is MCP.
        p_day = np.full(u_all.size, p_general, dtype=np.float32)

        if hasattr(self, "mcp_flags"):
            src_mcp = self.mcp_flags[u_all]
            dst_mcp = self.mcp_flags[v_all]
            one_mcp = src_mcp ^ dst_mcp  # exactly one endpoint is MCP
            both_mcp = src_mcp & dst_mcp  # both endpoints are MCP
            p_day[one_mcp] = np.float32(p_one_mcp)
            p_day[both_mcp] = np.float32(p_both_mcp)
        else:
            one_mcp = np.zeros(u_all.size, dtype=np.bool_)
            both_mcp = np.zeros(u_all.size, dtype=np.bool_)

        active = rng.random(u_all.size) < p_day
        if not np.any(active):
            return (np.empty(0, np.int64),
                    np.empty(0, np.int64),
                    np.empty(0, np.float32),
                    np.empty(0, np.float32))

        u = u_all[active].astype(np.int64, copy=False)
        v = v_all[active].astype(np.int64, copy=False)

        if (self.day % 14) == 0 and u.size:
            print(
                f"[day {self.day}] stable activation: edges={u.size}, "
                f"mean_p_day={float(np.mean(p_day[active])):.3f}, "
                f"general={p_general:.2f}, one_mcp={p_one_mcp:.2f}, "
                f"both_mcp={p_both_mcp:.2f}, "
                f"active_one_mcp={int(np.sum(one_mcp[active]))}, "
                f"active_both_mcp={int(np.sum(both_mcp[active]))}"
            )

        beta_rec = float(self.transmission_risks["sexual_anal_receptive"])
        beta_ins = float(self.transmission_risks["sexual_anal_insertive"])
        beta_vag = float(self.transmission_risks["sexual_vaginal"])
        beta_oral = float(self.transmission_risks["sexual_oral"])
        beta_skin = float(self.transmission_risks.get("close_touch", 0.0))  # usually unused here

        # returns per-couple probs for u->v and v->u
        puv, pvu = build_edge_risks(
            u, v,
            self.genders,
            self.sex_role,  # uses your person-level role for MM directional risk
            np.float32(beta_rec),
            np.float32(beta_ins),
            np.float32(beta_vag),
            np.float32(beta_oral),
            np.float32(beta_skin)
        )
        # ---- MF anal sex fraction on stable household edges ----
        # Calibrated from main.py via params["stable_mf_anal_frac"].
        # Keeping this as a parameter avoids hard-coding a high mixed-sex
        # stable-partner anal-contact fraction, which can otherwise inflate
        # female-known cases in late-stage calibration.
        mf_anal_frac = float(self.params.get("stable_mf_anal_frac", 0.05))
        if mf_anal_frac > 0.0 and u.size:
            gu = self.genders[u];
            gv = self.genders[v]
            is_mixed = (gu != gv)
            if np.any(is_mixed):
                idx_m = np.where(is_mixed)[0]
                take = rng.random(idx_m.size) < mf_anal_frac
                if np.any(take):
                    sel = idx_m[take]
                    puv[sel] = np.float32(self.transmission_risks['sexual_anal_receptive'])
                    pvu[sel] = np.float32(self.transmission_risks['sexual_anal_insertive'])

        # build directed edges both ways; one probability array is sufficient
        src = np.concatenate([u, v]).astype(np.int64, copy=False)
        dst = np.concatenate([v, u]).astype(np.int64, copy=False)
        p_uv = np.concatenate([puv, pvu]).astype(np.float32, copy=False)
        p_vu = np.concatenate([pvu, puv]).astype(np.float32, copy=False)

        # >>> 在这里插入 debug（用 src/dst/p_uv）
        if (self.day % 14) == 0 and src.size:
            s = src;
            d = dst
            bi_m_to_het_f = (
                    (self.genders[s] == 0) &
                    (self.orientations[s] == 'bi') &
                    (self.genders[d] == 1) &
                    (self.orientations[d] == 'hetero')
            )
            n_edges = int(np.sum(bi_m_to_het_f))
            if n_edges > 0:
                mean_p = float(np.mean(p_uv[bi_m_to_het_f]))
                inf_src = (self.infectious[s] | self.symptomatic[s])
                frac_inf = float(np.mean(inf_src[bi_m_to_het_f]))
                print(
                    f"[day {self.day}] stable bi_m->het_f edges={n_edges}, mean_p={mean_p:.4g}, inf_src_frac={frac_inf:.3%}")
            else:
                print(f"[day {self.day}] stable bi_m->het_f edges=0")

        return src, dst, p_uv, p_vu

    def _load_or_build_mcp_legacy(self, demo_key: str):
        mcp_params = dict(mcp_mean=0.1, mcp_disp=0.25, kclip=(2, 10), build_mcp_edges_version="v1")
        key = _hash_params({"demo_key": demo_key, "mcp": mcp_params})
        cache_path = f"cache/mcp/mcp_{key}.npz"

        if os.path.exists(cache_path):
            print("Loading MCP network from cache...")
            d = load_npz(cache_path)
            self.mcp_src = d["mcp_src"].astype(np.int64, copy=False)
            self.mcp_dst = d["mcp_dst"].astype(np.int64, copy=False)
            self.mcp_beta_uv = d["mcp_beta_uv"].astype(np.float32, copy=False)
            self.mcp_beta_vu = d["mcp_beta_vu"].astype(np.float32, copy=False)
            return

        print("Creating MCP network...")
        self.mcp_src, self.mcp_dst, self.mcp_beta_uv, self.mcp_beta_vu = self._create_static_mcp_network(mcp_params)

        save_npz(cache_path,
                 mcp_src=self.mcp_src,
                 mcp_dst=self.mcp_dst,
                 mcp_beta_uv=self.mcp_beta_uv,
                 mcp_beta_vu=self.mcp_beta_vu)

    def _normalize_simplified_params(self):
        """Normalize simplified calibration parameters into legacy/internal names.

        Simplified v3 exposes four core composition parameters:
          1. p_bi_m_to_msm
          2. party_core_intensity
          3. social_contact_scale
          4. social_same_sex_assort

        The former community_bridge_scale late-stage social ramp was removed
        (no literature basis, low interpretability, no effect once fixed at 1.0).

        The previous social_male_bias parameter is replaced by
        social_same_sex_assort, which samples nonsexual social targets by
        same-sex assortative mixing rather than directly forcing a high male
        target share. Older aliases are still accepted for backward
        compatibility.
        """
        p = self.params

        # Exposed composition-calibration parameters.
        p.setdefault("p_bi_m_to_msm", 0.66)
        p.setdefault("party_core_intensity", p.get("party_mix_attend_prob", 0.55))
        p.setdefault("social_contact_scale", p.get("social_contact_underascertain_mult", 4.20))
        p.setdefault("social_same_sex_assort", p.get("social_male_bias", p.get("social_target_male_share", 0.75)))

        # Fixed structural assumptions.
        p.setdefault("p_bi_f_to_wsw", 0.75)
        p.setdefault("mix_mf_anal_frac", 0.13)
        p.setdefault("mix_fm_anal_frac", 0.13)
        p.setdefault("party_mix_homo_spill_frac", 0.03)
        # Single shared event-layer close-touch expansion fraction.
        # This replaces older pool-specific party_msm_touch_frac,
        # party_wsw_touch_frac, and party_mix_touch_frac in the baseline.
        p.setdefault("party_touch_frac", p.get("sex_party_touch_frac", 0.20))
        p["sex_party_touch_frac"] = float(p["party_touch_frac"])
        p.setdefault("social_contacts_per_case", 2.4)
        p.setdefault("social_contact_window_days", 21.0)
        p.setdefault("social_p_eff", 0.0063)
        p.setdefault("social_adjust_for_r_sus", True)
        p.setdefault("social_p0_cap", 0.03)
        # CDC-aligned traced-contact monitoring, not mandatory quarantine.
        p.setdefault("MONITORING_DAYS", 21)
        p.setdefault("monitor_sexual_contact_mult", 0.30)
        p.setdefault("monitor_social_contact_mult", 0.70)
        # Table S4/S5 transmission/progression defaults.
        # Prefer direct per-act transmission-risk values because they are
        # clearer in calibration tables. Legacy scale/multiplier inputs are
        # still supported as a fallback for old scripts.
        p.setdefault("sexual_risk_scale", 0.35)
        scale = float(p.get("sexual_risk_scale", 0.35))
        legacy_defaults = {
            "sexual_anal_receptive": 0.80 * scale * float(p.get("sexual_anal_receptive_mult", 1.0)),
            "sexual_anal_insertive": 0.15 * scale * float(p.get("sexual_anal_insertive_mult", 1.0)),
            "sexual_vaginal": 0.10 * scale * float(p.get("sexual_vaginal_mult", 1.0)),
            "sexual_oral": 0.05 * scale * float(p.get("sexual_oral_mult", 1.0)),
        }
        for _risk_name, _default in legacy_defaults.items():
            p.setdefault(_risk_name, float(_default))
            p[_risk_name] = float(np.clip(float(p[_risk_name]), 0.0, 1.0))
        p.setdefault("close_touch", 0.0067)

        # ------------------------------------------------------------------
        # Natural-history parameters (LITERATURE-FIXED, not fitted).
        #   incubation:  lognormal(mu=2.108, sigma=0.254), clip (3,17) d
        #                -> median ~5.6 d (95% CrI 4.3-7.8) exposure->onset
        #                (Madewell ZJ et al. Emerg Infect Dis 2023;29:818-821)
        #   symptom_onset_delay: 1-4 d (rash follows symptoms, CDC clinical)
        #   infectious period: gamma(9, 2.33) mean ~21 d, clip (14,28) d
        #                (CDC ref model 27 d; Netherlands 14-28 d)
        # These may also be supplied explicitly via main.py params.
        # ------------------------------------------------------------------
        p.setdefault("incubation_lognorm_mu", 2.108)
        p.setdefault("incubation_lognorm_sigma", 0.254)
        p.setdefault("incubation_clip", (3, 17))
        p.setdefault("symptom_onset_delay_range", (1, 4))
        p.setdefault("infectious_period_gamma_shape", 9.0)
        p.setdefault("infectious_period_gamma_scale", 2.33)
        p.setdefault("infectious_period_clip", (14, 28))
        # Legacy aliases retained so older functions/scripts still work.
        # NOTE: community_bridge_start_day / community_bridge_ramp_days (the
        # "Los-Angeles late social ramp") were REMOVED in the baseline because
        # the ramp was a low-interpretability, unsupported fitting knob with no
        # effect once community_bridge_scale was fixed at 1.0. See the parameter
        # audit (reference/无依据参数清单与文献范围.md §七).

        # Remove direct forced heterosexual-male boosting in simplified v1.
        # Keep a neutral value for backward compatibility.
        p["unhoused_hetero_male_boost"] = float(p.get("unhoused_hetero_male_boost", 1.0))

        # Backward-compatible mappings used by existing functions.
        p["party_mix_attend_prob"] = float(p["party_core_intensity"])

        # Do not use a sex-specific female attendance multiplier in the MIX pool.
        # Heterosexual spillover is controlled by explicit routing, stable-bridge,
        # mixed-sex risk type, and event close-touch mechanisms rather than a
        # calibrated female-ratio knob.
        p["party_mix_female_mult"] = 1.0  # legacy alias only; not a reported parameter
        p.pop("party_hetero_leak", None)

        # Backward-compatible aliases.  social_target_male_share is no longer
        # used for sampling targets; it is kept only to avoid breaking old logs.
        p["social_contact_underascertain_mult"] = float(p["social_contact_scale"])
        p["social_same_sex_assort"] = float(min(1.0, max(0.0, p["social_same_sex_assort"])))
        p["social_target_male_share"] = float(p.get("social_target_male_share", 0.0))

    def __init__(self, n=100000, params: dict | None = None, seed=0, cache_label=None,
                 policy=None, importations=None, verbose: bool = True):
        self.n = n  ## population size
        self.day = 0  # simulation time‑step (0‑indexed integer)
        self._rng = np.random.default_rng(seed)
        self.params = params or {}
        self._normalize_simplified_params()
        # --- time-varying inputs (policy, importations) -------
        # policy: list[dict] with keys:
        #   start,end,contact_mult,oneoff_mult,party_mult,beta_mult,test_mult,trace_mult
        # importations: list[dict] with keys:
        #   start,end,daily_n   (applied to importation_mask if provided)
        # --- logging/ runtime control
        self.verbose = verbose
        self.policy = policy or []
        self.importations = importations or []
        self.importation_mask = None  # set in main.py if you want importations into a risk group only

        self.print_every = int(self.params.get("print_every", 1))

        # --- time-series buffers
        # base testing / tracing controls
        # Two testing modes are supported:
        #   1) scheduled_delay (recommended): at rash/symptom onset, a case is
        #      assigned a test-seeking delay and then a lab turnaround delay.
        #   2) daily_prob (legacy): each symptomatic person has a daily testing
        #      probability TEST_PROB_SYMP.
        self.test_delay_mode = str(self.params.get("test_delay_mode", "scheduled_delay"))
        self.testing_strategy_version = str(self.params.get("testing_strategy_version", "phase_sensitive_v1"))
        self.base_test_prob_symp = float(self.params.get("TEST_PROB_SYMP", TEST_PROB_SYMP))  # legacy daily mode
        self.test_prob_symp = self.base_test_prob_symp

        self.base_test_pathway_prob_symp = float(self.params.get("test_pathway_prob_symp", 0.85))
        self.test_pathway_prob_symp = self.base_test_pathway_prob_symp
        self.base_test_seek_delay_mean_days = float(self.params.get("test_seek_delay_mean_days", 5.0))
        self.test_seek_delay_mean_days = self.base_test_seek_delay_mean_days
        self.base_test_seek_delay_sd_days = float(self.params.get("test_seek_delay_sd_days", 3.0))
        self.test_seek_delay_sd_days = self.base_test_seek_delay_sd_days
        self.test_seek_delay_max_days = int(self.params.get("test_seek_delay_max_days", 21))
        self.base_lab_turnaround_days = int(self.params.get("LAB_TURNAROUND_DAYS", LAB_TURNAROUND_DAYS))
        self.lab_turnaround_days_today = self.base_lab_turnaround_days

        self.base_trace_prob_edge = float(self.params.get("TRACE_PROB_EDGE", TRACE_PROB_EDGE))
        self.trace_prob_edge = self.base_trace_prob_edge

        # Traced-contact monitoring, not mandatory full quarantine.
        # Monitored asymptomatic contacts remain active but have reduced
        # sexual/high-contact and routine social/close-contact activity.
        self.monitoring_days = int(self.params.get("MONITORING_DAYS", MONITORING_DAYS))
        self.monitor_sexual_contact_mult = float(self.params.get(
            "monitor_sexual_contact_mult", MONITOR_SEXUAL_CONTACT_MULT
        ))
        self.monitor_social_contact_mult = float(self.params.get(
            "monitor_social_contact_mult", MONITOR_SOCIAL_CONTACT_MULT
        ))
        self.monitor_sexual_contact_mult = float(np.clip(self.monitor_sexual_contact_mult, 0.0, 1.0))
        self.monitor_social_contact_mult = float(np.clip(self.monitor_social_contact_mult, 0.0, 1.0))

        # per-layer transmission multipliers (set daily by _apply_policy)
        self.scale_sexual = 1.0
        self.scale_oneoff = 1.0
        self.scale_sex_party = 1.0
        self.scale_household_close = 1.0
        self.scale_social = 1.0
        self.n_days = int(self.params.get("n_days", 300))
        self.daily_new_diagnoses = np.zeros(self.n_days, dtype=np.int32)

        # --- Observation model for surveillance "episode date" -----------
        # LAC dashboard defines episode date as the earliest available of serveral dates
        # (often symptom onset, specimen collection, diagnosis, reoirt recerved, etc.).
        # In the transmission model, everyone has a symotin onset day, but in real data
        # onset may be missing; we model this with p_onset_recorded.
        self.p_onset_recorded = float(self.params.get("p_onset_recorded", 0.8))

        # day of specimen collection / test submission (proxy for "specimen collection date")
        self.specimen_day = np.full(self.n, -1, dtype=np.int32)

        # daily incident cases by "episode date" (total, by sex, by sexual orientation)
        self.daily_episode_cases = np.zeros(self.n_days, dtype=np.int32)

        self.daily_episode_cases_by_sex = np.zeros((self.n_days, 2), dtype=np.int32)  # sex: 0=male,1=female
        self.daily_episode_cases_by_ori = np.zeros((self.n_days, 3), dtype=np.int32)  # ori: 0=hetero,1=bi,2=homo
        # sexcat: 0=female_hetero, 1=male_hetero, 2=male_homo,
        #         3=female_homo,   4=male_bi,     5=female_bi
        # This lets calibration use the same sex-by-orientation denominator as the observed case table.
        self.daily_episode_cases_by_sexcat = np.zeros((self.n_days, 6), dtype=np.int32)

        # precompute orientation codes for fast counting (hetero=0, bi=1, homo=2)
        self.ori_code = np.full(self.n, 0, dtype=np.int8)
        if hasattr(self, "orientations"):
            self.ori_code[self.orientations == "hetero"] = 0
            self.ori_code[self.orientations == "bi"] = 1
            self.ori_code[self.orientations == "homo"] = 2

        # Scratch buffers (allocated lazily to avoid big allocations unless tracing is used)
        self._pos_flag = None
        self._pos_flag_idx = np.empty(0, dtype=np.int64)

        self._seed = seed
        self._cache_label = cache_label or "default"
        # 1) demographics cache
        # ===== DEMOGRAPHICS  (placeholder draws) =========================
        t1 = time.time()
        demo_key = self._load_or_build_demographics()  # <- 改成返回 key
        t2 = time.time()
        print(f"Demographics load/build time: {t2 - t1:.3f}s")

        self._recompute_ori_code()
        # cache indices for dynamic layers (avoids scanning all n every day)
        self.party_idx = np.where(self.sex_party_eligible)[0].astype(np.int64)
        pidx = self.party_idx
        print("party_idx size", pidx.size,
              "hetero", int(np.sum(self.orientations[pidx] == 'hetero')),
              "bi", int(np.sum(self.orientations[pidx] == 'bi')),
              "homo", int(np.sum(self.orientations[pidx] == 'homo')),
              "female", int(np.sum(self.genders[pidx] == 1)))
        if self.verbose:
            m = self.mcp_idx
            print("mcp_idx size", m.size,
                  "male", int(np.sum(self.genders[m] == 0)),
                  "female", int(np.sum(self.genders[m] == 1)),
                  "hetero", int(np.sum(self.orientations[m] == 'hetero')),
                  "bi", int(np.sum(self.orientations[m] == 'bi')),
                  "homo", int(np.sum(self.orientations[m] == 'homo')))
        # --- Precompute membership / pools for the social layer (avoid O(n) scans each day) ---
        # MCP is static in this model, so we can cache masks once.
        self._is_mcp_mask = np.zeros(self.n, dtype=np.bool_)
        self._is_mcp_mask[self.mcp_idx] = True

        # Adults not in MCP (used for MCP→general-population bridging)
        _adult = (self.ages >= 18)
        _nonmcp_adult = np.where(_adult & (~self._is_mcp_mask))[0].astype(np.int32, copy=False)

        # Split by gender once for fast sampling
        self._nonmcp_all_male = _nonmcp_adult[self.genders[_nonmcp_adult] == 0].astype(np.int32, copy=False)
        self._nonmcp_all_female = _nonmcp_adult[self.genders[_nonmcp_adult] == 1].astype(np.int32, copy=False)

        # Often we only want to bridge into heterosexuals; cache a hetero-only pool too (optional)
        _nonmcp_hetero = _nonmcp_adult[
            self.orientations[_nonmcp_adult] == 'hetero'] if _nonmcp_adult.size else _nonmcp_adult
        self._nonmcp_hetero_male = _nonmcp_hetero[self.genders[_nonmcp_hetero] == 0].astype(np.int32, copy=False)
        self._nonmcp_hetero_female = _nonmcp_hetero[self.genders[_nonmcp_hetero] == 1].astype(np.int32, copy=False)

        # MCP split by gender (used if you want to bias male attendance without excluding females)
        self._mcp_male = self.mcp_idx[self.genders[self.mcp_idx] == 0].astype(np.int64, copy=False)
        self._mcp_female = self.mcp_idx[self.genders[self.mcp_idx] == 1].astype(np.int64, copy=False)

        # self.ages = self._rng.integers(0, 90, size=n, dtype=np.int16)
        # Gender_map={"male":0,"female":1} # # 0=male 1=female
        # #generate genders as numeric codes
        # self.genders = self._rng.choice(list(Gender_map.values()), size=n) # half male half female
        #
        #
        # # ---- gender‑specific orientation probabilities  --------------
        # self.orientations = np.empty(n, dtype='<U6')
        # male_mask   = self.genders == 0
        # female_mask = ~male_mask
        # self.orientations[male_mask] = self._rng.choice(
        #     ['hetero', 'bi', 'homo'], size=male_maagesk.sum(),
        #     p=[0.910, 0.039, 0.051])
        # self.orientations[female_mask] = self._rng.choice(
        #     ['hetero', 'bi', 'homo'], size=female_mask.sum(),
        #     p=[0.897, 0.076, 0.027])
        #
        # # in the bi and homo male,19.9% of MSM are exclusively receptive, 31.2% are exclusively insertive, and 48.9% are versatile
        # # 0: receptive; 1: insertive; 2:versatile
        # self.sex_role = np.full(n, -1, dtype=np.int8)
        # sex_role_mask=male_mask & np.isin(self.orientations,["homo","bi"])
        # self.sex_role[sex_role_mask]=self._rng.choice([0,1,2],size=sex_role_mask.sum(),p=[0.199,0.312,0.489])

        self.uids = np.arange(n, dtype=np.int64)
        # ===== SEX‑CATEGORY ENCODING & COMPATIBILITY ==================
        # self._build_sex_categories()

        # ===== DISEASE STATE ARRAYS (bool unless noted) ===============
        # --- disease states --------------------------------------------------
        self.susceptible = np.ones(n, dtype=bool)
        self.exposed = np.zeros(n, dtype=bool)
        self.infectious = np.zeros(n, dtype=bool)
        self.symptomatic = np.zeros(n, dtype=bool)
        self.recovered = np.zeros(n, dtype=bool)
        self.dead = np.zeros(n, dtype=bool)

        self.tested = np.zeros(n, dtype=bool)
        self.diagnosed = np.zeros(n, dtype=bool)
        # Legacy name kept for compatibility; in the final baseline this state
        # represents 21-day monitoring/risk reduction, not mandatory quarantine.
        self.monitored = np.zeros(n, dtype=bool)
        self.quarantined = self.monitored
        self.isolated = np.zeros(n, dtype=bool)

        # progression timer
        self.date_exposed = np.full(n, -1, dtype=np.int32)
        self.incubation_period = np.zeros(n, dtype=np.float32)  # exposure -> rash/symptom onset
        self.symptom_onset_delay = np.zeros(n, dtype=np.float32)  # infectious start -> rash/symptom onset
        self.infectious_period = np.zeros(n, dtype=np.float32)  # how long a person is infectious
        self.infectious_start_day = np.full(n, -1, dtype=np.int32)
        self._seen_day = np.full(n, -1, dtype=np.int32)  # # dedup marker: last day counted as new infection

        # ===== FLAGS ==================================================
        # self.mcp_flags = np.zeros(n, dtype=bool)
        # male = self.genders == 0
        # female = ~male
        # self._gender_uint8 = (self.genders == 1).astype(np.uint8)
        # self._mcp_ids = np.where(self.mcp_flags)[0].astype(np.int64)
        #
        # hetero = self.orientations == 'hetero'
        # bi = self.orientations == 'bi'
        # homo = self.orientations == 'homo'
        # self.mcp_flags[(male & hetero)] = self._rng.random((male & hetero).sum()) < 0.10
        # self.mcp_flags[(male & bi)] = self._rng.random((male & bi).sum()) < 0.25
        # self.mcp_flags[(male & homo)] = self._rng.random((male & homo).sum()) < 0.50
        # self.mcp_flags[(female & hetero)] = self._rng.random((female & hetero).sum()) < 0.06
        # self.mcp_flags[(female & bi)] = self._rng.random((female & bi).sum()) < 0.18
        # self.mcp_flags[(female & homo)] = self._rng.random((female & homo).sum()) < 0.10
        # self.mcp_idx=np.where(self.mcp_flags)[0].astype(np.int64)
        # party_prob = {
        #     (0, 'homo'): 0.40,
        #     (0, 'bi'): 0.10,
        #     (0, 'hetero'): 0.10,
        #     (1, 'homo'): 0.10,
        #     (1, 'bi'): 0.10,
        #     (1, 'hetero'): 0.10,
        # }

        # self.sex_party_eligible = np.zeros(self.n, dtype=bool)
        #
        # for (g, o), prob in party_prob.items():
        #     mask = (self.genders == g) & (self.orientations == o) & self.mcp_flags
        #     # draw one Bernoulli trial per person in this subgroup
        #     self.sex_party_eligible[mask] = self._rng.random(mask.sum()) < prob
        #
        # # ---------- 1 ⋄ transmission‑risk lookup table (per contact) ----
        # self.transmission_risks = {
        #     'sexual_anal_receptive': 0.8,
        #     'sexual_anal_insertive':0.15,
        #     'sexual_vaginal': 0.1,#0.08
        #     'sexual_oral': 0.05,#0.02
        #     'close_touch':0.01,  # NEW – casual / skin‑to‑skin contact at parties
        # }

        # self.same_bias = 4.0  # ← tweak here
        # self.same_bias_small_large = 3.0  # ← M-M or F-F contact is 3× more likely
        # ---------- 2 ⋄ demographic attributes -------------------------

        # draw once with the calibrated probabilities above
        # --- networks --------------------------------------------------------
        # self.hh_src, self.hh_dst, self.hh_beta = self._create_realistic_household_network()
        # self.sch_src, self.sch_dst, self.sch_beta    = self._create_school_network()
        # self.wp_src, self.wp_dst, self.wp_beta = self._create_workplace_network()
        # self.so_src, self.so_dst, self.so_beta  = self._create_social_network()
        # check the time here
        # 3) MCP cache here
        # ------------------------------------------------------------------
        # Table S4/S5 disease-transmission parameters
        # ------------------------------------------------------------------
        # These values are treated as calibrated relative per-contact risk
        # weights, not directly observed mpox per-act probabilities.
        #
        # Rationale:
        #   - Receptive anal exposure is the highest sexual-risk category.
        #   - Insertive anal, mixed-sex vaginal, and oral exposures are lower
        #     relative-risk categories.
        #   - Nonsexual household/party close-touch is anchored near the
        #     CDC-derived nonsexual close-contact effective risk (~0.0063),
        #     rounded here to 0.0067 for the household/party touch layer.
        #
        # Direct per-act values are easier to report and calibrate than
        # decomposed scale/multiplier expressions.
        close_touch_risk = float(self.params.get("close_touch", 0.0067))

        self.transmission_risks = {
            # Calibrated sexual-risk weights, reported directly.
            'sexual_anal_receptive': float(self.params.get("sexual_anal_receptive", 0.2800)),
            'sexual_anal_insertive': float(self.params.get("sexual_anal_insertive", 0.0525)),
            'sexual_vaginal': float(self.params.get("sexual_vaginal", 0.0350)),
            'sexual_oral': float(self.params.get("sexual_oral", 0.0175)),

            # Household/party nonsexual close-touch risk.
            # The source-based social layer still uses social_p_eff separately.
            'close_touch': close_touch_risk,
        }
        self.same_bias = 4.0  # ← tweak here
        self.same_bias_small_large = 3.0  # ← M-M or F-F contact is 3× more likely

        # in __init__
        self.txlog_max = int(self.params.get("txlog_max", 200000))
        self.txlog_i = 0
        self.txlog_day = np.empty(self.txlog_max, dtype=np.int32)
        self.txlog_layer = np.empty(self.txlog_max, dtype=np.int16)  # 例如: 1=stable,2=party,3=oneoff...
        self.txlog_src = np.empty(self.txlog_max, dtype=np.int64)
        self.txlog_dst = np.empty(self.txlog_max, dtype=np.int64)
        self.txlog_p = np.empty(self.txlog_max, dtype=np.float32)
        # 3) MCP cache here
        t3 = time.time()
        self._load_or_build_household_partners(demo_key)
        t4 = time.time()
        print(f"Household partner network load/build time: {t4 - t3:.3f} seconds")

        # ---------- 1 ⋄ transmission‑risk lookup table (per contact) ----

        # -----temporary per-day flag arrays --------------------
        self.dead_today = np.zeros(n, dtype=bool)  # people who died today
        self.new_exposed = np.zeros(n, dtype=bool)  # newly exposed today
        self.new_infectious = np.zeros(n, dtype=bool)  # newly infectious today
        self.new_symptomatic = np.zeros(n, dtype=bool)  # newly symptomatic today
        self.symptom_onset_day = np.full(n, -1, dtype=np.int32)
        self.new_recovered = np.zeros(n, dtype=bool)  # newly recovered today
        self.new_diagnosed = np.zeros(n, dtype=bool)  # newly diagnosed today
        self.new_vaccinated = np.zeros(n, dtype=bool)  # newly vaccinated today
        self.new_quarantined = np.zeros(n, dtype=bool)  # newly quarantined today
        self.new_isolated = np.zeros(n, dtype=bool)  # newly isolated today

        # # daily sex‑party attendance split
        self.sex_party_daily = {'MSM': [], 'WSW': [], 'MIX': []}
        # placeholders for mean-contact diagnostics
        self.sex_src_today = np.empty(0, np.int64)
        self.sex_dst_today = np.empty(0, np.int64)
        self.sp_src_today = np.empty(0, np.int64)
        self.sp_dst_today = np.empty(0, np.int64)
        self.hh_close_src_today = np.empty(0, np.int64)
        self.hh_close_dst_today = np.empty(0, np.int64)
        self.soc_src_today = np.empty(0, np.int64)
        self.soc_dst_today = np.empty(0, np.int64)
        # # --- age‑based susceptibility & symptom probability -----------------
        self.r_sus = np.ones(n, dtype=np.float32)  # relative susceptibility
        self.p_sym = np.ones(n, dtype=np.float32)  # probability symptomatic
        self.p_dea = np.zeros(n, dtype=np.float32)  # probability of death given infection (small)
        self._set_age_based_progression()
        self._r_sus_base = self.r_sus.copy()

        # event schedules: map day → list of agent IDs
        self._dose1_eff = {}  # effective first‐dose events
        self._dose2_eff = {}  # effective second‐dose events

        self.layers = [  # 'household', #'school', 'workplace', 'social',
            'stable', 'household_close', 'social', 'oneoff', 'sex_party', 'sex_party_touch', 'small_event',
            'large_event']  # ← add the last one
        self.daily_stats = []
        #
        self.layer_infections = {layer: [] for layer in self.layers}
        self.layer_infection_counts = {layer: 0 for layer in self.layers}
        self.infections_by_day = {layer: [0] for layer in self.layers}

        # --- Effective reproduction number (R_t) tracking ---
        # we estimate R_t using the Cori method:
        #   R_t = I_t / sum_{s>=1} I_{t-s} * w_s
        # where w_s is the (discretized) generation-interval PMF.

        # Notes on indexing:
        # - infections_by_day["total"] is appended once per simulated day (no leading 0),
        #   so rt_by_day["total"] will have the same length and uses the same day index.
        self.gi_mean = float(params.get("gi_mean", 9.0))
        self.gi_sd = float(params.get("gi_sd", 3.0))
        self.gi_max = int(params.get("gi_max", 20))

        ### ---R_t controls (important for early stability)---
        self.rt_smooth_window = int(params.get("rt_smooth_window", 7))  # τ
        self.rt_burn_in = int(params.get("rt_burn_in", 10))  # ignore early days
        self.rt_prior_a = float(params.get("rt_prior_a", 1.0))  # Gamma prior shape
        self.rt_prior_b = float(params.get("rt_prior_b", 1.0))  # Gamma prior rate
        self.rt_min_cases = int(params.get("rt_min_cases", 20))  # avoid unstable early Rt
        self.gi_pmf = self._make_gamma_gi_pmf(self.gi_mean, self.gi_sd, self.gi_max)

        # R_t per day (aligned to infections_by_day["total"])
        self.rt_by_day = {"total": []}
        self._rt_lambda_by_day = {"total": []}  # force of infection term used in denominator

        self.gender_orientation_stats = []
        # track daily totals + orientation-specific incidence
        self.infections_by_day["total"] = [0]  # overall new infections per day (aligned to day index)
        self.orientation_daily = {"hetero": [0], "bi": [0], "homo": [0]}  # new infections by orientation

        # Additional daily incidence trackers used ONLY for calibration diagnostics.
        # They do not affect transmission.  sexcat encoding is defined in
        # _build_sex_categories():
        #   0 = heterosexual female, 1 = heterosexual male,
        #   2 = homosexual male/MSM, 3 = homosexual female/WSW,
        #   4 = bisexual male, 5 = bisexual female.
        self.sex_daily = {"male": [0], "female": [0]}
        self.sexcat_daily = {
            "female_hetero": [0],
            "male_hetero": [0],
            "male_homo": [0],
            "female_homo": [0],
            "male_bi": [0],
            "female_bi": [0],
        }

        # -------vaccination coverage --------------------------------
        self.vax_dose = np.zeros(n, dtype=np.uint8)
        self.vax1_day = np.full(n, -999, dtype=np.int32)
        self.vax2_day = np.full(n, -999, dtype=np.int32)

        self._first_cov = np.array([
            0.0000, 0.0001, 0.0002, 0.0003,
            0.0011, 0.0025, 0.0057, 0.0153,
            0.0375, 0.0713, 0.1217, 0.1860,
            0.2380, 0.2826, 0.3174, 0.3401,
            0.3623, 0.3789, 0.3922, 0.4025,
            0.4108, 0.4180, 0.4237, 0.4283,
            0.4323, 0.4358, 0.4376, 0.4402,
            0.4423, 0.4438, 0.4438, 0.4438
        ], dtype=np.float32)
        # Weekly second-dose coverage
        self._second_cov = np.array([
            0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0001, 0.0002,
            0.0005, 0.0010, 0.0019, 0.0047,
            0.0124, 0.0279, 0.0507, 0.0811,
            0.1162, 0.1539, 0.1814, 0.2006,
            0.2157, 0.2276, 0.2370, 0.2440,
            0.2501, 0.2558, 0.2583, 0.2626,
            0.2656, 0.2677, 0.2678, 0.2678
        ], dtype=np.float32)

        # -- testing, tracing, and monitoring state arrays --------------
        self.test_pending_day = np.full(n, -1, dtype=np.int32)  # day when test result is expected
        self.isolated = np.zeros(n, dtype=bool)
        self.iso_end_day = np.full(n, -1, dtype=int)
        # Traced contacts enter monitoring/risk reduction.
        # self.quarantined is an alias retained for backward compatibility.
        self.monitored = np.zeros(n, dtype=bool)
        self.quarantined = self.monitored
        self.monitor_end_day = np.full(n, -1, dtype=int)
        self.quar_end_day = self.monitor_end_day

        self._tt_daily = []  # for plotting test/tracing stats

        # edges available for next-day contact tracing (built in step())
        self._trace_src_prev = np.empty(0, np.int64)
        self._trace_dst_prev = np.empty(0, np.int64)

    def _sample_test_seeking_delay(self, size: int,
                                    mean_days: float | None = None,
                                    sd_days: float | None = None,
                                    max_days: int | None = None) -> np.ndarray:
        """Sample integer days from rash/symptom onset to specimen/test seeking.

        The scheduled-delay testing model separates:
          rash/symptom onset -> specimen collection / test seeking
          specimen collection -> laboratory result

        Delay is sampled from a gamma distribution parameterized by the
        currently active policy phase mean and standard deviation.  Values are
        rounded up to integer days and capped to avoid implausible long tails.
        Optional arguments allow _schedule_tests_for_symptomatic() to pass an
        explicit phase-specific testing strategy while preserving the existing
        object attributes as defaults.
        """
        if size <= 0:
            return np.empty(0, dtype=np.int32)

        if mean_days is None:
            mean_days = getattr(self, "test_seek_delay_mean_days", 5.0)
        if sd_days is None:
            sd_days = getattr(self, "test_seek_delay_sd_days", 3.0)
        if max_days is None:
            max_days = getattr(self, "test_seek_delay_max_days", 21)

        mean = max(float(mean_days), 0.0)
        sd = max(float(sd_days), 0.0)
        max_days = max(int(max_days), 0)

        if mean <= 0.0:
            delay = np.zeros(size, dtype=np.int32)
        elif sd <= 1e-8:
            delay = np.full(size, int(round(mean)), dtype=np.int32)
        else:
            shape = (mean / sd) ** 2
            scale = (sd ** 2) / mean
            delay = np.ceil(self._rng.gamma(shape, scale, size=size)).astype(np.int32)

        if max_days > 0:
            delay = np.minimum(delay, max_days).astype(np.int32, copy=False)
        return delay

    def _schedule_tests_for_symptomatic(self, cand_idx: np.ndarray,
                                        test_pathway_prob: float | None = None,
                                        test_seek_delay_mean_days: float | None = None,
                                        test_seek_delay_sd_days: float | None = None,
                                        lab_turnaround_days: int | None = None) -> np.ndarray:
        """Schedule testing for symptomatic/rash-onset cases using staged delays.

        A symptomatic case first enters the testing pathway with phase-specific
        probability ``test_pathway_prob``.  If selected, we assign:
          specimen_day = current day + test_seeking_delay
          result_day   = specimen_day + lab_turnaround_days

        ``self.test_pending_day`` stores the result day.  ``self.tested`` is
        used as a one-time scheduling flag so the same symptomatic case is not
        repeatedly rescheduled.  If optional phase arguments are omitted, the
        attributes already set by _apply_policy() are used.
        """
        if cand_idx is None or cand_idx.size == 0:
            return np.empty(0, dtype=np.int64)

        cand_idx = cand_idx.astype(np.int64, copy=False)
        eligible = (
            self.symptomatic[cand_idx]
            & (self.test_pending_day[cand_idx] < 0)
            & (~self.isolated[cand_idx])
            & (~self.diagnosed[cand_idx])
            & (~self.tested[cand_idx])
        )
        cand_idx = cand_idx[eligible]
        if cand_idx.size == 0:
            return np.empty(0, dtype=np.int64)

        if test_pathway_prob is None:
            test_pathway_prob = getattr(self, "test_pathway_prob_symp", 0.85)
        p_enter = float(np.clip(float(test_pathway_prob), 0.0, 1.0))
        take = self._rng.random(cand_idx.size) < p_enter
        take_idx = cand_idx[take]
        if take_idx.size == 0:
            # Mark non-test-seekers as tested/scheduled to avoid repeated draws.
            # This keeps test_pathway_prob_symp interpretable as a one-time
            # probability of entering the testing pathway after rash onset.
            self.tested[cand_idx] = True
            return np.empty(0, dtype=np.int64)

        # Also mark all candidates as processed once, including those who do
        # not enter the testing pathway.
        self.tested[cand_idx] = True

        seek_delay = self._sample_test_seeking_delay(
            take_idx.size,
            mean_days=test_seek_delay_mean_days,
            sd_days=test_seek_delay_sd_days,
            max_days=getattr(self, "test_seek_delay_max_days", 21),
        )
        if lab_turnaround_days is None:
            lab_turnaround_days = getattr(self, "lab_turnaround_days_today", LAB_TURNAROUND_DAYS)
        lab_delay = max(int(lab_turnaround_days), 0)

        specimen_day = (self.day + seek_delay).astype(np.int32, copy=False)
        result_day = (specimen_day + lab_delay).astype(np.int32, copy=False)

        self.specimen_day[take_idx] = specimen_day
        self.test_pending_day[take_idx] = result_day
        return take_idx

    def _run_test_trace(self):
        d = self.day
        rng = self._rng

        # (optional) policy multiplier if you have it
        policy = getattr(self, "_policy_today", None) or {}
        trace_mult = float(policy.get("trace_mult", 1.0))

        # 1) release finished isolation / monitoring
        done_iso = self.isolated & (d >= self.iso_end_day)
        done_monitor = self.monitored & (d >= self.monitor_end_day)
        self.isolated[done_iso] = False
        self.monitored[done_monitor] = False

        # 2) Testing model
        if self.test_delay_mode == "daily_prob":
            # Legacy mode: every eligible symptomatic case has a daily probability
            # of testing, followed by lab turnaround.
            need_test = (
                    (self.symptomatic) &
                    (d >= (self.symptom_onset_day + 1)) &
                    (self.test_pending_day < 0) &
                    (~self.isolated) &
                    (~self.tested)
            )
            need_idx = np.where(need_test)[0]
            if need_idx.size:
                take = rng.random(need_idx.size) < self.test_prob_symp
                take_test_idx = need_idx[take]
                if take_test_idx.size:
                    self.specimen_day[take_test_idx] = d
                    self.test_pending_day[take_test_idx] = d + int(self.lab_turnaround_days_today)
                    self.tested[take_test_idx] = True
            else:
                take_test_idx = np.empty(0, np.int64)
        else:
            # Recommended mode: direct staged delay from rash/symptom onset to
            # specimen/test seeking, followed by lab turnaround.  This fallback
            # schedules any already symptomatic but unscheduled cases.  Newly
            # symptomatic cases are also scheduled immediately after symptom onset
            # is generated later in step().
            need_test = (
                    (self.symptomatic) &
                    (self.test_pending_day < 0) &
                    (~self.isolated) &
                    (~self.diagnosed) &
                    (~self.tested)
            )
            need_idx = np.where(need_test)[0]
            take_test_idx = self._schedule_tests_for_symptomatic(
                need_idx,
                test_pathway_prob=self.test_pathway_prob_symp,
                test_seek_delay_mean_days=self.test_seek_delay_mean_days,
                test_seek_delay_sd_days=self.test_seek_delay_sd_days,
                lab_turnaround_days=self.lab_turnaround_days_today,
            )

        # 3) return results today (imperfect sens/spec)
        got_idx = np.where(self.test_pending_day == d)[0]
        if got_idx.size:
            true_inf = self.infectious[got_idx] | self.exposed[got_idx] | self.symptomatic[got_idx]
            r = rng.random(got_idx.size)
            pos_mask = (true_inf & (r < TEST_SENS)) | ((~true_inf) & (r < (1.0 - TEST_SPEC)))
            pos_idx = got_idx[pos_mask]
            neg_idx = got_idx[~pos_mask]

            # isolate positives
            if pos_idx.size:
                newly = ~self.diagnosed[pos_idx]
                if newly.any():
                    new_idx = pos_idx[newly]
                    self.new_diagnosed[new_idx] = True
                    if d < self.daily_new_diagnoses.shape[0]:
                        self.daily_new_diagnoses[d] += int(newly.sum())

                        # --- episode-date counting (observation layer) -------------------
                        p = self.p_onset_recorded
                        onset = self.symptom_onset_day[new_idx].astype(np.int32, copy=False)
                        specimen = self.specimen_day[new_idx].astype(np.int32, copy=False)
                        specimen = np.where(specimen >= 0, specimen, d).astype(np.int32, copy=False)

                        onset_rec = (rng.random(new_idx.size) < p) & (onset >= 0)
                        ep = np.minimum(specimen, d)
                        ep = np.where(onset_rec, np.minimum(ep, onset), ep).astype(np.int32, copy=False)
                        ep = np.clip(ep, 0, self.n_days - 1)

                        cnt = np.bincount(ep, minlength=self.n_days).astype(np.int32)
                        self.daily_episode_cases[:self.n_days] += cnt

                        sex = self.genders[new_idx].astype(np.int8, copy=False)
                        for s in (0, 1):
                            msk = (sex == s)
                            if np.any(msk):
                                cs = np.bincount(ep[msk], minlength=self.n_days).astype(np.int32)
                                self.daily_episode_cases_by_sex[:self.n_days, s] += cs

                        oc = self.ori_code[new_idx].astype(np.int8, copy=False)
                        for k in (0, 1, 2):
                            msk = (oc == k)
                            if np.any(msk):
                                ck = np.bincount(ep[msk], minlength=self.n_days).astype(np.int32)
                                self.daily_episode_cases_by_ori[:self.n_days, k] += ck

                        # Sex-by-orientation episode-date cases for calibration against
                        # observed columns such as Male-Homo, Male-Bisexual,
                        # Male-Heterosexual, and Female-Heterosexual.
                        sc = self.sexcat[new_idx].astype(np.int8, copy=False)
                        for k in (0, 1, 2, 3, 4, 5):
                            msk = (sc == k)
                            if np.any(msk):
                                ck = np.bincount(ep[msk], minlength=self.n_days).astype(np.int32)
                                self.daily_episode_cases_by_sexcat[:self.n_days, k] += ck

                self.isolated[pos_idx] = True
                self.iso_end_day[pos_idx] = d + ISOLATION_DAYS
                self.diagnosed[pos_idx] = True

            # clear pending flags for everyone who got results
            self.test_pending_day[got_idx] = -1
        else:
            pos_idx = np.empty(0, np.int64)
            neg_idx = np.empty(0, np.int64)

        # 4) trace contacts of positives into monitoring/risk reduction (using PREVIOUS day edge lists) ──────────
        base = float(self.params.get("TRACE_PROB_EDGE", TRACE_PROB_EDGE))
        by_layer = self.params.get("TRACE_PROB_EDGE_BY_LAYER", {}) or {}

        # layer_code -> name（与你的 step() 存储一致）
        LAYER_NAME = {
            0: "household_close",
            1: "stable",
            2: "oneoff",
            3: "sex_party",
            4: "sex_party_touch",
            5: "social",
        }

        def _trace_prob_for_layer(layer_code: int) -> float:
            name = LAYER_NAME.get(int(layer_code), None)
            p = by_layer.get(name, base)
            p = float(p) * trace_mult
            if p <= 0.0:
                return 0.0
            if p >= 1.0:
                return 1.0
            return p

        # nothing to trace
        if pos_idx.size:
            # build / reuse pos flag (O(m) membership)
            if (getattr(self, "_pos_flag", None) is None) or (self._pos_flag.size != self.n):
                self._pos_flag = np.zeros(self.n, dtype=bool)
                self._pos_flag_idx = np.empty(0, dtype=np.int64)
            else:
                if self._pos_flag_idx.size:
                    self._pos_flag[self._pos_flag_idx] = False

            self._pos_flag[pos_idx] = True
            self._pos_flag_idx = pos_idx

            traced_any = 0

            # prefer per-layer dict; fallback to aggregated if missing
            trace_prev_by_layer = getattr(self, "_trace_prev_by_layer", None)

            if isinstance(trace_prev_by_layer, dict) and trace_prev_by_layer:
                for layer_code, (src_prev, dst_prev) in trace_prev_by_layer.items():
                    if src_prev is None or src_prev.size == 0:
                        continue

                    p_layer = _trace_prob_for_layer(int(layer_code))
                    if p_layer <= 0.0:
                        continue

                    # undirected expansion but avoid big concat:
                    # contacts where src is pos -> dst, AND dst is pos -> src
                    c1 = dst_prev[self._pos_flag[src_prev]]
                    c2 = src_prev[self._pos_flag[dst_prev]]
                    if c1.size == 0 and c2.size == 0:
                        continue

                    contacts = c1 if c2.size == 0 else (c2 if c1.size == 0 else np.concatenate([c1, c2]))
                    if contacts.size == 0:
                        continue

                    contacts = np.unique(contacts)
                    keep = ~(self.isolated[contacts] | self.monitored[contacts] | self.diagnosed[contacts])
                    contacts = contacts[keep]
                    if contacts.size == 0:
                        continue

                    draw = rng.random(contacts.size) < p_layer
                    traced_contacts = contacts[draw]
                    if traced_contacts.size:
                        self.monitored[traced_contacts] = True
                        self.monitor_end_day[traced_contacts] = d + self.monitoring_days
                        traced_any += int(traced_contacts.size)

            else:
                # fallback to your old aggregated approach (single probability = base * trace_mult)
                p_layer = min(1.0, max(0.0, base * trace_mult))
                src_prev = getattr(self, "_trace_src_prev", np.empty(0, np.int64))
                dst_prev = getattr(self, "_trace_dst_prev", np.empty(0, np.int64))
                if src_prev.size and p_layer > 0.0:
                    all_src = np.concatenate([src_prev, dst_prev])
                    all_dst = np.concatenate([dst_prev, src_prev])
                    contacts = all_dst[self._pos_flag[all_src]]
                    if contacts.size:
                        contacts = np.unique(contacts)
                        keep = ~(self.isolated[contacts] | self.monitored[contacts] | self.diagnosed[contacts])
                        contacts = contacts[keep]
                        if contacts.size:
                            draw = rng.random(contacts.size) < p_layer
                            traced_contacts = contacts[draw]
                            if traced_contacts.size:
                                self.monitored[traced_contacts] = True
                                self.monitor_end_day[traced_contacts] = d + self.monitoring_days
                                traced_any += int(traced_contacts.size)

        # 5) daily bookkeeping
        self._tt_daily.append({
            'day': d,
            'tests': int(take_test_idx.size),
            'pos': int(pos_idx.size),
            'neg': int(neg_idx.size),
            'iso': int(self.isolated.sum()),
            'monitor': int(self.monitored.sum()),
        })

    def _build_sex_categories(self):
        # ----------  sex-orientation category (0…5)  ------------------------
        sexcat = np.full(self.n, 99, dtype=np.uint8)  # 99 = sentinel / error

        sexcat[(self.genders == 1) & (self.orientations == 'hetero')] = 0  # FH: female heterosexual
        sexcat[(self.genders == 0) & (self.orientations == 'hetero')] = 1  # MH: male heterosexual
        sexcat[(self.genders == 0) & (self.orientations == 'homo')] = 2  # MSM
        sexcat[(self.genders == 1) & (self.orientations == 'homo')] = 3  # WSW
        sexcat[(self.genders == 0) & (self.orientations == 'bi')] = 4  # MB: male bisexual
        sexcat[(self.genders == 1) & (self.orientations == 'bi')] = 5  # FB: female bisexual

        assert (sexcat != 99).all(), "unmapped gender/orientation combo"
        self.sexcat = sexcat  # uint8, shape (n,)

        # compatibility matrx that implements rule
        compat = np.zeros((6, 6), dtype=bool)
        # ---- heterosexual pair  -------------------------------------------
        compat[0, 1] = compat[1, 0] = True  # FH ↔ MH

        # ---- homosexual pairs ---------------------------------------------
        compat[2, 2] = True  # MSM
        compat[3, 3] = True  # WSW

        # ---- bisexual with allowed partners -------------------------------
        # male-bi (MB = 4)
        compat[4, 0] = compat[0, 4] = True  # MB ↔ FH   (opp-gender hetero)
        compat[4, 2] = compat[2, 4] = True  # MB ↔ MSM  (same-gender homo)
        compat[4, 4] = True  # MB ↔ MB
        compat[4, 5] = compat[5, 4] = True  # MB ↔ FB

        # female-bi (FB = 5)
        compat[5, 1] = compat[1, 5] = True  # FB ↔ MH   (opp-gender hetero)
        compat[5, 3] = compat[3, 5] = True  # FB ↔ WSW  (same-gender homo)
        compat[5, 5] = True  # FB ↔ FB

        self.compat = compat  # shape (6, 6) boolean

        self._compat_flat = self.compat.ravel()  # length 36 (row-major)

    # vaccination data
    # ------------------------------------------------------------------
    #  Isolation / quarantine helper
    # ------------------------------------------------------------------
    def _inactive(self, idx: int) -> bool:
        """
        Return True if person *idx* must not transmit today.

        In the final baseline, traced asymptomatic contacts are monitored for
        21 days with partial risk reduction rather than removed from all
        contacts. Therefore only confirmed/symptomatic isolated cases are fully
        inactive.
        """
        return self.isolated[idx]

    # ------------------------------------------------------------------
    #  Behaviour-change helpers
    # ------------------------------------------------------------------
    def _downsample_edges(self, G: nx.Graph, keep_frac: float) -> None:
        """Randomly delete (1-keep_frac) of edges in *G* in-place."""
        if keep_frac >= 1.0 or G.number_of_edges() == 0:
            return
        edges = list(G.edges())
        to_remove = int(len(edges) * (1.0 - keep_frac))
        idx_remove = np.random.choice(len(edges), to_remove, replace=False)
        for idx in idx_remove:
            u, v = edges[idx]
            if G.has_edge(u, v):  # double-check (may have gone already)
                G.remove_edge(u, v)

    def _sexual_scale_today(self) -> float:
        """
        Gradual contact-reduction schedule
        ----------------------------------
        • day   0 ............. 100 % (scale = 1.00)
        • day  90 .............  75 % (scale = 0.75)
        • day >90 .............  75 %

        If your t = 0 is not 30 May 2022, shift the day numbers
        accordingly (e.g. 90 → your desired “switch-over” day).
        """
        if self.day >= 90:
            return 0.75  # floor value after day 90
        else:
            # linear interpolation: 1.00 → 0.75 over the first 90 days
            return 1.0 - 0.25 * (self.day / 90.0)

    def count_sex_types(network, pop):
        counts = {"anal": 0, "vaginal": 0, "oral": 0, "unknown": 0}
        for u, v in network.edges():
            sex_type = pop._get_sex_type(u, v)
            if sex_type in counts:
                counts[sex_type] += 1
            else:
                counts["unknown"] += 1
        total = sum(counts.values())
        for k in counts:
            counts[k] = round(100 * counts[k] / total, 2) if total > 0 else 0
        return counts

    def _get_sex_type(self, i: int, j: int) -> str:
        g1, g2 = self.genders[[i, j]]
        if g1 == 0 and g2 == 0:      return 'anal'
        if g1 != g2:                         return 'vaginal'
        return 'oral'

    # ------------------------------------------------------------------
    # Probability that two people will have a *non-sexual* close contact
    # at BAR / LARGE events, given their sex-orientation combo.
    # ------------------------------------------------------------------
    def _close_touch_prob(self, i: int, j: int) -> float:
        """Return P(close touch) between person i and j at a *non-sexual* event."""
        g1, g2 = self.genders[i], self.genders[j]
        o1, o2 = self.orientations[i], self.orientations[j]

        same_sex = (g1 == g2)

        # --- tuning knobs -------------------------------------------------
        p_same = 0.30  # baseline for same-sex friendship hugging etc.
        p_opp = 0.10  # opposite-sex casual touch is rarer
        # If either participant is bisexual (or homosexual), we assume no bias
        # between same- vs opposite-sex touching – use the higher rate.
        if ('bi' in (o1, o2)) or ('homo' in (o1, o2)):
            return p_same  # equal chance regardless of sex-pair
        # Both hetero
        return p_same if same_sex else p_opp

    def _maybe_add_edge(self, G, i, j, base_p: float, trans_key: str):
        """
        Add an edge i–j with probability `base_p` * bias,
        """
        if G.has_edge(i, j):
            return  # already present

        # bias = 1   for same-sex      (or any rule you prefer)
        # bias = 1/3 for opposite-sex  (example value)
        bias = self.same_bias if self.genders[i] == self.genders[j] else 1.0
        if np.random.rand() < base_p * bias:
            G.add_edge(i, j, trans_prob=self.transmission_risks[trans_key])

    def _maybe_add_edge_sex(self, G, i, j, base_p: float, trans_key: str):
        """
        Add an edge i–j with probability `base_p` * bias,
        but only if i ↔ j are orientation-compatible.
        """
        if G.has_edge(i, j):
            return  # already present
        if not self._is_partner_compatible(i, j):
            return  # e.g. hetero–hetero same-sex

        if np.random.rand() < base_p:  # single Bernoulli draw
            G.add_edge(i, j,
                       trans_prob=self.transmission_risks[trans_key])

    def _split_bisexual_partners(self, uid: int, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        For a bisexual person `uid` return two arrays:
        same-sex compatible & opposite-sex compatible partners.
        """
        g = self.genders[uid]
        ss = candidates[self.genders[candidates] == g]  # same sex
        opp = candidates[self.genders[candidates] != g]  # opposite sex
        return ss, opp

    # ------------------------------------------------------------------
    # Orientation‑compatibility logic (same as used in sexual network)
    # ------------------------------------------------------------------
    def _is_partner_compatible(self, i: int, j: int) -> bool:
        """Compatibility check using only (sexcat, compat).

        NOTE: This intentionally replaces any orientation-string / role-based branching
        to keep the logic consistent across layers and speed up checks.
        """

        return bool(self.compat[self.sexcat[i], self.sexcat[j]])

    def _is_touch_compatible(self, i, j) -> bool:
        """True if the pair passes orientation and age‑gap constraints."""
        # orientation
        g1, g2 = self.genders[i], self.genders[j]
        o1, o2 = self.orientations[i], self.orientations[j]
        orient_ok = (
                (o1 == 'hetero' and o2 == 'hetero' and g1 != g2) or  # heterosexual
                (o1 == 'homo' and o2 in ('homo', 'bi') and g1 == g2) or
                (o1 == 'bi') or (o2 == 'bi')
        )
        # age
        age_ok = abs(self.ages[i] - self.ages[j]) <= 10
        return orient_ok and age_ok

    def print_mean_contacts_per_layer(self):
        """
        Compute the *current* mean degree for every layer by using the
        src/dst arrays that drive transmission.  Works for the refactored,
        NetworkX-free model.
        """
        layers = {
            # 'household': (self.hh_src, self.hh_dst),
            # 'school': (self.sch_src, self.sch_dst),
            # 'workplace': (self.wp_src, self.wp_dst),
            # 'social': (self.so_src, self.so_dst),
            'stable': (self.sex_src_today, self.sex_dst_today),  # see below
            'household_close': (self.hh_close_src_today, self.hh_close_dst_today),
            'social': (getattr(self, 'soc_src_today', np.empty(0, np.int64)),
                       getattr(self, 'soc_dst_today', np.empty(0, np.int64))),
            'sex_party': (self.sp_src_today, self.sp_dst_today),  # see below
        }

        print("Mean daily contacts per person per layer:")
        for name, (src, dst) in layers.items():
            if src.size == 0:
                print(f"  {name:<10}: 0 (no contacts)")
                continue

            # degree = how many times an id appears in src or dst
            deg = np.bincount(
                np.concatenate([src, dst]),
                minlength=self.n
            )
            print(f"  {name:<10}: {deg.mean():.2f}")

    def _set_age_based_progression(self):
        """Set Table S4/S5 age-linked susceptibility, symptomatic probability,
        and mortality parameters.

        Interpretation
        --------------
        r_sus(age)
            Relative susceptibility proxy.  Values below 1.0 among older
            cohorts are an operational proxy for residual historical smallpox
            vaccination protection.  These should be reported as scenario
            assumptions and sensitivity-tested, not as directly observed
            mpox susceptibility estimates.

        p_sym(age)
            Age-linked symptomatic/progression/ascertainment proxy.

        p_dea(age)
            Age-linked death probability after infection.  Deaths are rare in
            the clade IIb 2022 outbreak, so these values do not drive fitting.
        """
        age_bins = [
            (0, 29),
            (30, 39),
            (40, 45),
            (46, 50),
            (51, 59),
            (60, 69),
            (70, 120),
        ]

        # Relative susceptibility proxy.
        sus_vals = [1.00, 1.00, 1.00, 0.28, 0.33, 0.38, 0.43]

        # Probability of symptomatic/progressed infection.
        sym_vals = [0.85, 0.85, 0.85, 0.50, 0.55, 0.60, 0.65]

        # Death probability after infection.
        dea_vals = [0.00005, 0.00010, 0.00010, 0.00010, 0.00010, 0.002, 0.005]

        for (lo, hi), rs, ps, pd in zip(age_bins, sus_vals, sym_vals, dea_vals):
            idx = (self.ages >= lo) & (self.ages <= hi)
            self.r_sus[idx] = np.float32(rs)
            self.p_sym[idx] = np.float32(ps)
            self.p_dea[idx] = np.float32(pd)

    def simulate_transmission_by_layer(self, network, layer_name):
        """
        1) Ensure every sexual or sex_party edge has a correct `trans_prob`.
        2) Walk each edge once, infecting susceptible nodes.
        3) For each newly exposed, stamp their timers exactly once.
        4) Update layer‐level counters.
        """
        # determine condom uptake & efficacy by layer
        if layer_name in ('stable', "oneoff"):
            condom_rate = 0.30  # 30% of sexual encounters use condoms in stable and one-off layers
        elif layer_name == 'sex_party':
            condom_rate = 0.50
        else:
            condom_rate = 0.0

        condom_efficacy = 0.50  # reduce transmission by 50% when used

        # 1) fix-up any missing trans_prob on sexual layers
        if layer_name in ['stable', 'sex_party']:
            for u, v in network.edges():
                e = network[u][v]
                if 'trans_prob' not in e:
                    sex_type = self._get_sex_type(u, v)
                    key = f"sexual_{sex_type}"
                    print(f"warning: edge ({u}, {v}) missing trans_prob, assigning based on sex_type")
                    e['trans_prob'] = self.transmission_risks.get(key, 0.01)

        new_infections = 0
        newly_exposed = []  # collect everyone who flips exposed now

        # 2) traverse edges once
        for u, v, attr in network.edges(data=True):
            # skip contacts if either side is currently inactive
            if self._inactive(u) or self._inactive(v):
                continue
            based_p = attr['trans_prob']

            # apply condom reduction if this is a sexual layer
            if layer_name in ("sexual", "sex_party", "oneoff"):
                if np.random.rand() < condom_rate:
                    trans_p = based_p * (1 - condom_efficacy)
                else:
                    trans_p = based_p
            else:
                trans_p = based_p

            # u → v
            if self.infectious[u] and self.susceptible[v] and not self.dead[v]:
                if np.random.rand() < trans_p * self.r_sus[v]:
                    self.susceptible[v] = False
                    self.exposed[v] = True
                    newly_exposed.append(v)
                    new_infections += 1
            # v → u
            elif self.infectious[v] and self.susceptible[u] and not self.dead[u]:
                if np.random.rand() < trans_p * self.r_sus[u]:
                    self.susceptible[u] = False
                    self.exposed[u] = True
                    newly_exposed.append(u)
                    new_infections += 1

        # 3) stamp timers *once* on everyone just exposed
        for idx in newly_exposed:
            self.date_exposed[idx] = self.day
            # Table S4/S5: exposure-to-rash/incubation-like timer.
            # Lognormal(mu=2.108, sigma=0.254), clipped to 3-17 days.
            # In this code path the timer is used as the exposure-to-infectious
            # progression timer, so it should be described as incubation-like
            # or latent/progression timing rather than a directly observed
            # symptom-onset-only incubation period.
            _inc_mu = float(self.params.get("incubation_lognorm_mu", 2.108))
            _inc_sig = float(self.params.get("incubation_lognorm_sigma", 0.254))
            _inc_clip = self.params.get("incubation_clip", (3, 17))
            ip = np.clip(np.random.lognormal(_inc_mu, _inc_sig), *_inc_clip)
            self.incubation_period[idx] = ip

            # Delay from infectious start to rash/symptom onset.
            _delay_lo, _delay_hi = self.params.get("symptom_onset_delay_range", (1, 4))
            self.symptom_onset_delay[idx] = np.random.uniform(_delay_lo, _delay_hi)

            # Duration of infectious state: Gamma(9, 2.33), clipped 14-28 days.
            _inf_shape = float(self.params.get("infectious_period_gamma_shape", 9.0))
            _inf_scale = float(self.params.get("infectious_period_gamma_scale", 2.33))
            _inf_clip = self.params.get("infectious_period_clip", (14, 28))
            inf_dur = np.clip(np.random.gamma(_inf_shape, _inf_scale), *_inf_clip)
            self.infectious_period[idx] = inf_dur

            # Day they actually become infectious.
            self.infectious_start_day[idx] = self.day + int(ip)

        # 4) record for plots
        self.layer_infection_counts[layer_name] += new_infections
        self.infections_by_day[layer_name].append(new_infections)

        return new_infections, newly_exposed

    def log_sex_party_activity(self):
        degrees = dict(self.sex_party_network.degree())
        if degrees:
            avg_encounters = sum(degrees.values()) / len(degrees)
        else:
            avg_encounters = 0
        print(f"Day {self.day}: Avg sex party encounters per participant: {avg_encounters:.2f}")

    # def _create_realistic_household_network(self):
    #
    #     """
    #     Build the household contact layer without NetworkX.
    #     Returns
    #     -------
    #     hh_src, hh_dst : int64
    #         One directed edge per household pair (the kernel treats it as undirected).
    #     hh_beta        : float32
    #         Per-edge transmission probability for the household layer.
    #     """
    #
    #     n = self.n
    #     rng = self._rng  # single Generator created in __init__
    #     assigned = np.zeros(n, dtype=bool)  # track people already placed
    #
    #     # ---------------------------------------------------------------
    #     # 1. CHILDREN (<18)  – each gets one adult "parent"
    #     # ---------------------------------------------------------------
    #     kids = np.where(self.ages < 18)[0]
    #     candidate_adults = np.where((self.ages >= 25) & (self.ages <= 60))[0]
    #
    #     rng.shuffle(candidate_adults)
    #     pair_to = candidate_adults[:len(kids)]  # one parent per child
    #
    #     src_chunks = [kids]  # child ▸ parent
    #     dst_chunks = [pair_to]
    #
    #     assigned[kids] = True
    #     assigned[pair_to] = True
    #
    #     # ---------------------------------------------------------------
    #     # 2. ELDERLY (65+)  – 50 % form elderly-only pairs
    #     # ---------------------------------------------------------------
    #     seniors = np.where(self.ages >= 65)[0]
    #     free_seniors = seniors[~assigned[seniors]]
    #
    #     rng.shuffle(free_seniors)
    #     m = (free_seniors.size // 2) * 2  # even count
    #     elder_pairs = free_seniors[:m].reshape(-1, 2)
    #
    #     if elder_pairs.size:
    #         src_chunks.append(elder_pairs[:, 0])
    #         dst_chunks.append(elder_pairs[:, 1])
    #
    #     assigned[free_seniors[:m]] = True
    #
    #     # ---------------------------------------------------------------
    #     # 3. ADULT HOUSEHOLDS (18–64)
    #     # ---------------------------------------------------------------
    #     adults_pool = np.where(~assigned & (self.ages >= 18) & (self.ages < 65))[0]
    #     rng.shuffle(adults_pool)
    #
    #     # household-size distribution: 0.3×1, 0.3×2, 0.2×3, 0.15×4, 0.05×5
    #     size_probs = np.array([.3, .3, .2, .15, .05])
    #     hh_sizes = rng.choice(np.arange(1, 6), p=size_probs, size=adults_pool.size)
    #
    #     ptr = 0 #pointer to assign adults to households
    #     for sz in hh_sizes:
    #         if ptr >= adults_pool.size:
    #             break
    #         members = adults_pool[ptr:ptr + sz]
    #         ptr += sz
    #         if members.size < 2:
    #             continue  # single-person household → no edges
    #
    #         # orientation compatibility: look up once with the 6×6 matrix
    #         mask = self.compat[self.sexcat[members][:, None],
    #         self.sexcat[members]]
    #         i, j = np.where(np.triu(mask, k=1))  # upper triangle → unique pairs
    #
    #         if i.size:
    #             src_chunks.append(members[i])
    #             dst_chunks.append(members[j])
    #
    #     # ---------------------------------------------------------------
    #     # 4. Assemble arrays
    #     # ---------------------------------------------------------------
    #     hh_src = np.concatenate(src_chunks).astype(np.int64, copy=False)
    #     hh_dst = np.concatenate(dst_chunks).astype(np.int64, copy=False)
    #     hh_beta = np.full(hh_src.size,
    #                       self.transmission_risks['household'],
    #                       dtype=np.float32)
    #
    #     return hh_src, hh_dst, hh_beta

    # def _create_school_network(self):
    #     pupils = np.where((self.ages >= 5) & (self.ages <= 18))[0]
    #     if pupils.size == 0:
    #         return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    #
    #     rng = self._rng
    #     # assign each pupil to a random class of size ≈ 25
    #     class_size = 25
    #     rng.shuffle(pupils)
    #     class_ids = np.repeat(np.arange((pupils.size + class_size - 1) // class_size),
    #                           class_size)[:pupils.size]
    #
    #     # edges: fully connect pupils within each class (*once* per pair)
    #     src_chunks, dst_chunks = [], []
    #     for cid in np.unique(class_ids):
    #         members = pupils[class_ids == cid]
    #         if members.size > 1:
    #             # unique unordered pairs
    #             i, j = np.triu_indices(members.size, k=1)
    #             src_chunks.append(members[i])
    #             dst_chunks.append(members[j])
    #
    #     school_src = np.concatenate(src_chunks).astype(np.int64)
    #     school_dst = np.concatenate(dst_chunks).astype(np.int64)
    #     school_beta = np.full(school_src.size,
    #                           self.transmission_risks["school"],
    #                           dtype=np.float32)
    #     select= rng.random(school_src.size)<0.1
    #     school_src = school_src[select];school_dst= school_dst[select];school_beta= school_beta[select]
    #     return school_src, school_dst, school_beta
    # def _create_workplace_network(self):
    #     workers = np.where((self.ages >= 22) & (self.ages <= 65))[0]
    #
    #     self._rng.shuffle(workers)
    #     # 20 % of workers draw ONE random co-worker edge
    #     try_mask = self._rng.random(workers.size) < 0.20
    #     src = workers[try_mask]
    #     dst = self._rng.choice(workers, size=src.size, replace=True)
    #     mask = src != dst
    #     src = src[mask].astype(np.int64, copy=False)
    #     dst = dst[mask].astype(np.int64, copy=False)
    #     beta = np.full(src.size,
    #                    self.transmission_risks['workplace'],
    #                    dtype=np.float32)
    #     return src, dst, beta

    # def _create_social_network(self):
    #     n = self.n
    #     src = np.arange(n, dtype=np.int64)
    #     dst = self._rng.choice(src, size=n, replace=True)
    #     mask = (src != dst) & (self._rng.random(n) < 0.20)
    #     src = src[mask]; dst = dst[mask]
    #     beta = np.full(src.size,
    #                    self.transmission_risks['social'],
    #                    dtype=np.float32)
    #     return src, dst, beta

    def _create_static_mcp_network(self, mcp_params):
        # Construction-phase RNG: independent of self._rng so static network
        # building never perturbs the runtime transmission RNG stream.
        rng = np.random.default_rng(self._seed)
        mcp_idx = self.mcp_idx  # 你 demographics 里已经算好了
        # partner counts per person
        k_part = self._draw_nb_rng(
            rng,
            mcp_idx.size,
            mean=float(mcp_params["mcp_mean"]),
            disp=float(mcp_params["mcp_disp"]),
        )
        lo, hi = mcp_params["kclip"]
        k_part = np.clip(k_part, int(lo), int(hi)).astype(np.int32, copy=False)

        src, dst, beta_uv, beta_vu = build_mcp_edges(
            mcp_idx,
            self.ages,
            self.genders,
            self.sexcat,
            self.sex_role,
            self.compat,
            k_part,
            self.transmission_risks['sexual_anal_receptive'],
            self.transmission_risks['sexual_anal_insertive'],
            self.transmission_risks['sexual_vaginal'],
            self.transmission_risks['sexual_oral'],
            self.transmission_risks['close_touch'],
            rng_seed=(int(self._seed) * 1000000) % (2**31 - 1),
        )
        return src, dst, beta_uv, beta_vu

    def _update_sexual_network(self):
        """
        Daily sexual contacts (MCP-derived).
        -----------------------------------
        Start frin tge statuc NCO edge list, then sample a fraction to be active today. Two parameters to
        tune the *number of edges

        -sexual_active_frac: faction of MCP edges active each day (default 0.10)
        -sexual_edge)mult: post-sampling tunning multiplier (default 1.0)
        Each day, randomly pick ~10% of the static MCP edges to be active.
        Returns sex_src, sex_dst, sex_puv, sex_pvu (NumPy arrays).
        """
        # total number of MCP edges
        if hasattr(self, "hh_u") and hasattr(self, "hh_v"):
            return self._update_household_network()

        # Fallback: MCP-driven daily edges (legacy)
        if not hasattr(self, "mcp_src"):
            return (np.empty(0, np.int64),
                    np.empty(0, np.int64),
                    np.empty(0, np.float32),
                    np.empty(0, np.float32))

        E = self.mcp_src.size
        if E == 0:
            return (np.empty(0, np.int64),
                    np.empty(0, np.int64),
                    np.empty(0, np.float32),
                    np.empty(0, np.float32))

        rng = self._rng
        active_frac = float(self.params.get("sexual_active_frac", 0.10))
        edge_mult = float(self.params.get("sexual_edge_mult", 1.0))

        # 1) pick a fraction of MCP edges to be active today
        idx = rng.random(self.mcp_src.size) < active_frac  # 10% of edges active today

        # slice out today's sexual contacts
        sex_src = self.mcp_src[idx]
        sex_dst = self.mcp_dst[idx]
        sex_puv = self.mcp_beta_uv[idx]
        sex_pvu = self.mcp_beta_vu[idx]

        # 2) optional additional thinning
        if edge_mult < 1.0 and sex_src.size:
            keep = rng.random(sex_src.size) < edge_mult
            sex_src = sex_src[keep]
            sex_dst = sex_dst[keep]
            sex_puv = sex_puv[keep]
            sex_pvu = sex_pvu[keep]

        return sex_src, sex_dst, sex_puv, sex_pvu

    # def _update_oneoff_layer(self):
    #     rng = self._rng
    #
    #     # 1. random subset of MCP people (1%) active today
    #     active = np.where(self.mcp_flags & (rng.random(self.n) < 0.01))[0]

    def _update_household_close_layer(self):
        """Daily non-sexual close-contact within stable partner households.

        This is meant to capture household/intimate non-sexual contact (e.g., cohabitation, caregiving).
        It uses the pre-built stable partner pairs (self.hh_u, self.hh_v) and activates each pair
        each day with probability ``p_household_close_per_day`` (default 1.0).

        Returns
        -------
        src, dst : int64
        p_uv, p_vu : float32  (both directions, same baseline)
        """
        if (not hasattr(self, "hh_u")) or self.hh_u.size == 0:
            return (np.empty(0, np.int64), np.empty(0, np.int64),
                    np.empty(0, np.float32), np.empty(0, np.float32))

        p = float(self.params.get("p_household_close_per_day", 1.0))
        if p <= 0.0:
            return (np.empty(0, np.int64), np.empty(0, np.int64),
                    np.empty(0, np.float32), np.empty(0, np.float32))

        rng = self._rng
        if p < 1.0:
            active = rng.random(self.hh_u.size) < p
            src = self.hh_u[active].astype(np.int64, copy=False)
            dst = self.hh_v[active].astype(np.int64, copy=False)
        else:
            src = self.hh_u.astype(np.int64, copy=False)
            dst = self.hh_v.astype(np.int64, copy=False)

        base = float(self.transmission_risks.get("close_touch", 0.0))
        mult = float(self.params.get("household_close_risk_mult", 1.0))
        beta = base * mult
        p_uv = np.full(src.size, beta, dtype=np.float32)
        p_vu = p_uv.copy()
        return src, dst, p_uv, p_vu

    def _update_social_layer(self):
        """Simplified nonsexual/social close-contact layer.

        This layer is a compact bridge from contagious people into the adult
        general population. It is not intended to explicitly simulate every
        workplace, bar, shelter, restaurant, festival, or casual household
        encounter. Instead, it captures the effective contribution of routine
        close contact to late spillover outside the MSM core.

        Simplified v1 design
        --------------------
        1) Sources are people who are infectious OR symptomatic today.
        2) Edges are generated directly from those sources, avoiding construction
           of a huge social network that would later be filtered.
        3) The number of nonsexual contacts is anchored to CDC data:
              2.4 identified nonsexual contacts per case.
           This is spread across an effective contagious window D.
        4) social_contact_scale multiplies contact opportunities only.
           social_p_eff remains externally anchored and should not be tuned
           together with social_contact_scale.
        5) social_same_sex_assort controls sex-assortative social mixing.
           Because most contagious sources are male in the outbreak, same-sex
           assortativity permits heterosexual-male spillover without a direct
           hard-coded male-target boost.
        6) The former "community_bridge_scale" late-stage social ramp was
           REMOVED (no literature basis; a low-interpretability LA-specific
           fitting knob). The social layer now scales contacts uniformly over
           time via social_contact_scale only.
        """
        rng = self._rng

        def _empty():
            return (np.empty(0, np.int64), np.empty(0, np.int64),
                    np.empty(0, np.float32), np.empty(0, np.float32))

        # 1) Contagious adult sources: infectious OR symptomatic.
        contagious = (self.infectious | self.symptomatic)
        src_pool = np.where(contagious & (self.ages >= 18))[0].astype(np.int64, copy=False)
        if src_pool.size == 0:
            return _empty()

        max_sources = int(self.params.get("social_max_sources_per_day", 100_000))
        if src_pool.size > max_sources:
            src_pool = rng.choice(src_pool, size=max_sources, replace=False).astype(np.int64, copy=False)

        # 2) Cached adult target pools.
        if getattr(self, "_social_v1_target_pools_ready", False) is False:
            adults = (self.ages >= 18)
            male = (self.genders == 0)
            female = (self.genders == 1)

            self._social_v1_male_pool = np.where(adults & male)[0].astype(np.int64, copy=False)
            self._social_v1_female_pool = np.where(adults & female)[0].astype(np.int64, copy=False)
            self._social_v1_all_pool = np.where(adults)[0].astype(np.int64, copy=False)

            pool = self._social_v1_all_pool
            self._social_v1_mean_r_sus = float(self.r_sus[pool].mean()) if pool.size else 1.0
            self._social_v1_target_pools_ready = True

        male_pool = self._social_v1_male_pool
        female_pool = self._social_v1_female_pool
        all_pool = self._social_v1_all_pool
        if all_pool.size == 0:
            return _empty()

        # 3) Contacts per contagious source per day.
        contacts_per_case = float(self.params.get("social_contacts_per_case", 2.4))
        D = max(float(self.params.get("social_contact_window_days", 21.0)), 1.0)

        contact_scale = float(self.params.get(
            "social_contact_scale",
            self.params.get("social_contact_underascertain_mult", 4.20)
        ))

        lam = (contacts_per_case / D) * contact_scale
        lam = float(np.clip(lam, 0.0, 20.0))

        n_contacts_each = rng.poisson(lam=lam, size=src_pool.size).astype(np.int32, copy=False)
        max_per_source = int(self.params.get("social_max_contacts_per_source_per_day", 30))
        if max_per_source > 0:
            n_contacts_each = np.minimum(n_contacts_each, max_per_source)

        keep_src = n_contacts_each > 0
        if not np.any(keep_src):
            return _empty()

        src = np.repeat(src_pool[keep_src], n_contacts_each[keep_src]).astype(np.int64, copy=False)
        if src.size == 0:
            return _empty()

        # 4) Target selection with sex-assortative social mixing.
        # This replaces the older direct male-targeting parameter.  For each
        # social/nonsexual contact generated by a contagious source, the target
        # is sampled from the same biological sex with probability
        # social_same_sex_assort; otherwise, from the opposite sex. Because the
        # outbreak source pool is strongly male-predominant, this still creates
        # male-to-male social spillover opportunities without explicitly forcing
        # 92% of targets to be male.
        same_sex_assort = float(self.params.get(
            "social_same_sex_assort",
            self.params.get("social_male_bias", self.params.get("social_target_male_share", 0.75))
        ))
        same_sex_assort = float(np.clip(same_sex_assort, 0.0, 1.0))

        src_is_male = (self.genders[src] == 0)
        choose_same_sex = rng.random(src.size) < same_sex_assort
        draw_male = (choose_same_sex & src_is_male) | ((~choose_same_sex) & (~src_is_male))
        dst = np.empty(src.size, dtype=np.int64)

        n_m = int(draw_male.sum())
        n_f = int(src.size - n_m)

        if n_m > 0:
            if male_pool.size > 0:
                dst[draw_male] = rng.choice(male_pool, size=n_m, replace=True)
            else:
                dst[draw_male] = rng.choice(all_pool, size=n_m, replace=True)

        if n_f > 0:
            if female_pool.size > 0:
                dst[~draw_male] = rng.choice(female_pool, size=n_f, replace=True)
            else:
                dst[~draw_male] = rng.choice(all_pool, size=n_f, replace=True)

        # Resample self-loops a few times, preserving the same-sex/opposite-sex
        # target choice where possible, then drop any remaining loops.
        same = (src == dst)
        for _ in range(3):
            if not np.any(same):
                break
            idx = np.where(same)[0]
            dm = draw_male[idx]
            nm = int(dm.sum())
            nf = int(idx.size - nm)
            if nm > 0:
                idx_m = idx[dm]
                dst[idx_m] = rng.choice(male_pool if male_pool.size else all_pool, size=nm, replace=True)
            if nf > 0:
                idx_f = idx[~dm]
                dst[idx_f] = rng.choice(female_pool if female_pool.size else all_pool, size=nf, replace=True)
            same = (src == dst)

        if np.any(same):
            keep = ~same
            src = src[keep]
            dst = dst[keep]

        if src.size == 0:
            return _empty()

        # 5) Per-contact probability.
        # fast_tx_flags multiplies by r_sus[target]; therefore convert the
        # externally anchored effective p into p0 using target-pool mean r_sus.
        p_eff = float(self.params.get("social_p_eff", self.params.get("social_p", 0.0063)))
        if bool(self.params.get("social_adjust_for_r_sus", True)):
            mean_r = float(getattr(self, "_social_v1_mean_r_sus", 1.0))
            p0 = p_eff / max(mean_r, 1e-6)
            p_cap = float(self.params.get("social_p0_cap", 0.03))
            p0 = min(p0, p_cap)
        else:
            mean_r = float(self.r_sus[dst].mean()) if dst.size else 1.0
            p0 = p_eff

        p_uv = np.full(src.size, p0, dtype=np.float32)
        p_vu = p_uv.copy()

        if self.day % 14 == 0:
            actual_mean_r = float(self.r_sus[dst].mean()) if dst.size else 1.0
            male_target_frac = float(np.mean(self.genders[dst] == 0)) if dst.size else 0.0
            hetero_target_frac = float(np.mean(self.orientations[dst] == "hetero")) if dst.size else 0.0
            same_sex_target_frac = float(np.mean(self.genders[src] == self.genders[dst])) if dst.size else 0.0
            print(
                f"[day {self.day}] social(v1): sources={src_pool.size}, edges={src.size}, "
                f"D={D:.1f}, scale={contact_scale:.2f}, lam={lam:.4f}, "
                f"p_eff={p_eff:.4g}, p0={p0:.4g}, "
                f"mean_r_pool={mean_r:.3f}, mean_r_dst={actual_mean_r:.3f}, "
                f"same_sex_assort={same_sex_assort:.2f}, same_sex_targets={same_sex_target_frac:.1%}, "
                f"male_targets={male_target_frac:.1%}, hetero_targets={hetero_target_frac:.1%}"
            )

        return src.astype(np.int64, copy=False), dst.astype(np.int64, copy=False), p_uv, p_vu

    def _update_oneoff_layer(self):
        rng = self._rng
        mcp = self.mcp_idx

        # 1) MCP 子集里，今日活跃。默认均匀活跃度；可选按 6 层 activity
        #    level 分层活跃（heavy-tail，对齐 CDC MMWR 的 riskg 设计）。
        oneoff_frac = float(self.params.get('oneoff_active_frac', 0.03))
        stratified = bool(self.params.get('oneoff_activity_stratified', False))

        # Orientation-stratified one-off activity. CDC's one-time network is
        # effectively MSM-only (the reference model has no heterosexual nodes);
        # in the 2022 clade IIb outbreak, women were 2.7% of cases and
        # heterosexual-only contacts 9% of NYC cases. Running one-off activity
        # uniformly across the whole MCP pool therefore over-amplifies MF
        # spillover once one-off activity rises. We bias one-off participation
        # toward MSM / male-bisexual strata and damp hetero strata.
        ori_bias = self.params.get('oneoff_orientation_bias', None)

        if ori_bias is not None and mcp.size:
            # ori_bias: dict-like mapping sexcat -> relative participation weight.
            # sexcat: 0=FH, 1=MH, 2=MSM, 3=WSW, 4=MB, 5=FB.
            w = np.ones(6, dtype=np.float64)
            for k, v in ori_bias.items():
                w[int(k)] = float(v)
            sc = self.sexcat[mcp]
            ori_w = w[sc]
            # normalise so the *weighted mean* participation equals oneoff_frac
            # (keeps total one-off contact volume anchored, only re-weights
            #  WHO participates toward MSM).
            mean_w = float(np.mean(ori_w))
            if mean_w > 0:
                ori_w = ori_w * (oneoff_frac / mean_w)
            ori_p = np.clip(ori_w, 0.0, 1.0)

        if stratified and hasattr(self, "activity_level") and mcp.size:
            # Per-level daily active fraction, aligned with CDC qnts relative
            # one-off partner-formation rates (CDC_RISKG_ONETIME_7D):
            #   level 0..5 -> 0.000, 0.007, 0.038, 0.071, 0.221, 2.000 (per 7 days)
            # Normalised so that the *population-weighted* mean active fraction
            # equals oneoff_frac (keeps total one-off contact volume anchored).
            qnts = CDC_RISKG_ONETIME_7D
            # convert rate-per-7-days to a daily active probability via
            #   p_active_day = 1 - exp(-rate/7)   (poisson arrival of partners)
            level_rate = qnts / 7.0
            level_p_active = 1.0 - np.exp(-level_rate)
            level_p_active[0] = 0.0  # stratum 0 never active in one-off layer

            # population weights (CDC riskg: 5x19% + 1x5%)
            w = CDC_RISKG_POP_WEIGHTS / CDC_RISKG_POP_WEIGHTS.sum()
            mean_active = float(np.sum(w * level_p_active))

            # anchor: rescale all non-zero strata so weighted mean == oneoff_frac
            if mean_active > 0:
                scale = oneoff_frac / mean_active
            else:
                scale = 0.0
            p_active_by_level = level_p_active * scale
            p_active_by_level = np.clip(p_active_by_level, 0.0, 1.0)

            lvl = self.activity_level[mcp]  # int8, 0..5 (all MCP are >=0)
            p_draw = p_active_by_level[lvl]

            # Combine with orientation bias: p_draw * ori_w (already normalised
            # so its mean == oneoff_frac). To keep the *activity-level* heavy-tail
            # shape intact while re-weighting toward MSM, multiply level-active
            # probability by the orientation weight and re-anchor the mean.
            if ori_bias is not None:
                p_draw = p_draw * ori_p
                # re-anchor mean to oneoff_frac
                if p_draw.size and float(np.mean(p_draw)) > 0:
                    p_draw = p_draw * (oneoff_frac / float(np.mean(p_draw)))
                p_draw = np.clip(p_draw, 0.0, 1.0)

            active = mcp[rng.random(mcp.size) < p_draw]
        else:
            if ori_bias is not None:
                active = mcp[rng.random(mcp.size) < ori_p]
            else:
                active = mcp[rng.random(mcp.size) < oneoff_frac]

        if active.size == 0:
            return (np.empty(0, np.int64),
                    np.empty(0, np.int64),
                    np.empty(0, np.float32),
                    np.empty(0, np.float32))

        # 2) 行为缩放（与你 sexual 层一致）
        scale = self._sexual_scale_today()

        age_gap = int(self.params.get("oneoff_age_gap", 20))
        oneoff_p = float(self.params.get("oneoff_p", 0.15))
        mf_anal_frac = float(self.params.get("oneoff_mf_anal_frac", 0.10))

        # 3) 调用稳定核
        one_src, one_dst, p_uv, p_vu = build_oneoff_edges_fast(
            active,
            self.ages, self.genders, self.sexcat, self.sex_role, self.compat,
            self.transmission_risks['sexual_anal_receptive'],
            self.transmission_risks['sexual_anal_insertive'],
            self.transmission_risks['sexual_vaginal'],
            self.transmission_risks['sexual_oral'],
            self.transmission_risks['close_touch'],
            mf_anal_frac=mf_anal_frac,
            age_gap=age_gap,
            oneoff_p=oneoff_p,
            keep_scale=scale,
            rng_seed=(int(self._seed) * 100000 + int(self.day)) % (2**31 - 1),
        )

        return one_src, one_dst, p_uv, p_vu

        # # desired partner count per active person   NegBin(1,0.20)
        # k_desired = np.random.negative_binomial(1, 0.20, size=active.size).astype(np.int32)
        # shuffled = rng.permutation(active)
        #
        # srcL, dstL = [], []
        # for uid, kk in zip(active, k_desired):
        #     kk = int(kk)
        #     if kk == 0:
        #         continue
        #
        #     cand = shuffled[shuffled != uid]
        #
        #     # orientation compatibility + age ≤10
        #     ok = (np.abs(self.ages[cand] - self.ages[uid]) <= 10) & \
        #          self.compat[self.sexcat[uid], self.sexcat[cand]]
        #     cand = cand[ok]
        #     if cand.size == 0:
        #         continue
        #
        #     partners = rng.choice(cand, size=min(kk, cand.size), replace=False)
        #     srcL.append(np.full(partners.size, uid, dtype=np.int64))
        #     dstL.append(partners.astype(np.int64))
        #
        # if not srcL: #no one-off contacts today
        #     return (np.empty(0, np.int64),
        #             np.empty(0, np.int64),
        #             np.empty(0, np.float32),
        #             np.empty(0, np.float32))
        # # concatenate all one-off contacts
        # one_src = np.concatenate(srcL)
        # one_dst = np.concatenate(dstL)
        #
        # # 2. edge-specific β  (vectorised by biological sex)
        # g_src = self.genders[one_src]
        # g_dst = self.genders[one_dst]
        #
        # p_uv = np.empty(one_src.size, dtype=np.float32)
        # p_vu = np.empty(one_src.size, dtype=np.float32)
        #
        # mm = (g_src == 0) & (g_dst == 0)  # MSM
        # ff = (g_src == 1) & (g_dst == 1)  # WSW
        # mf = ~(mm | ff)  # mixed  ➜ vaginal
        # p_anal_receptive, p_anal_insertive, p_vag, p_oral, p_skin = (self.transmission_risks['sexual_anal_receptive'],
        #                                                              self.transmission_risks['sexual_anal_insertive'],
        #                                                              self.transmission_risks['sexual_vaginal'],
        #                                                              self.transmission_risks['sexual_oral'],
        #                                                              self.transmission_risks['close_touch'])
        # # ---MSM: use insertive/receptive roles to set p_uv, p_vu ---
        # for i in np.where(mm)[0]:
        #     u,v= one_src[i], one_dst[i]
        #     r1, r2= self.sex_role[u], self.sex_role[v]
        #     if r1 == 1 and r2 == 0:  # uid insertive, v
        #         p_uv[i] = p_anal_receptive
        #         p_vu[i] = p_anal_insertive
        #     elif r1 == 0 and r2 == 1:  # uid receptive, v insertive
        #         p_uv[i] = p_anal_insertive
        #         p_vu[i] = p_anal_receptive
        #     elif r1 == 2 and r2 == 0:  # uid versatile, v receptive
        #         p_uv[i] = p_anal_receptive
        #         p_vu[i] = p_anal_insertive
        #     elif r1 == 0 and r2 == 2:  # uid receptive, v versatile
        #         p_uv[i] = p_anal_insertive
        #         p_vu[i] = p_anal_receptive
        #     elif r1 == 1 and r2 == 2:  # uid insertive, v versatile
        #         p_uv[i] = p_anal_receptive
        #         p_vu[i] = p_anal_insertive
        #     elif r1 == 2 and r2 == 1:  # uid versatile, v insertive
        #         p_uv[i] = p_anal_insertive
        #         p_vu[i] = p_anal_receptive
        #     elif r1 == 2 or r2 == 2:  # either versatile
        #          avg = 0.5 * (p_anal_receptive + p_anal_insertive)
        #          p_uv[i], p_vu[i]= avg, avg
        #     else:  # both exclusive same role
        #         p_uv[i], p_vu[i] = p_skin, p_skin
        # # --- WSW: oral risk both directions ---
        # p_uv[ff] = p_oral
        # p_vu[ff] = p_oral
        # # --- Mixed: vaginal risk both directions ---
        # p_uv[mf] = p_vag
        # p_vu[mf] = p_vag
        #
        #
        #
        # # 3. schedule down-scaling (e.g. behaviour change after day 90)
        # scale = self._sexual_scale_today()
        # if scale < 1.0:
        #     keep = rng.random(one_src.size) < scale
        #     one_src, one_dst = one_src[keep], one_dst[keep]
        #     p_uv, p_vu = p_uv[keep], p_vu[keep]
        #
        # return one_src, one_dst, p_uv, p_vu

    def _assign_bi_party(self, uid):
        """Return 'msm', 'wsw' or 'general' for a bi‑sexual person uid."""
        g = self.genders[uid]
        if g == 0:
            return 'msm'  # default: male‑bi joins MSM night
        elif g == 1:
            return 'wsw'  # default: female‑bi joins WSW night
        return 'general'

    def _draw_partners(self, n_draws=1):
        """
        Negative‑binomial r=1, p=0.2  → mean≈4, var≈20 (heavy right tail)
        Returns an ndarray of length *n_draws*.
        """
        return np.random.negative_binomial(1, 0.2, size=n_draws)

        # ==========================================================================
        #  SMALL & LARGE *non‑sexual* MCP EVENTS
        # ==========================================================================

    # def _update_small_large_event_networks(self):
    #     rng = self._rng
    #     mcp = np.where(self.mcp_flags)[0]
    #     if mcp.size == 0:
    #         empty = np.empty(0, np.int64)
    #         return empty, empty, empty, empty, empty, empty
    #
    #     small_att = mcp[rng.random(mcp.size) < 0.20]
    #     large_att = mcp[rng.random(mcp.size) < 0.10]
    #
    #     def _make_layer(att, sz_min, sz_max, base_p):
    #         if att.size < 2:
    #             return np.empty(0, np.int64), np.empty(0, np.int64)
    #         # partition once
    #         off = _cum_offsets(self._rng.integers(sz_min, sz_max + 1,
    #                                               size=(att.size // sz_min) + 1))
    #         off[-1] = att.size  # ensure exact length
    #         return _party_edges(att.astype(np.int64), off,
    #                             base_p, self.same_bias_small_large,
    #                             self._gender_uint8, self.sexcat,
    #                             self._compat_flat)
    #
    #     sm_src, sm_dst = _make_layer(small_att, 10, 30, 0.35)
    #     lg_src, lg_dst = _make_layer(large_att, 100, 500, 0.20)
    #
    #     sm_beta = np.full(sm_src.size, self.transmission_risks['small_event'], dtype=np.float32)
    #     lg_beta = np.full(lg_src.size, self.transmission_risks['large_event'], dtype=np.float32)
    #
    #     return sm_src, sm_dst, sm_beta, lg_src, lg_dst, lg_beta
    # ───────────────── SEX‑PARTY NETWORK (three sub‑pools) ──────────────
    def _update_sex_party_network(self):
        """
        Sex-party network (three pools) with:
          - MSM / WSW / MIX pools
          - bisexual has equal chance to be assigned to themed vs MIX (bi_m -> MSM or MIX; bi_f -> WSW or MIX)
          - per-day attendance mutually exclusive (one pool max per person)
          - partner generation: ~p_sex have sex; mean partners ~2 overall (two-stage: had_sex then NB(r=1,p))
          - generate FULL party sex edges (for stats), but do transmission ONLY on edges where at least one endpoint
            is tx_source (infectious|symptomatic) (Covasim-style filter you used)
          - MIX edges are male-female only; transmission baseline vaginal; with p_anal fraction upgraded to anal
          - debug every 14 days: attendance + MIX composition + infected fractions + MIX MF/FM edges + degree + bridge count
        Returns:
          (sex_src_tx, sex_dst_tx, sp_puv, sp_pvu,
           tp_src_tx,  tp_dst_tx,  tp_puv, tp_pvu,
           n_msm_att, n_wsw_att, n_mix_att)
        """
        rng = self._rng

        # ---- Day-of-week modulation ----
        wd = int(self.day % 7)  # 0..6
        is_weekend_night = (wd in (4, 5))  # Fri, Sat
        weekend_only = bool(self.params.get("party_weekend_only", False))
        if weekend_only and (not is_weekend_night):
            empty = (np.empty(0, np.int64), np.empty(0, np.int64),
                     np.empty(0, np.float32), np.empty(0, np.float32),
                     np.empty(0, np.int64), np.empty(0, np.int64),
                     np.empty(0, np.float32), np.empty(0, np.float32),
                     0, 0, 0)
            return empty

        if weekend_only:
            weekday_mult = 0.0
            weekend_mult = float(self.params.get("party_weekend_mult", 3.5))
        else:
            weekday_mult = float(self.params.get("party_weekday_mult", 0.7))
            weekend_mult = float(self.params.get("party_weekend_mult", 1.3))
        day_mult = weekend_mult if is_weekend_night else weekday_mult

        # ---- Eligible attendees ----
        eligible = self.party_idx
        if eligible.size == 0:
            empty = (np.empty(0, np.int64), np.empty(0, np.int64),
                     np.empty(0, np.float32), np.empty(0, np.float32),
                     np.empty(0, np.int64), np.empty(0, np.int64),
                     np.empty(0, np.float32), np.empty(0, np.float32),
                     0, 0, 0)
            return empty

        # ---- Attendance probabilities (per day) ----
        base_att = float(self.params.get("sex_party_attend_prob", 3.0 / 7.0))
        p_msm = float(self.params.get("party_msm_attend_prob", base_att))
        p_wsw = float(self.params.get("party_wsw_attend_prob", base_att))
        p_mix = float(self.params.get("party_mix_attend_prob", base_att))
        # No sex-specific MIX attendance multiplier is used in the final simplified baseline.
        # Male and female MIX participants use the same attendance probability once eligible.
        # This avoids using a hidden female-ratio tuning knob and makes the pathway clearer:
        # heterosexual/bisexual agents can enter MIX events directly, stable partners can
        # bridge infection, and event close-touch can add nonsexual spillover.
        p_mix_m = p_mix
        p_mix_f = p_mix

        if day_mult != 1.0:
            p_msm = min(1.0, max(0.0, p_msm * day_mult))
            p_wsw = min(1.0, max(0.0, p_wsw * day_mult))
            p_mix = min(1.0, max(0.0, p_mix * day_mult))
            p_mix_m = min(1.0, max(0.0, p_mix_m * day_mult))
            p_mix_f = min(1.0, max(0.0, p_mix_f * day_mult))
        # ---- Close-touch expansion fraction ----
        # Final simplified baseline uses one shared event-layer close-touch
        # expansion parameter across MSM, WSW, and MIX high-contact event pools.
        # This avoids unsupported pool-specific tuning.
        #
        # Preferred parameter:
        #   party_touch_frac
        # Legacy fallback:
        #   sex_party_touch_frac
        party_touch_frac = float(self.params.get(
            "party_touch_frac",
            self.params.get("sex_party_touch_frac", 0.20)
        ))
        party_touch_frac = float(np.clip(party_touch_frac, 0.0, 1.0))
        touch_msm = party_touch_frac
        touch_wsw = party_touch_frac
        touch_mix = party_touch_frac

        # ---- Routing rules ----
        homo_spill = float(self.params.get("party_mix_homo_spill_frac", 0.00))  # share of homo moved into MIX

        # ---- Partner distribution controls (Table2/3-aligned) ----
        p_sex = float(self.params.get("party_sex_frac", 0.73))
        # mean partners overall ~2  => E[k|sex]=2/p_sex ; NB(r=1,p): mean=(1-p)/p => p=1/(1+mean)
        default_nb_p = 1.0 / (1.0 + (2.0 / max(p_sex, 1e-6)))
        nb_p = float(self.params.get("party_nb_p", default_nb_p))
        k_clip_party = int(self.params.get("party_k_clip", 15))

        # ---- Heavy-tail activity stratification for sex_party partner degree ----
        # Aligned with CDC MMWR 6-level `riskg` (Weiss et al. 2020) and the Milan
        # case series (23% of cases report >50 partners / 3 months). A small
        # fraction of the MSM core pool is flagged as "high-activity" and draws a
        # much heavier partner tail than the bulk, driving intra-party amplification
        # WITHOUT widening hetero spill — the high-activity stratum is restricted to
        # the same-sex male core, so its extra edges stay MM (this layer's MM edges
        # already dominate). See reference/sex_party_参数文献依据.md.
        party_ht_enabled = bool(self.params.get("party_heavy_tail", True))
        # Fraction of the MSM core pool flagged high-activity (heavy tail).
        # Kept for backward-compat documentation; with the full 6-level riskg
        # gradient the top stratum share is fixed by CDC_RISKG_POP_WEIGHTS (5%).
        party_ht_frac = float(self.params.get("party_ht_frac", 0.05))
        # E[k | sexually active] for the TOP (stratum 5) activity stratum.
        # Anchored to CDC top-5% (~100 partners/year) and Milan 23%>50/3mo;
        # calibrated to 20 in the 10M multi-seed sweep. Lower strata scale by
        # the CDC past-3-week partner-count ratio (CDC_RISKG_PARTNERS_3W).
        party_ht_top_mean = float(self.params.get("party_ht_top_mean", 20.0))
        # Clip for the top activity stratum (bulk keeps `party_k_clip`).
        party_ht_top_clip = int(self.params.get("party_ht_top_clip", 30))
        # Per-stratum E[k] (0..5) for the sex_party partner degree, scaled from
        # the CDC past-3-week partner ratios and anchored to party_ht_top_mean.
        # With top=20 -> [2.42, 2.42, 2.73, 3.03, 4.55, 20.0].
        sexparty_ek_by_level = _cpc_sexparty_ek(party_ht_top_mean)

        # ---- MIX anal fraction (upgrade vaginal->anal on a subset of MF edges) ----
        # Directional split (2026-11-04): female-as-receptive (male->female)
        # anal is independently controllable from female-as-insertive
        # (female->male). Biologically, heterosexual transmission is
        # predominantly vaginal (Lancet 2022 Thornhill: cis-women show
        # vulvovaginal lesions, not anorectal), so the MF receptive-anal
        # fraction should be well below the MSM anal fraction.
        mix_mf_anal_frac = float(self.params.get("mix_mf_anal_frac", 0.10))  # male->female (female receptive)
        mix_fm_anal_frac = float(self.params.get("mix_fm_anal_frac", 0.10))  # female->male (female insertive)

        # ---- Candidate pools by demographics/orientation ----
        g = self.genders[eligible]
        ori = self.orientations[eligible]

        msm_core = eligible[(g == 0) & (ori == 'homo')]
        wsw_core = eligible[(g == 1) & (ori == 'homo')]

        bi_m = eligible[(g == 0) & (ori == 'bi')]
        bi_f = eligible[(g == 1) & (ori == 'bi')]

        het_m = eligible[(g == 0) & (ori == 'hetero')]
        het_f = eligible[(g == 1) & (ori == 'hetero')]

        # ---- Spillover of homo into MIX (remove from core so mutually exclusive routing is real) ----
        msm_spill = np.empty(0, np.int64)
        wsw_spill = np.empty(0, np.int64)
        if msm_core.size and homo_spill > 0:
            take = int(round(homo_spill * msm_core.size))
            if take > 0:
                msm_spill = rng.choice(msm_core, size=min(take, msm_core.size), replace=False)
                msm_core = msm_core[~np.isin(msm_core, msm_spill)]
        if wsw_core.size and homo_spill > 0:
            take = int(round(homo_spill * wsw_core.size))
            if take > 0:
                wsw_spill = rng.choice(wsw_core, size=min(take, wsw_core.size), replace=False)
                wsw_core = wsw_core[~np.isin(wsw_core, wsw_spill)]

        # ---- Attendance assignment (mutually exclusive) ----
        taken = np.empty(0, np.int64)

        def _take(x: np.ndarray):
            nonlocal taken
            if x.size:
                taken = np.concatenate([taken, x]) if taken.size else x

        def _remaining(cand: np.ndarray) -> np.ndarray:
            if cand.size == 0:
                return cand
            if taken.size == 0:
                return cand
            m = ~np.isin(cand, taken)
            return cand[m]

        msm_att = np.empty(0, np.int64)
        wsw_att = np.empty(0, np.int64)
        mix_att = np.empty(0, np.int64)

        # 1) bi attendance: attend with p_bi ≈ average, then split 50/50 to themed vs MIX
        p_bi_m = 0.5 * (p_msm + p_mix)
        if bi_m.size and p_bi_m > 0:
            bi_m_today = bi_m[rng.random(bi_m.size) < p_bi_m]
            if bi_m_today.size:
                p_bi_m_to_msm = float(self.params.get('p_bi_m_to_msm', 0.5))
                go_msm = rng.random(bi_m_today.size) < p_bi_m_to_msm
                bi_m_to_msm = bi_m_today[go_msm]
                bi_m_to_mix = bi_m_today[~go_msm]
                if bi_m_to_msm.size:
                    msm_att = np.concatenate([msm_att, bi_m_to_msm]) if msm_att.size else bi_m_to_msm
                    _take(bi_m_to_msm)
                if bi_m_to_mix.size:
                    mix_att = np.concatenate([mix_att, bi_m_to_mix]) if mix_att.size else bi_m_to_mix
                    _take(bi_m_to_mix)

        p_bi_f = 0.5 * (p_wsw + p_mix)
        if bi_f.size and p_bi_f > 0:
            bi_f_today = bi_f[rng.random(bi_f.size) < p_bi_f]
            if bi_f_today.size:
                p_bi_f_to_wsw = float(self.params.get('p_bi_f_to_wsw', 0.5))
                go_wsw = rng.random(bi_f_today.size) < p_bi_f_to_wsw
                bi_f_to_wsw = bi_f_today[go_wsw]
                bi_f_to_mix = bi_f_today[~go_wsw]
                if bi_f_to_wsw.size:
                    wsw_att = np.concatenate([wsw_att, bi_f_to_wsw]) if wsw_att.size else bi_f_to_wsw
                    _take(bi_f_to_wsw)
                if bi_f_to_mix.size:
                    mix_att = np.concatenate([mix_att, bi_f_to_mix]) if mix_att.size else bi_f_to_mix
                    _take(bi_f_to_mix)

        # 2) remaining core/hetero/spill
        msm_cand = _remaining(msm_core)
        if msm_cand.size and p_msm > 0:
            draw = msm_cand[rng.random(msm_cand.size) < p_msm]
            if draw.size:
                msm_att = np.concatenate([msm_att, draw]) if msm_att.size else draw
                _take(draw)

        wsw_cand = _remaining(wsw_core)
        if wsw_cand.size and p_wsw > 0:
            draw = wsw_cand[rng.random(wsw_cand.size) < p_wsw]
            if draw.size:
                wsw_att = np.concatenate([wsw_att, draw]) if wsw_att.size else draw
                _take(draw)

        mix_cand = np.concatenate([het_m, het_f, msm_spill, wsw_spill]) \
            if (het_m.size + het_f.size + msm_spill.size + wsw_spill.size) else np.empty(0, np.int64)
        mix_cand = _remaining(mix_cand)
        if mix_cand.size and (p_mix_m > 0 or p_mix_f > 0):
            mix_m = mix_cand[self.genders[mix_cand] == 0]
            mix_f = mix_cand[self.genders[mix_cand] == 1]
            draw_m = mix_m[rng.random(mix_m.size) < p_mix_m] if (mix_m.size and p_mix_m > 0) else np.empty(0, np.int64)
            draw_f = mix_f[rng.random(mix_f.size) < p_mix_f] if (mix_f.size and p_mix_f > 0) else np.empty(0, np.int64)
            draw = draw_m if draw_f.size == 0 else (draw_f if draw_m.size == 0 else np.concatenate([draw_m, draw_f]))
            if draw.size:
                mix_att = np.concatenate([mix_att, draw]) if mix_att.size else draw
                _take(draw)

        if (msm_att.size + wsw_att.size + mix_att.size) < 3:
            empty = (np.empty(0, np.int64), np.empty(0, np.int64),
                     np.empty(0, np.float32), np.empty(0, np.float32),
                     np.empty(0, np.int64), np.empty(0, np.int64),
                     np.empty(0, np.float32), np.empty(0, np.float32),
                     int(msm_att.size), int(wsw_att.size), int(mix_att.size))
            return empty

        # ---- Edge generation helpers ----
        N = int(self.n)

        def _undirected_unique(u: np.ndarray, v: np.ndarray):
            if u.size == 0:
                return np.empty(0, np.int64), np.empty(0, np.int64)
            a = np.minimum(u, v)
            b = np.maximum(u, v)
            m = (a != b)
            if not np.any(m):
                return np.empty(0, np.int64), np.empty(0, np.int64)
            a = a[m].astype(np.int64, copy=False)
            b = b[m].astype(np.int64, copy=False)
            key = a * N + b
            _, idx = np.unique(key, return_index=True)
            return a[idx], b[idx]

        def _bi(u: np.ndarray, v: np.ndarray):
            if u.size == 0:
                return np.empty(0, np.int64), np.empty(0, np.int64)
            src = np.concatenate([u, v]).astype(np.int64, copy=False)
            dst = np.concatenate([v, u]).astype(np.int64, copy=False)
            return src, dst

        def _make_offsets(n: int, min_sz: int = 3, max_sz: int = 60) -> np.ndarray:
            if n <= 0:
                return np.array([0], dtype=np.int32)
            sizes = rng.integers(min_sz, max_sz + 1, size=max(1, n // min_sz) + 8)
            cs = np.cumsum(sizes)
            cs = cs[cs < n]
            if cs.size == 0:
                return np.array([0, n], dtype=np.int32)
            off = np.empty(cs.size + 2, dtype=np.int32)
            off[0] = 0
            off[1:-1] = cs.astype(np.int32)
            off[-1] = n
            return off

        def _gen_edges_same_pool_grouped(pool: np.ndarray):
            if pool.size < 3:
                return np.empty(0, np.int64), np.empty(0, np.int64)

            pool = pool.copy()
            rng.shuffle(pool)
            offsets = _make_offsets(pool.size, 3, 60)

            src_all, dst_all = [], []
            for i in range(offsets.size - 1):
                grp = pool[offsets[i]:offsets[i + 1]]
                n = grp.size
                if n < 3:
                    continue

                active = rng.random(n) < p_sex
                idx = np.where(active)[0]
                if idx.size == 0:
                    continue

                # Partner degree draws, stratified by the 6-level CDC riskg
                # activity stratum (restricted to the MSM core pool). Each
                # stratum draws NB(r=1) with its own E[k] from
                # sexparty_ek_by_level, giving a smooth heavy-tail gradient
                # rather than a two-tier split. The top stratum uses
                # party_ht_top_clip; lower strata use party_k_clip.
                if party_ht_enabled and hasattr(self, "sex_party_activity_level"):
                    grp_uid = grp[idx]  # global uid of active attendees
                    lvl = self.sex_party_activity_level[grp_uid]  # -1..5
                    # attendee level (guard -1 -> treat as bulk mean)
                    lvl_safe = np.where(lvl < 0, 0, lvl).astype(np.int32)
                    # per-attendee E[k] from the 6-level gradient
                    ek = sexparty_ek_by_level[lvl_safe]
                    # NB(r=1, p) with mean = (1-p)/p  =>  p = 1/(1+mean), per attendee
                    p_vec = 1.0 / (1.0 + ek)
                    k = rng.negative_binomial(1, p_vec, size=idx.size).astype(np.int32)
                    # clip: top stratum (5) uses party_ht_top_clip; others party_k_clip
                    is_top = lvl_safe == 5
                    if party_ht_top_clip > 0 and is_top.any():
                        k[is_top] = np.clip(k[is_top], 0, party_ht_top_clip)
                    bulk = ~is_top
                    if k_clip_party > 0 and bulk.any():
                        k[bulk] = np.clip(k[bulk], 0, k_clip_party)
                else:
                    k = rng.negative_binomial(1, nb_p, size=idx.size).astype(np.int32)
                    if k_clip_party > 0:
                        k = np.clip(k, 0, k_clip_party)

                tot = int(np.sum(k))
                if tot <= 0:
                    continue

                src_pos = np.repeat(idx, k).astype(np.int64, copy=False)
                rand = rng.integers(0, n - 1, size=tot, dtype=np.int64)
                partner_pos = rand + (rand >= src_pos)

                u = grp[src_pos].astype(np.int64, copy=False)
                v = grp[partner_pos].astype(np.int64, copy=False)

                src_all.append(u)
                dst_all.append(v)

            if not src_all:
                return np.empty(0, np.int64), np.empty(0, np.int64)

            u = np.concatenate(src_all)
            v = np.concatenate(dst_all)
            return _undirected_unique(u, v)

        def _gen_edges_mix_grouped(mix_pool: np.ndarray):
            """
            MIX = male-female only pairing.
            Generate UNIQUE undirected pairs (u,v) (u<->v later expanded to directed).

            Target: overall mean partners per attendee around `party_mix_target_mean`
                    (computed on undirected unique pairs).
            Notes:
              - If females < males (common), females will *necessarily* have higher mean degree.
                This function avoids extra algorithmic skew beyond that structural constraint.
              - Uses per-person "degree budgets" then random stub matching with de-dup.
            """
            if mix_pool.size < 3:
                return np.empty(0, np.int64), np.empty(0, np.int64)

            # ---- knobs (all optional) ----
            target_mean = float(self.params.get("party_mix_target_mean", 2.0))  # overall mean partners per attendee
            gmin = int(self.params.get("party_mix_group_min", 10))
            gmax = int(self.params.get("party_mix_group_max", 100))

            # hard safety caps (avoid pathological runtime in huge groups)
            max_pairs_cap = int(self.params.get("party_mix_max_pairs_per_group", 800))
            # degree caps (soft-ish): set None/<=0 to disable
            max_k_m = int(self.params.get("party_mix_max_k_m", 6))  # e.g. 6
            max_k_f = int(self.params.get("party_mix_max_k_f", 8))  # e.g. 8

            # sampling / rejection control
            max_attempts_mult = int(self.params.get("party_mix_max_attempts_mult", 50))  # * target_pairs
            batch_mult = int(self.params.get("party_mix_batch_mult", 6))

            mix_pool = mix_pool.copy()
            rng.shuffle(mix_pool)
            offsets = _make_offsets(mix_pool.size, gmin, gmax)

            u_all, v_all = [], []
            N = int(self.n)

            for i in range(offsets.size - 1):
                grp = mix_pool[offsets[i]:offsets[i + 1]]
                if grp.size < 3:
                    continue

                males = grp[self.genders[grp] == 0]
                females = grp[self.genders[grp] == 1]
                if males.size == 0 or females.size == 0:
                    continue

                # who has sex today
                m_act = males[rng.random(males.size) < p_sex]
                f_act = females[rng.random(females.size) < p_sex]
                if m_act.size == 0 or f_act.size == 0:
                    continue

                nm = int(m_act.size)
                nf = int(f_act.size)

                # target number of unique undirected pairs in this group
                target_pairs = int(round(target_mean * 0.5 * (nm + nf)))
                emax = nm * nf
                if target_pairs <= 0 or emax <= 0:
                    continue
                target_pairs = min(target_pairs, emax, max_pairs_cap)

                # --- Build per-person degree "budgets" on both sides ---
                # If we want total pairs = target_pairs, total degrees on each side must sum to target_pairs.
                # Start with Poisson around the natural means, then rescale to hit the sum.
                mu_m = target_pairs / max(nm, 1)
                mu_f = target_pairs / max(nf, 1)

                km = rng.poisson(mu_m, size=nm).astype(np.int32)
                kf = rng.poisson(mu_f, size=nf).astype(np.int32)

                # avoid all-zero edge case
                if km.sum() == 0:
                    km[rng.integers(0, nm)] = 1
                if kf.sum() == 0:
                    kf[rng.integers(0, nf)] = 1

                # optional caps
                if max_k_m and max_k_m > 0:
                    km = np.minimum(km, max_k_m).astype(np.int32)
                if max_k_f and max_k_f > 0:
                    kf = np.minimum(kf, max_k_f).astype(np.int32)

                sm = int(km.sum())
                sf = int(kf.sum())
                if sm <= 0 or sf <= 0:
                    continue

                # We need EXACTLY target_pairs stubs per side. Do a simple proportional rescale.
                # (This keeps the overall distribution shape but matches the edge count.)
                def _rescale_to_sum(k: np.ndarray, target_sum: int) -> np.ndarray:
                    s = int(k.sum())
                    if s <= 0:
                        return k
                    # scale
                    x = (k.astype(np.float64) * (target_sum / s))
                    k2 = np.floor(x).astype(np.int32)
                    # distribute remaining +1s by largest fractional parts
                    rem = target_sum - int(k2.sum())
                    if rem > 0:
                        frac = x - k2
                        idx = np.argsort(frac)[-rem:]
                        k2[idx] += 1
                    elif rem < 0:
                        # remove from largest k2 first
                        idx = np.argsort(k2)[::-1]
                        j = 0
                        need = -rem
                        while need > 0 and j < idx.size:
                            t = idx[j]
                            if k2[t] > 0:
                                dec = min(int(k2[t]), need)
                                k2[t] -= dec
                                need -= dec
                            j += 1
                    return k2

                # choose a feasible pair count <= caps-implied limits
                # (if caps too tight, target_pairs will drop)
                cap_lim = target_pairs
                if max_k_m and max_k_m > 0:
                    cap_lim = min(cap_lim, nm * max_k_m)
                if max_k_f and max_k_f > 0:
                    cap_lim = min(cap_lim, nf * max_k_f)
                if cap_lim <= 0:
                    continue
                target_pairs2 = cap_lim

                km = _rescale_to_sum(km, target_pairs2)
                kf = _rescale_to_sum(kf, target_pairs2)

                # stubs
                m_stub = np.repeat(np.arange(nm, dtype=np.int32), km).astype(np.int64, copy=False)
                f_stub = np.repeat(np.arange(nf, dtype=np.int32), kf).astype(np.int64, copy=False)

                if m_stub.size == 0 or f_stub.size == 0:
                    continue

                rng.shuffle(m_stub)
                rng.shuffle(f_stub)
                L = min(m_stub.size, f_stub.size)
                if L <= 0:
                    continue

                m_stub = m_stub[:L]
                f_stub = f_stub[:L]

                # --- Match stubs into unique pairs (reject duplicates) ---
                # We batch propose pairings by pairing shuffled stubs; duplicates are rejected with retries.
                # This is fast when target_pairs is not too close to nm*nf.
                seen = set()
                keys = np.empty(L, dtype=np.int64)
                fill = 0

                max_attempts = max(2000, max_attempts_mult * target_pairs2)
                attempts = 0

                # Start with a direct pairing from the current shuffled stubs
                while fill < target_pairs2 and attempts < max_attempts:
                    remaining = target_pairs2 - fill
                    batch = min(max(500, batch_mult * remaining), 200000, L)

                    # sample indices into stub arrays
                    idx = rng.integers(0, L, size=batch, dtype=np.int64)
                    mi = m_stub[idx]
                    fi = f_stub[idx]

                    m_ids = m_act[mi].astype(np.int64, copy=False)
                    f_ids = f_act[fi].astype(np.int64, copy=False)

                    a = np.minimum(m_ids, f_ids)
                    b = np.maximum(m_ids, f_ids)
                    k = a * N + b

                    # unique within batch
                    k_u, idx_u = np.unique(k, return_index=True)
                    mi_u = mi[idx_u]
                    fi_u = fi[idx_u]

                    for kk, mm, ff in zip(k_u, mi_u, fi_u):
                        if fill >= target_pairs2:
                            break
                        ik = int(kk)
                        if ik in seen:
                            continue
                        seen.add(ik)
                        keys[fill] = kk
                        fill += 1

                    attempts += batch

                if fill == 0:
                    continue

                keys = keys[:fill]
                u = (keys // N).astype(np.int64, copy=False)
                v = (keys % N).astype(np.int64, copy=False)

                u_all.append(u)
                v_all.append(v)

            if not u_all:
                return np.empty(0, np.int64), np.empty(0, np.int64)

            u = np.concatenate(u_all)
            v = np.concatenate(v_all)
            return _undirected_unique(u, v)

        # ---- FULL edges ----
        msm_u, msm_v = _gen_edges_same_pool_grouped(msm_att)
        wsw_u, wsw_v = _gen_edges_same_pool_grouped(wsw_att)
        mix_u, mix_v = _gen_edges_mix_grouped(mix_att)

        if (self.day % 14) == 0:
            # mix_u/mix_v 是 undirected unique pairs（你现在的实现）
            n_pairs = int(mix_u.size)
            n_att = int(mix_att.size)
            if n_att > 0:
                mean_partners_all = 2.0 * n_pairs / n_att  # 全体 attendee 的平均 partner 数
                print(f"[day {self.day}] MIX mean partners per attendee (unique pairs) = {mean_partners_all:.3f} "
                      f"(pairs={n_pairs}, att={n_att})")

        msm_src_full, msm_dst_full = _bi(msm_u, msm_v)
        wsw_src_full, wsw_dst_full = _bi(wsw_u, wsw_v)
        mix_src_full, mix_dst_full = _bi(mix_u, mix_v)

        sex_src_full = np.concatenate([msm_src_full, wsw_src_full, mix_src_full]) \
            if (msm_src_full.size + wsw_src_full.size + mix_src_full.size) else np.empty(0, np.int64)
        sex_dst_full = np.concatenate([msm_dst_full, wsw_dst_full, mix_dst_full]) \
            if (msm_dst_full.size + wsw_dst_full.size + mix_dst_full.size) else np.empty(0, np.int64)

        # ---- TX source + TX filter ----
        tx_source = (self.infectious | self.symptomatic).copy()

        if sex_src_full.size:
            tx_mask = tx_source[sex_src_full] | tx_source[sex_dst_full]
            sex_src_tx = sex_src_full[tx_mask]
            sex_dst_tx = sex_dst_full[tx_mask]
        else:
            tx_mask = None
            sex_src_tx = np.empty(0, np.int64)
            sex_dst_tx = np.empty(0, np.int64)

        self.last_sex_party_sex_src_full = sex_src_full
        self.last_sex_party_sex_dst_full = sex_dst_full
        self.last_sex_party_sex_tx_n = int(sex_src_tx.size)

        # -----------------------
        # DEBUG (every 14 days)
        # -----------------------
        if (self.day % 14) == 0:
            def _count(pool, gender=None, ori_val=None):
                if pool.size == 0:
                    return 0
                m = np.ones(pool.size, dtype=bool)
                if gender is not None:
                    m &= (self.genders[pool] == gender)
                if ori_val is not None:
                    m &= (self.orientations[pool] == ori_val)
                return int(np.sum(m))

            print(f"[day {self.day}] attend msm/wsw/mix = {msm_att.size}/{wsw_att.size}/{mix_att.size} | "
                  f"mix female={_count(mix_att, gender=1)} mix hetero={_count(mix_att, ori_val='hetero')} mix bi={_count(mix_att, ori_val='bi')} "
                  f"mix male_hetero={_count(mix_att, gender=0, ori_val='hetero')} mix female_hetero={_count(mix_att, gender=1, ori_val='hetero')}")

            def _frac(pool, mask):
                if pool.size == 0:
                    return 0.0
                return float(np.mean(mask[pool]))

            inf_sym = self.infectious | self.symptomatic
            inf_frac_msm = _frac(msm_att, inf_sym)
            inf_frac_mix = _frac(mix_att, inf_sym)
            mix_male = mix_att[self.genders[mix_att] == 0] if mix_att.size else np.empty(0, np.int64)
            inf_frac_mix_male = _frac(mix_male, inf_sym)
            print(
                f"[day {self.day}] infectious+symptomatic fraction: MSM={inf_frac_msm:.3%}, MIX={inf_frac_mix:.3%}, MIX_male={inf_frac_mix_male:.3%}")

            if mix_src_full.size:
                gsrc = self.genders[mix_src_full]
                gdst = self.genders[mix_dst_full]
                mf = int(np.sum((gsrc == 0) & (gdst == 1)))
                fm = int(np.sum((gsrc == 1) & (gdst == 0)))
                txn = int(np.sum(tx_mask)) if (tx_mask is not None) else 0
                print(
                    f"[day {self.day}] MIX edges(full, directed) count={mix_src_full.size} MF={mf} FM={fm} | tx_edges={txn}")

            # degree stats among MIX hetero females (on undirected edges)
            if mix_u.size:
                mix_het_f = mix_att[(self.genders[mix_att] == 1) & (self.orientations[mix_att] == 'hetero')]
                if mix_het_f.size:
                    deg = np.zeros(int(self.n), dtype=np.int32)
                    np.add.at(deg, mix_u, 1)
                    np.add.at(deg, mix_v, 1)
                    d = deg[mix_het_f]
                    print(f"[day {self.day}] MIX hetero_f degree: mean={float(np.mean(d)):.2f}, "
                          f"p50={int(np.median(d))}, p90={int(np.quantile(d, 0.90))}, "
                          f"p99={int(np.quantile(d, 0.99))}, max={int(np.max(d))}")

                # undirected bi_m - het_f edges count
                u = mix_u
                v = mix_v
                gu = self.genders[u];
                gv = self.genders[v]
                ou = self.orientations[u];
                ov = self.orientations[v]
                bridge = ((gu == 0) & (ou == 'bi') & (gv == 1) & (ov == 'hetero')) | (
                        (gv == 0) & (ov == 'bi') & (gu == 1) & (ou == 'hetero'))
                n_bridge = int(np.sum(bridge))
                print(f"[day {self.day}] MIX undirected bi_m-het_f edges={n_bridge} (of {u.size} undirected)")
        if (self.day % 14) == 0 and sex_src_tx.size:
            gsrc = self.genders[sex_src_tx]
            gdst = self.genders[sex_dst_tx]
            n_mf_tx = int(np.sum((gsrc == 0) & (gdst == 1)))
            n_fm_tx = int(np.sum((gsrc == 1) & (gdst == 0)))
            print(f"[day {self.day}] sex_party TX edges: total={sex_src_tx.size}, MF={n_mf_tx}, FM={n_fm_tx}")

        if (self.day % 14) == 0 and mix_att.size:
            males = mix_att[self.genders[mix_att] == 0]
            females = mix_att[self.genders[mix_att] == 1]
            print(f"[day {self.day}] MIX attendees: M={males.size}, F={females.size}, "
                  f"E_max(M*F)={males.size * females.size}")
        # ---- risks ONLY for tx edges ----
        if sex_src_tx.size:
            sp_puv, sp_pvu = build_edge_risks(
                sex_src_tx, sex_dst_tx,
                self.genders, self.sex_role,
                float(self.transmission_risks['sexual_anal_receptive']),
                float(self.transmission_risks['sexual_anal_insertive']),
                float(self.transmission_risks['sexual_vaginal']),
                float(self.transmission_risks['sexual_oral']),
                float(self.transmission_risks['close_touch'])
            )

            # MIX vaginal-only baseline already produced by build_edge_risks for mixed-sex edges.
            # Upgrade a fraction of MF/FM tx edges to anal (10% by default).
            gsrc = self.genders[sex_src_tx]
            gdst = self.genders[sex_dst_tx]
            is_mf = (gsrc == 0) & (gdst == 1)
            is_fm = (gsrc == 1) & (gdst == 0)
            is_mixed = is_mf | is_fm
            if np.any(is_mixed) and (mix_mf_anal_frac > 0 or mix_fm_anal_frac > 0):
                # Directional anal upgrade: MF (male->female) uses
                # mix_mf_anal_frac, FM (female->male) uses mix_fm_anal_frac.
                idx_mf = np.where(is_mf)[0]
                idx_fm = np.where(is_fm)[0]
                take = np.empty(0, np.int64)
                if idx_mf.size and mix_mf_anal_frac > 0:
                    take = np.concatenate([take, idx_mf[rng.random(idx_mf.size) < mix_mf_anal_frac]])
                if idx_fm.size and mix_fm_anal_frac > 0:
                    take = np.concatenate([take, idx_fm[rng.random(idx_fm.size) < mix_fm_anal_frac]])
                if np.any(take):
                    idxa = take.astype(np.int64, copy=False)
                    beta_rec = np.float32(self.transmission_risks['sexual_anal_receptive'])
                    beta_ins = np.float32(self.transmission_risks['sexual_anal_insertive'])
                    # MF / FM directional anal upgrade, split so that the
                    # female-as-receptive (male->female) anal fraction is
                    # independently controllable from the female-as-insertive
                    # (female->male) fraction. Biologically, receptive-anal
                    # transmission (beta_rec) dominates mpox sexual spread and
                    # applies to the receptive partner:
                    #   - is_mf (src=male, dst=female): female is receptive
                    #       p_uv (male->female) = beta_rec  (female receptive)
                    #       p_vu (female->male) = beta_ins  (male insertive)
                    #   - is_fm (src=female, dst=male): female is insertive
                    #       p_uv (female->male) = beta_ins  (female insertive)
                    #       p_vu (male->female) = beta_rec  (female receptive)
                    # 2026-11-04 bug fix: the is_fm branch previously assigned
                    # beta_rec to p_uv and beta_ins to p_vu (both reversed),
                    # over-estimating female->male transmission.
                    sel_mf = is_mf[idxa]
                    sel_fm = is_fm[idxa]
                    sp_puv[idxa[sel_mf]] = beta_rec
                    sp_pvu[idxa[sel_mf]] = beta_ins
                    sp_puv[idxa[sel_fm]] = beta_ins
                    sp_pvu[idxa[sel_fm]] = beta_rec
        else:
            sp_puv = np.empty(0, np.float32)
            sp_pvu = np.empty(0, np.float32)

        # ---- Behavior scaling schedule (apply to TX edges only) ----
        sc = self._sexual_scale_today()
        if sc < 1.0 and sex_src_tx.size:
            m = rng.random(sex_src_tx.size) < sc
            sex_src_tx, sex_dst_tx = sex_src_tx[m], sex_dst_tx[m]
            sp_puv, sp_pvu = sp_puv[m], sp_pvu[m]

        # ---- Close-touch edges for party (full edges -> sample -> then TX filter) ----
        tp_src_chunks, tp_dst_chunks = [], []

        def _touch_from_edges(src_full, dst_full, frac):
            if src_full.size == 0 or frac <= 0:
                return
            keep = rng.random(src_full.size) < frac
            if np.any(keep):
                tp_src_chunks.append(src_full[keep].astype(np.int64, copy=False))
                tp_dst_chunks.append(dst_full[keep].astype(np.int64, copy=False))

        _touch_from_edges(msm_src_full, msm_dst_full, touch_msm)
        _touch_from_edges(wsw_src_full, wsw_dst_full, touch_wsw)
        _touch_from_edges(mix_src_full, mix_dst_full, touch_mix)

        if tp_src_chunks:
            tp_src_full = np.concatenate(tp_src_chunks)
            tp_dst_full = np.concatenate(tp_dst_chunks)

            tp_mask = tx_source[tp_src_full] | tx_source[tp_dst_full]
            tp_src_tx = tp_src_full[tp_mask]
            tp_dst_tx = tp_dst_full[tp_mask]

            if tp_src_tx.size:
                tp_puv = np.full(tp_src_tx.size, float(self.transmission_risks['close_touch']), dtype=np.float32)
                tp_pvu = tp_puv.copy()
            else:
                tp_src_tx = np.empty(0, np.int64)
                tp_dst_tx = np.empty(0, np.int64)
                tp_puv = np.empty(0, np.float32)
                tp_pvu = np.empty(0, np.float32)

            self.last_sex_party_touch_src_full = tp_src_full
            self.last_sex_party_touch_dst_full = tp_dst_full
            self.last_sex_party_touch_tx_n = int(tp_src_tx.size)
        else:
            tp_src_tx = np.empty(0, np.int64)
            tp_dst_tx = np.empty(0, np.int64)
            tp_puv = np.empty(0, np.float32)
            tp_pvu = np.empty(0, np.float32)
            self.last_sex_party_touch_src_full = tp_src_tx
            self.last_sex_party_touch_dst_full = tp_dst_tx
            self.last_sex_party_touch_tx_n = 0

        return (sex_src_tx, sex_dst_tx, sp_puv, sp_pvu,
                tp_src_tx, tp_dst_tx, tp_puv, tp_pvu,
                int(msm_att.size), int(wsw_att.size), int(mix_att.size))


    def _derive_infectious_start_day(self, exposure_day, incubation_period, symptom_onset_delay):
        """Derive infectious start from clinical-onset timing.

        Definitions:
          incubation_period   = exposure -> rash/symptom onset
          symptom_onset_delay = infectious start -> rash/symptom onset

        Therefore:
          infectious_start_day = exposure_day + incubation_period - symptom_onset_delay
        """
        return np.maximum(
            exposure_day,
            (exposure_day + incubation_period - symptom_onset_delay).astype(int)
        )

    # Updated seed_infection method
    def seed_infection(self, n_infected: int = 10, mask=None):
        elig = np.where(mask)[0] if mask is not None else self.uids
        if len(elig) < n_infected:
            raise ValueError('Not enough eligible people to seed.')
        # Use self._rng (NOT global np.random) so seed selection is fully
        # reproducible and decoupled from any global-RNG state left behind by
        # population/network construction or cache load paths. This is the
        # single point of divergence that broke seed-to-seed reproducibility:
        # transmission runs on self._rng, but seed selection used global
        # np.random, so the two streams drifted whenever construction differed.
        idx = self._rng.choice(elig, n_infected, replace=False)
        self.susceptible[idx] = False
        self.exposed[idx] = True
        self.date_exposed[idx] = self.day

        # Disease timing (natural-history parameters are LITERATURE-FIXED;
        # values read from self.params so main.py is the single source):
        #   incubation_period      = exposure -> rash/symptom onset
        #   symptom_onset_delay    = infectious start -> rash/symptom onset
        #   infectious_start_day   = exposure + incubation_period - symptom_onset_delay
        _inc_mu = float(self.params.get("incubation_lognorm_mu", 2.108))
        _inc_sig = float(self.params.get("incubation_lognorm_sigma", 0.254))
        _inc_clip = self.params.get("incubation_clip", (3, 17))
        _delay_lo, _delay_hi = self.params.get("symptom_onset_delay_range", (1, 4))
        _inf_shape = float(self.params.get("infectious_period_gamma_shape", 9.0))
        _inf_scale = float(self.params.get("infectious_period_gamma_scale", 2.33))
        _inf_clip = self.params.get("infectious_period_clip", (14, 28))

        self.incubation_period[idx] = np.clip(self._rng.lognormal(_inc_mu, _inc_sig, size=n_infected), *_inc_clip)
        self.symptom_onset_delay[idx] = self._rng.uniform(_delay_lo, _delay_hi, size=n_infected)
        self.infectious_period[idx] = np.clip(self._rng.gamma(_inf_shape, _inf_scale, size=n_infected), *_inf_clip)

        rash_or_symptom_onset_day = self.date_exposed[idx] + self.incubation_period[idx]
        self.infectious_start_day[idx] = np.maximum(
            self.date_exposed[idx],
            (rash_or_symptom_onset_day - self.symptom_onset_delay[idx]).astype(int)
        )

    def _vaccinate_by_coverage(self) -> None:
        """
        Give first- and second-dose vaccinations so that the *effective*
        (≥14 d old) coverage among MCP-eligible people follows the cumulative
        proportions in ``self._first_cov`` and ``self._second_cov``.
        Called once at the top of ``step()``.
        """
        week = min(self.day // 7, len(self._first_cov) - 1)

        # --- size of the denominator = everyone who could ever be offered a dose
        mcp_pool = np.where(self.mcp_flags)[0]
        N = mcp_pool.size
        if N == 0:  # unlikely, but guard anyway
            return

        rng = self._rng  # shortcut

        # ======================================================================
        # 1) FIRST DOSE ---------------------------------------------------------
        # ======================================================================
        target1 = int(np.ceil(self._first_cov[week] * N))  # cumulative %
        have1 = np.sum((self.vax_dose >= 1) & self.mcp_flags)  # any day
        need1 = max(0, target1 - have1)
        if need1:
            # ─ eligible for 1st dose = MCP and still unvaccinated
            elig_m = mcp_pool[(self.genders[mcp_pool] == 0) &
                              (self.vax_dose[mcp_pool] == 0)]
            elig_f = mcp_pool[(self.genders[mcp_pool] == 1) &
                              (self.vax_dose[mcp_pool] == 0)]

            # sex split taken from CDC first-dose data
            n_m = int(round(0.915 * need1))
            n_f = need1 - n_m

            pick_m = rng.choice(elig_m, min(n_m, elig_m.size), replace=False)
            pick_f = rng.choice(elig_f, min(n_f, elig_f.size), replace=False)
            chosen = np.concatenate((pick_m, pick_f))

            self.vax_dose[chosen] = 1
            self.vax1_day[chosen] = self.day
            eff_day = self.day + 14
            self._dose1_eff.setdefault(eff_day, []).extend(chosen.tolist())

        # ======================================================================
        # 2) SECOND DOSE --------------------------------------------------------
        # ======================================================================
        target2 = int(np.ceil(self._second_cov[week] * N))
        have2 = np.sum((self.vax_dose >= 2) & self.mcp_flags)
        need2 = max(0, target2 - have2)
        if need2:
            # ─ eligible for 2nd dose = got 1st ≥28 d ago and none yet
            eligible = np.where((self.vax_dose == 1) &
                                ((self.day - self.vax1_day) >= 28))[0]
            elig_m = eligible[self.genders[eligible] == 0]
            elig_f = eligible[self.genders[eligible] == 1]

            n_m = int(round(0.94 * need2))  # CDC second-dose sex split
            n_f = need2 - n_m

            pick_m = rng.choice(elig_m, min(n_m, elig_m.size), replace=False)
            pick_f = rng.choice(elig_f, min(n_f, elig_f.size), replace=False)
            chosen = np.concatenate((pick_m, pick_f))

            self.vax_dose[chosen] = 2
            self.vax2_day[chosen] = self.day
            eff_day = self.day + 14
            self._dose2_eff.setdefault(eff_day, []).extend(chosen.tolist())

    def update_second_doses(self):
        for p in self.people:
            if p.vax_dose == 1 and (self.day - p.vax_day) >= 28:
                p.vax_dose = 2

    def step(self):
        """
        One full day:
        1) update all dynamic networks
        2) run transmission for each layer
        3) advance exposed→infectious by incubation
        4) draw symptoms
        5) finish infections (recovery or death) at end of infectious period
        6) log daily stats and increment day
        """
        # apply time-varying policy and importations for this day (before testing / transmission)
        d = self.day
        self._apply_policy(d)
        self._apply_importations(d)

        # ── Test & Trace first ─────────────────────────────────────
        self._run_test_trace()
        # ── Vaccination schedule … ──────────────────────────────
        self._vaccinate_by_coverage()

        # ── 0) reset susceptibility to baseline ───────────────────
        # (this wipes out yesterday’s multipliers)
        if self.day == 0:
            self.r_sus[:] = self._r_sus_base

        # dose1 effectiveness today?
        new1 = self._dose1_eff.pop(self.day, None)
        if new1:
            arr = np.array(new1, dtype=np.int32)
            self.r_sus[arr] = self._r_sus_base[arr] * 0.63

        # dose2 effectiveness today?
        new2 = self._dose2_eff.pop(self.day, None)
        if new2:
            arr = np.array(new2, dtype=np.int32)
            self.r_sus[arr] = self._r_sus_base[arr] * 0.31

        # ── 2) now build layers, run transmissions with self.r_sus … ──
        self.new_exposed.fill(False)
        self.new_diagnosed.fill(False)
        # ── 1. intervention mask (quarantine / isolation) ────────────────
        # Full contact removal applies only to isolated diagnosed/symptomatic cases.
        # Traced asymptomatic contacts are handled below as monitored contacts with
        # layer-specific partial risk/contact reduction.
        self._mask_inactive = self.isolated

        # ── 2. build today’s dynamic layers  ─────────────────────────────

        sex_src, sex_dst, sex_puv, sex_pvu = self._update_sexual_network()  # ← new daily sexual contacts
        o_src, o_dst, o_puv, o_pvu = self._update_oneoff_layer()  # your one-off sexual layer
        # sm_src, sm_dst, sm_beta, lg_src, lg_dst, lg_beta = self._update_small_large_event_networks()  # if you have these
        sp_src, sp_dst, sp_puv, sp_pvu, tp_src, tp_dst, tp_puv, tp_pvu, msm_n, wsw_n, mix_n = self._update_sex_party_network()
        # right after building dynamic layers
        self.sex_src_today, self.sex_dst_today = sex_src, sex_dst
        self.sp_src_today, self.sp_dst_today = sp_src, sp_dst
        # --- household close-contact layer (non-sexual, within stable partner households) ---
        hh_close_src, hh_close_dst, hh_close_puv, hh_close_pvu = self._update_household_close_layer()
        self.hh_close_src_today, self.hh_close_dst_today = hh_close_src, hh_close_dst

        # --- social close-contact layer (friendship / casual close contact) ---
        soc_src, soc_dst, soc_puv, soc_pvu = self._update_social_layer()
        self.soc_src_today, self.soc_dst_today = soc_src, soc_dst

        # ── 3. run transmission on every layer  ──────────────────────────
        # store ALL relevant edges for tracing on the next day (ordering-safe)
        src_list = [a for a in (sex_src, hh_close_src, soc_src, o_src, sp_src, tp_src) if a.size]
        dst_list = [a for a in (sex_dst, hh_close_dst, soc_dst, o_dst, sp_dst, tp_dst) if a.size]

        if src_list:
            self._trace_src_prev = np.concatenate(src_list)
            self._trace_dst_prev = np.concatenate(dst_list)
        else:
            self._trace_src_prev = np.empty(0, np.int64)
            self._trace_dst_prev = np.empty(0, np.int64)

        # per-layer edge store for layer-specific tracing probabilities
        self._trace_prev_by_layer = {}
        if hh_close_src.size: self._trace_prev_by_layer[0] = (hh_close_src, hh_close_dst)
        if sex_src.size:      self._trace_prev_by_layer[1] = (sex_src, sex_dst)
        if o_src.size:        self._trace_prev_by_layer[2] = (o_src, o_dst)
        if sp_src.size:       self._trace_prev_by_layer[3] = (sp_src, sp_dst)
        if tp_src.size:       self._trace_prev_by_layer[4] = (tp_src, tp_dst)
        if soc_src.size:      self._trace_prev_by_layer[5] = (soc_src, soc_dst)

        # self._fast_transmit(self.hh_src,self.hh_dst, self.hh_beta, 'household')
        # self._fast_transmit(self.sch_src, self.sch_dst, self.sch_beta, 'school')
        # self._fast_transmit(self.wp_src, self.wp_dst, self.wp_beta, 'workplace')
        # self._fast_transmit(self.so_src, self.so_dst, self.so_beta, 'social')

        # apply policy multipliers to directional p (do NOT mutate cached arrays)
        if self.scale_sexual != 1.0:
            sex_puv = sex_puv * self.scale_sexual
            sex_pvu = sex_pvu * self.scale_sexual
        if self.scale_oneoff != 1.0:
            o_puv = o_puv * self.scale_oneoff
            o_pvu = o_pvu * self.scale_oneoff
        if self.scale_sex_party != 1.0:
            sp_puv = sp_puv * self.scale_sex_party
            sp_pvu = sp_pvu * self.scale_sex_party

        if self.scale_household_close != 1.0:
            hh_close_puv = hh_close_puv * self.scale_household_close
            hh_close_pvu = hh_close_pvu * self.scale_household_close
        if self.scale_social != 1.0:
            soc_puv = soc_puv * self.scale_social
            soc_pvu = soc_pvu * self.scale_social

        self._fast_transmit(sex_src, sex_dst, sex_puv, sex_pvu, 'stable', self._condom_rate_for('stable'))
        self._fast_transmit(hh_close_src, hh_close_dst, hh_close_puv, hh_close_pvu, 'household_close', 0.0)
        self._fast_transmit(soc_src, soc_dst, soc_puv, soc_pvu, 'social', 0.0)
        self._fast_transmit(o_src, o_dst, o_puv, o_pvu, 'oneoff', self._condom_rate_for('oneoff'))
        self._fast_transmit(sp_src, sp_dst, sp_puv, sp_pvu, 'sex_party', self._condom_rate_for('sex_party'))
        self._fast_transmit(tp_src, tp_dst, tp_puv, tp_pvu, 'sex_party_touch', 0.0)

        if self.verbose and (self.day % self.print_every == 0):
            print(
                f"""Day {self.day}: size of sexual is {sex_src.size} edges, mean beta is {np.mean(sex_puv):.3f} and {np.mean(sex_pvu):.3f}""")
        if self.verbose and (self.day % self.print_every == 0):
            print(
                f"""Day {self.day}: size of oneoff is {o_src.size} edges, mean beta is {np.mean(o_puv):.3f} and {np.mean(o_pvu):.3f}""")
        # print(f"""Day {self.day}: size of small_event is {sm_src.size} edges, mean beta is {np.mean(sm_beta):.3f}""")
        # print(f"""Day {self.day}: size of large_event is {lg_src.size} edges, mean beta is {np.mean(lg_beta):.3f}""")
        # print(f"""Day {self.day}: size of sex_party is {sp_src.size} edges, mean beta is {np.mean(sp_beta):.3f}""")
        # self._fast_transmit(tp_src,tp_dst,tp_beta,'sex_party_touch')
        self._today_edges = lambda: [
            # (self.hh_src, self.hh_dst),
            # (self.sch_src, self.sch_dst),
            # (self.wp_src, self.wp_dst),
            # (self.so_src, self.so_dst),
            (sex_src, sex_dst),
            (hh_close_src, hh_close_dst),
            (soc_src, soc_dst),
            (o_src, o_dst),
            # (sm_src, sm_dst),
            # (lg_src, lg_dst),
            (sp_src, sp_dst),
            (tp_src, tp_dst),
        ]
        # ── 4. disease progression (very brief)  ─────────────────────────
        # 4a. exposed → infectious when day >= infectious_start_day
        newly_inf = self.exposed & (self.day >= self.infectious_start_day)
        if newly_inf.any():
            self.infectious[newly_inf] = True
            self.exposed[newly_inf] = False
            self.new_infectious[newly_inf] = True

        # 4b. rash/symptom onset
        # incubation_period is defined as exposure -> rash/symptom onset.
        # symptom_onset_delay is already used to derive infectious_start_day
        # and must not be added again here.
        sym_day = self.day >= (self.date_exposed + self.incubation_period).astype(int)

        new_sym = self.infectious & ~self.symptomatic & sym_day

        if new_sym.any():
            self.symptomatic[new_sym] = (self._rng.random(new_sym.sum())
                                         < self.p_sym[new_sym])
            new_sym_idx = np.where(new_sym & self.symptomatic)[0].astype(np.int64, copy=False)
            self.new_symptomatic[new_sym_idx] = True
            self.symptom_onset_day[new_sym_idx] = self.day

            # In scheduled-delay mode, assign test-seeking and result dates at
            # rash/symptom onset.  This directly represents symptom/rash onset
            # -> specimen/test seeking -> laboratory result, instead of drawing
            # a new daily testing Bernoulli trial.
            if self.test_delay_mode != "daily_prob" and new_sym_idx.size:
                self._schedule_tests_for_symptomatic(
                    new_sym_idx,
                    test_pathway_prob=self.test_pathway_prob_symp,
                    test_seek_delay_mean_days=self.test_seek_delay_mean_days,
                    test_seek_delay_sd_days=self.test_seek_delay_sd_days,
                    lab_turnaround_days=self.lab_turnaround_days_today,
                )
        # 4c. end of infectious period: recover or die
        finishing = self.infectious & (self.day >= (self.infectious_start_day + self.infectious_period).astype(int))
        if finishing.any():
            roll = self._rng.random(finishing.sum())
            idx = np.where(finishing)[0]
            die_m = roll < self.p_dea[finishing]
            deaths = idx[die_m]
            recov = idx[~die_m]

            if recov.size:
                self.infectious[recov] = False
                self.recovered[recov] = True
                self.new_recovered[recov] = True
            if deaths.size:
                self.infectious[deaths] = False
                self.dead[deaths] = True
                self.dead_today[deaths] = True
        # ── 5. orientation-specific incidence & party attendance  ─────────────────────────────────────────
        for ori in ['hetero', 'bi', 'homo']:
            mask = self.orientations == ori
            self.orientation_daily[ori].append(
                int(np.sum(self.new_exposed & mask))
            )

        # Calibration-only daily incidence by sex and sexcat.  This makes it
        # possible to compare model outputs with observed table columns such as
        # Male-Homo, Male-Bisexual, Male-Heterosexual, and Female-Heterosexual.
        self.sex_daily["male"].append(int(np.sum(self.new_exposed & (self.genders == 0))))
        self.sex_daily["female"].append(int(np.sum(self.new_exposed & (self.genders == 1))))
        self.sexcat_daily["female_hetero"].append(int(np.sum(self.new_exposed & (self.sexcat == 0))))
        self.sexcat_daily["male_hetero"].append(int(np.sum(self.new_exposed & (self.sexcat == 1))))
        self.sexcat_daily["male_homo"].append(int(np.sum(self.new_exposed & (self.sexcat == 2))))
        self.sexcat_daily["female_homo"].append(int(np.sum(self.new_exposed & (self.sexcat == 3))))
        self.sexcat_daily["male_bi"].append(int(np.sum(self.new_exposed & (self.sexcat == 4))))
        self.sexcat_daily["female_bi"].append(int(np.sum(self.new_exposed & (self.sexcat == 5))))

        self.sex_party_daily['MSM'].append(msm_n)
        self.sex_party_daily['WSW'].append(wsw_n)
        self.sex_party_daily['MIX'].append(mix_n)
        # --- 6.overall daily totals-----------------------------------
        total_day = int(self.new_exposed.sum())
        self.infections_by_day["total"].append(total_day)
        self._update_rt_total()

        self.daily_stats.append({
            'day': self.day,
            'susceptible': int(self.susceptible.sum()),
            'exposed': int(self.exposed.sum()),
            'infectious': int(self.infectious.sum()),
            'symptomatic': int(self.symptomatic.sum()),
            'recovered': int(self.recovered.sum()),
            'dead': int(self.dead.sum())
        })

        # advance clock
        self.day += 1

    def _condom_rate_for(self, layer: str) -> float:
        """Return the fraction of acts protected by condoms on this layer."""
        return {'stable': 0.30,  # long-term partners
                'oneoff': 0.30,  # casual hook-ups
                'sex_party': 0.50}.get(layer, 0.0)  # higher at parties

    # ────────────────────────────────────────────────────────────────────
    #  Fast, vectorised, bidirectional transmission
    # ────────────────────────────────────────────────────────────────────
    def _policy_for_day(self, d: int) -> dict:
        """Return the policy dict active on day d (inclusive bounds), else empty dict."""
        for ph in self.policy:
            if ph.get('start', -1) <= d <= ph.get('end', -2):
                return ph
        return {}

    def _apply_policy(self, d: int) -> None:
        """Update daily multipliers for transmission/testing/tracing.

        The same phase dictionary controls both behaviour/contact multipliers
        and testing/tracing parameters.  Bounds are inclusive and are normally
        supplied from main.py as:
          Phase I   day 0-91   = May 1-Jul 31, 2022
          Phase II  day 92-170 = Aug 1-Oct 18, 2022
          Phase III day 171+   = Oct 19-Dec 31, 2022
        """
        ph = self._policy_for_day(d)
        self._policy_today = ph
        contact_mult = float(ph.get('contact_mult', 1.0))
        oneoff_mult = float(ph.get('oneoff_mult', 1.0))
        party_mult = float(ph.get('party_mult', 1.0))
        test_mult = float(ph.get('test_mult', 1.0))
        trace_mult = float(ph.get('trace_mult', 1.0))

        self.scale_sexual = contact_mult
        self.scale_household_close = contact_mult
        self.scale_oneoff = oneoff_mult
        self.scale_sex_party = party_mult
        self.scale_social = float(ph.get('social_mult', contact_mult))

        # Testing controls.  In scheduled-delay mode, policy phases directly
        # set the probability of entering the testing pathway and the mean
        # rash/symptom-onset-to-specimen delay.  If explicit phase values are
        # absent, test_mult shortens the baseline delay.
        self.test_prob_symp = min(1.0, self.base_test_prob_symp * test_mult)  # legacy daily_prob mode
        self.test_pathway_prob_symp = float(np.clip(
            ph.get('test_pathway_prob', self.base_test_pathway_prob_symp), 0.0, 1.0
        ))
        self.test_seek_delay_mean_days = float(ph.get(
            'test_seek_delay_mean_days',
            self.base_test_seek_delay_mean_days / max(test_mult, 1e-6)
        ))
        self.test_seek_delay_sd_days = float(ph.get(
            'test_seek_delay_sd_days', self.base_test_seek_delay_sd_days
        ))
        # Optional time-varying reporting delay (symptom -> specimen/report).
        # Literature: Hinch 2024 (Sci Rep) symptom->presentation 22d (early May)
        # -> 10d (early Jun) -> 8d (Aug); NYC nowcasting (JMIR 2025) onset->
        # report 10d (Jul) -> 8d (Sep); Tennessee (Public Health Rep 2024)
        # symptom->specimen 5d + specimen->report 3d. When enabled, the
        # phase-specific mean delay is overridden by a continuous curve from
        # report_delay_t0 (day 0) decaying to report_delay_t1 (late), which
        # delays early-case recording and shifts the recorded epidemic curve
        # later toward the observed LAC peak.
        if self.params.get("report_delay_time_varying", False):
            _t0 = float(self.params.get("report_delay_t0", 12.0))
            _t1 = float(self.params.get("report_delay_t1", 5.0))
            _tau = float(self.params.get("report_delay_tau", 40.0))
            _w = float(np.exp(-d / _tau))
            self.test_seek_delay_mean_days = _t1 + (_t0 - _t1) * _w
        self.lab_turnaround_days_today = int(round(ph.get(
            'lab_turnaround_days', self.base_lab_turnaround_days
        )))

        self.trace_prob_edge = min(1.0, self.base_trace_prob_edge * trace_mult)

    def _apply_importations(self, d: int) -> int:
        """Apply exogenous importations on day d. Returns number of newly exposed."""
        if not self.importations:
            return 0
        n_new = 0
        for imp in self.importations:
            if imp.get('start', -1) <= d <= imp.get('end', -2):
                k = int(imp.get('daily_n', 0))
                if k <= 0:
                    continue
                if self.importation_mask is None:
                    cand = np.where(self.susceptible)[0]
                else:
                    cand = np.where(self.susceptible & self.importation_mask)[0]
                if cand.size == 0:
                    continue
                take = min(k, cand.size)
                idx = self._rng.choice(cand, size=take, replace=False)
                self._expose_batch(idx)
                n_new += take
        return n_new

    def _expose_batch(self, idx: np.ndarray):
        """Vectorised timer stamping for newly exposed people."""
        self.susceptible[idx] = False
        self.exposed[idx] = True
        self.new_exposed[idx] = True

        self.date_exposed[idx] = self.day

        # incubation until infectious (natural-history params, literature-fixed)
        _inc_mu = float(self.params.get("incubation_lognorm_mu", 2.108))
        _inc_sig = float(self.params.get("incubation_lognorm_sigma", 0.254))
        _inc_clip = self.params.get("incubation_clip", (3, 17))
        ip = np.clip(self._rng.lognormal(_inc_mu, _inc_sig, idx.size), *_inc_clip)
        self.incubation_period[idx] = ip

        # symptom onset delay after infectious start
        _delay_lo, _delay_hi = self.params.get("symptom_onset_delay_range", (1, 4))
        self.symptom_onset_delay[idx] = self._rng.uniform(_delay_lo, _delay_hi, idx.size)

        # calendar day they *start* shedding
        inf_start = (self.day + ip - self.symptom_onset_delay[idx]).astype(int)
        # clamp to ≥ tomorrow just in case lead > inc
        inf_start = np.maximum(inf_start, self.day + 1)
        self.infectious_start_day[idx] = inf_start

        # infectious period duration
        _inf_shape = float(self.params.get("infectious_period_gamma_shape", 9.0))
        _inf_scale = float(self.params.get("infectious_period_gamma_scale", 2.33))
        _inf_clip = self.params.get("infectious_period_clip", (14, 28))
        inf_dur = np.clip(self._rng.gamma(_inf_shape, _inf_scale, idx.size), *_inf_clip)
        self.infectious_period[idx] = inf_dur

    def _txlog_append(self, layer_code: int, src: np.ndarray, dst: np.ndarray, p: np.ndarray):
        """Append successful transmissions into ring-like fixed buffers."""
        if src.size == 0:
            return
        j0 = int(self.txlog_i)
        if j0 >= int(self.txlog_max):
            return
        k = int(src.size)
        j1 = min(j0 + k, int(self.txlog_max))
        take = j1 - j0
        if take <= 0:
            return

        self.txlog_day[j0:j1] = self.day
        self.txlog_layer[j0:j1] = layer_code
        self.txlog_src[j0:j1] = src[:take]
        self.txlog_dst[j0:j1] = dst[:take]
        self.txlog_p[j0:j1] = p[:take]
        self.txlog_i = j1


    def _monitor_contact_mult_for_layer(self, layer: str) -> float:
        """Return contact-retention multiplier for monitored traced contacts.

        Monitoring is CDC-aligned symptom monitoring/risk reduction, not
        mandatory quarantine. The multiplier is applied when either endpoint
        of an edge is currently monitored.
        """
        if layer in ("stable", "oneoff", "sex_party", "sex_party_touch"):
            return self.monitor_sexual_contact_mult
        if layer in ("household_close", "social"):
            return self.monitor_social_contact_mult
        return 1.0

    def _fast_transmit(self,
                       src: np.ndarray,
                       dst: np.ndarray,
                       p_uv: np.ndarray,
                       p_vu: np.ndarray,
                       layer: str,
                       condom_rate: float | None = None,
                       condom_eff: float = 0.50) -> int:
        """
        Run transmission for one contact layer.
        Stores each undirected edge once; kernel applies both directions.

        Returns
        -------
        int  – number of new exposures generated on this layer today
        """

        if src.size == 0:  # no contacts, quick exit
            self.infections_by_day[layer].append(0)
            return 0

        # ---- monitoring/risk-reduction effect for traced asymptomatic contacts ----
        # Monitored contacts are not fully removed from the network. Instead,
        # if either endpoint is monitored, the layer-specific multiplier reduces
        # effective contact/transmission opportunities on that edge.
        monitor_mult = self._monitor_contact_mult_for_layer(layer)
        if monitor_mult < 1.0 and getattr(self, "monitored", None) is not None:
            monitored_edge = self.monitored[src] | self.monitored[dst]
            if np.any(monitored_edge):
                p_uv = p_uv.copy()
                p_vu = p_vu.copy()
                p_uv[monitored_edge] *= monitor_mult
                p_vu[monitored_edge] *= monitor_mult

        # ---- choose condom uptake if caller left it to default ----------
        if condom_rate is None:
            condom_rate = self._condom_rate_for(layer)
        if condom_rate > 0.0:
            mask = self._rng.random(src.size) < condom_rate  # Boolean
            p_uv = p_uv.copy()
            p_vu = p_vu.copy()
            p_uv[mask] *= (1.0 - condom_eff)
            p_vu[mask] *= (1.0 - condom_eff)

        # ---- one random draw per edge, passed into Numba kernel ---------
        roll_uv = self._rng.random(src.size).astype(np.float32)
        roll_vu = self._rng.random(src.size).astype(np.float32)
        # new_ids = fast_tx_core(src,
        #                        dst,
        #                        p_uv.astype(np.float32, copy=False),
        #                        p_vu.astype(np.float32, copy=False),
        #                        self.infectious,
        #                        self.susceptible,
        #                        self.dead,
        #                        self.r_sus,  # r_sus vector
        #                        self._mask_inactive,  # quarantine / iso mask
        #                        roll)
        # ---- stamp timers & update states -------------------------------
        # if new_ids.size:
        #     self._expose_batch(new_ids)  # sets self.new_exposed flag too
        #
        # # ---- layering bookkeeping ---------------------------------------
        # n_new = int(new_ids.size)
        # self.layer_infection_counts[layer] += n_new
        # self.infections_by_day[layer].append(n_new)
        # return n_new
        succ_uv, succ_vu = fast_tx_flags(src.astype(np.int64, copy=False),
                                         dst.astype(np.int64, copy=False),
                                         p_uv.astype(np.float32, copy=False),
                                         p_vu.astype(np.float32, copy=False),
                                         self.infectious,
                                         self.susceptible,
                                         self.dead,
                                         self.r_sus,
                                         self._mask_inactive,
                                         roll_uv,
                                         roll_vu)

        # 4) 从“按边成功”还原出“按人新发”，再一次去重（O(E)）
        if succ_uv.any() or succ_vu.any():
            ids = np.concatenate((dst[succ_uv], src[succ_vu])).astype(np.int64, copy=False)  #
            new_ids = dedup_by_day(ids.astype(np.int64, copy=False), self._seen_day, self.day)  # 小批量，代价远低于扫描 1e7
            if new_ids.size:
                self._expose_batch(new_ids)
            n_new = int(new_ids.size)
        else:
            n_new = 0

        self.layer_infection_counts[layer] += n_new
        self.infections_by_day[layer].append(n_new)

        # layer -> int code (keep simple, or use your own mapping)
        layer_map = getattr(self, "txlog_layer_map", None)
        if layer_map is None:
            # default mapping; you can adjust
            layer_map = {"stable": 1, "oneoff": 2, "sex_party": 3, "household_close": 4, "sex_party_touch": 5,
                         "social": 6}
        layer_code = int(layer_map.get(layer, 0))

        # succ_vu means dst -> src succeeded using p_vu (reverse direction!)
        if np.any(succ_vu):
            self._txlog_append(layer_code, dst[succ_vu].astype(np.int64, copy=False),
                               src[succ_vu].astype(np.int64, copy=False),
                               p_vu[succ_vu].astype(np.float32, copy=False))
        if np.any(succ_uv):
            self._txlog_append(layer_code, src[succ_uv].astype(np.int64, copy=False),
                               dst[succ_uv].astype(np.int64, copy=False),
                               p_uv[succ_uv].astype(np.float32, copy=False))

        return n_new

    def run_simulation(self, days=90):
        for _ in range(days):
            self.step()

    def empty_layer(self):
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32))

    # ---------------------------
    # R_t (effective reproduction number)
    # ---------------------------
    @staticmethod
    def _make_gamma_gi_pmf(mean: float, sd: float, max_lag: int) -> np.ndarray:
        """Discrete generation-interval PMF over days 1..max_lag using a Gamma(mean, sd).

        We approximate P(GI = s) ∝ gamma_pdf(s) for integer day s and renormalize.
        This is lightweight (no SciPy dependency) and sufficient for online Cori R_t.
        """
        if max_lag < 1:
            raise ValueError("max_lag must be >= 1")
        if mean <= 0 or sd <= 0:
            raise ValueError("mean and sd must be > 0")

        k = (mean / sd) ** 2  # shape
        theta = (sd ** 2) / mean  # scale

        x = np.arange(1, max_lag + 1, dtype=np.float64)
        # log-pdf for numerical stability: (k-1)log x - x/theta - k log theta - lgamma(k)
        logpdf = (k - 1.0) * np.log(x) - (x / theta) - k * np.log(theta) - math.lgamma(k)
        pdf = np.exp(logpdf)

        # guard against underflow / degenerate cases
        s = pdf.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback to uniform weights over lags
            pdf = np.ones_like(pdf)
            s = pdf.sum()

        w = np.zeros(max_lag + 1, dtype=np.float64)  # w[0]=0 by definition
        w[1:] = pdf / s
        return w

    def _update_rt_total(self) -> None:
        """Update R_t for the most recent day using infections_by_day['total']."""
        inc = self.infections_by_day["total"]
        t = len(inc) - 1
        tau = max(1, self.rt_smooth_window)
        # burn-in 或历史不足：不计算
        if t < max(self.rt_burn_in, tau):
            self.rt_by_day["total"].append(np.nan)
            return
        start = t - tau + 1
        num = float(sum(inc[start:t + 1]))

        # Early in the epidemic, incidence is tiny and Cori R_t can explode; suppress until enough cases.
        if num < self.rt_min_cases:
            self.rt_by_day["total"].append(np.nan)
            return

        denom = 0.0
        for k in range(start, t + 1):
            lam_k = 0.0
            max_s = min(k, self.gi_max)

            for s in range(1, max_s + 1):
                lam_k += inc[k - s] * self.gi_pmf[s]
            denom += lam_k

        a = self.rt_prior_a
        b = self.rt_prior_b
        self.rt_by_day["total"].append((a + num) / (b + denom) if denom > 0 else np.nan)

    def get_rt_series(self) -> np.ndarray:
        """Return R_t series (total), aligned to infections_by_day['total'] day index."""
        return np.asarray(self.rt_by_day["total"], dtype=np.float64)

    def plot_rt(self) -> None:
        """Plot total R_t over time (requires running the simulation first)."""
        rt = self.get_rt_series()
        if rt.size == 0:
            print("No R_t values to plot (run simulation first).")
            return

        days = np.arange(rt.size, dtype=int)
        plt.figure(figsize=(8, 4))
        plt.plot(days, rt, label="R_t (Cori)")
        plt.axhline(1.0, linestyle="--", linewidth=1.0, label="R_t = 1")
        plt.ylim(bottom=0)
        plt.xlabel("Day")
        plt.ylabel("R_t")
        plt.title("Effective reproduction number (total)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_epidemic_curve(self):
        days = [s['day'] for s in self.daily_stats]
        infected = [s['infectious'] for s in self.daily_stats]
        recovered = [s['recovered'] for s in self.daily_stats]
        dead = [s['dead'] for s in self.daily_stats]

        plt.figure(figsize=(10, 6))
        plt.plot(days, infected, label="Infectious", color='red')
        plt.plot(days, recovered, label="Recovered", color='green')
        plt.plot(days, dead, label="Dead", color='black')
        plt.xlabel("Day")
        plt.ylabel("Population count")
        plt.title("Epidemic Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_orientation_incidence(self):
        """Stacked area plot of daily new infections by sexual orientation."""
        days = range(len(self.orientation_daily['hetero']))
        h = self.orientation_daily['hetero']
        b = self.orientation_daily['bi']
        m = self.orientation_daily['homo']

        plt.figure(figsize=(10, 6))
        plt.stackplot(days, h, b, m,
                      labels=['Heterosexual', 'Bisexual', 'Homosexual'])
        plt.xlabel('Day')
        plt.ylabel('New infections')
        plt.title('Daily new infections by sexual orientation')
        plt.legend(loc='upper right')
        plt.grid(True, axis='y', alpha=.3)
        plt.tight_layout()
        plt.show()

    def plot_layerwise_transmissions(self):
        plt.figure(figsize=(12, 6))
        for layer, total in self.layer_infection_counts.items():
            plt.bar(layer, total)
        plt.ylabel("Cumulative Infections")
        plt.title("Total Infections by Layer")
        plt.grid(True, axis='y')
        plt.show()

    def _recompute_ori_code(self) -> None:
        """Recompute integer orientation codes from self.orientations.

        Mapping: hetero -> 0, bi -> 1, homo -> 2.
        This must be called after demographics are loaded/built.
        """
        # Default to 0 (hetero) if orientations missing
        if not hasattr(self, "orientations"):
            self.ori_code[:] = 0
            return
        ori = np.asarray(self.orientations)
        self.ori_code[:] = 0
        self.ori_code[ori == "hetero"] = 0
        self.ori_code[ori == "bi"] = 1
        self.ori_code[ori == "homo"] = 2

        def plot_layerwise_transmissions(self):
            plt.figure(figsize=(12, 6))
            for layer, total in self.layer_infection_counts.items():
                plt.bar(layer, total)
            plt.ylabel("Cumulative Infections")
            plt.title("Total Infections by Layer")
            plt.grid(True, axis='y')
            plt.show()

    def plot_time_series_by_layer(self, observed_episode=Case_data, start_date="2022-05-01",
                                  show_episode_fit=True, show_cumulative=False,
                                  title_prefix="Daily infections by layer"):
        """Plot daily infections by transmission layer, with optional LAC episode-date fit panel.

        Parameters
        ----------
        observed_episode : sequence[int] | None
            Observed daily incident *cases by episode date* (e.g., LAC PowerBI). If provided and
            ``show_episode_fit`` is True, a second panel will compare observed vs simulated
            episode-date cases (and optionally cumulative).
        start_date : str
            Day-0 label.
        show_episode_fit : bool
            Whether to add an episode-fit comparison panel when observed data are provided.
        show_cumulative : bool
            Also plot cumulative episode-fit (uses an additional figure).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # --- Panel 1: infections by layer (model internal infections)
        fig, ax = plt.subplots(1, 2 if (observed_episode is not None and show_episode_fit) else 1,
                               figsize=(16, 6))
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])

        ax0 = ax[0]
        for layer, series in self.infections_by_day.items():
            ax0.plot(series, label=layer)
        ax0.set_xlabel("Day")
        ax0.set_ylabel("New infections")
        ax0.set_title(title_prefix)
        ax0.grid(True)
        ax0.legend()

        # --- Panel 2: episode-date fit (observed vs simulated episode cases)
        if observed_episode is not None and show_episode_fit:
            obs = np.asarray(observed_episode, dtype=float)
            sim_ep = np.asarray(getattr(self, "daily_episode_cases", []), dtype=float)
            if sim_ep.size == 0:
                raise AttributeError("Population has no daily_episode_cases. Did you update to the episode version?")
            T = int(min(obs.size, sim_ep.size))
            obs = obs[:T]
            sim_ep = sim_ep[:T]
            x = np.arange(T)

            ax1 = ax[1]
            ax1.plot(x, obs, label="Observed cases (episode date)")
            ax1.plot(x, sim_ep, label="Simulated cases (episode date)")
            ax1.set_xlabel(f"Day since {start_date}")
            ax1.set_ylabel("Daily cases")
            ax1.set_title("Episode-date fit (daily)")
            ax1.grid(True)
            ax1.legend()

        plt.tight_layout()
        plt.show()

        # Optional cumulative comparison in a separate figure (keeps the main layout clean)
        if observed_episode is not None and show_episode_fit and show_cumulative:
            obs = np.asarray(observed_episode, dtype=float)
            sim_ep = np.asarray(getattr(self, "daily_episode_cases", []), dtype=float)
            T = int(min(obs.size, sim_ep.size))
            obs = obs[:T];
            sim_ep = sim_ep[:T]
            x = np.arange(T)

            plt.figure(figsize=(11, 5))
            plt.plot(x, np.cumsum(obs), label="Observed cumulative (episode date)")
            plt.plot(x, np.cumsum(sim_ep), label="Simulated cumulative (episode date)")
            plt.xlabel(f"Day since {start_date}")
            plt.ylabel("Cumulative cases")
            plt.title("Episode-date fit (cumulative)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def plot_breakdown_by_gender_orientation(self):
        genders = np.array(self.genders)
        orientations = np.array(self.orientations)
        active = ~self.susceptible

        labels = []
        counts = []
        for g in [0, 1]:
            for o in ['hetero', 'bi', 'homo']:
                label = f"{g}-{o}"
                mask = (genders == g) & (orientations == o)
                count = np.sum(active & mask)
                labels.append(label)
                counts.append(count)

        plt.figure(figsize=(12, 6))
        plt.bar(labels, counts)
        plt.xticks(rotation=45)
        plt.ylabel("Cumulative Infections")
        plt.title("Infections by Gender and Sexual Orientation")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

    def plot_sex_party_attendance(self):
        """Stacked area chart of daily party attendees by pool."""
        import matplotlib.pyplot as plt
        days = range(len(self.sex_party_daily['MSM']))
        plt.figure(figsize=(10, 5))
        plt.stackplot(
            days,
            self.sex_party_daily['MSM'],
            self.sex_party_daily['WSW'],
            self.sex_party_daily['MIX'],
            labels=['MSM‑party', 'WSW‑party', 'Mixed party']
        )
        plt.ylabel('Attendees')
        plt.xlabel('Day')
        plt.title('Daily sex‑party attendance by pool')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def visualize_network(self):
        colors = ['red' if self.people[n].infectious else 'green' if self.people[n].recovered else 'skyblue'
                  for n in self.sexual_network.nodes()]
        plt.figure(figsize=(10, 8))
        nx.draw(self.sexual_network, node_color=colors, with_labels=True)
        plt.title("Sexual Network with Infection Status")
        plt.show()

    def plot_episode_vs_observed(self, observed_daily=Case_data, start_date="2022-05-01",
                                 show_daily=True, show_cumulative=True,
                                 compare_diagnosis=True,
                                 title_prefix="Episode-date fit"):
        """Plot simulated episode-date cases against observed (LAC dashboard) series.

        Parameters
        ----------
        observed_daily : sequence[int]
            Observed daily incident cases by *episode date* (LAC definition).
        start_date : str
            ISO date string for day 0 (used only for labels).
        show_daily : bool
            Plot daily incidence comparison.
        show_cumulative : bool
            Plot cumulative incidence comparison.
        compare_diagnosis : bool
            Also plot cumulative diagnoses vs observed cumulative episode cases.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        obs = np.asarray(observed_daily, dtype=float)
        sim_ep = np.asarray(getattr(self, "daily_episode_cases", []), dtype=float)

        if sim_ep.size == 0:
            raise AttributeError(
                "Population has no daily_episode_cases. Did you update population.py to the episode version?")
        T = int(min(obs.size, sim_ep.size))
        obs = obs[:T]
        sim_ep = sim_ep[:T]

        x = np.arange(T)

        if show_daily:
            plt.figure(figsize=(11, 5))
            plt.plot(x, obs, label="Observed cases (episode date)")
            plt.plot(x, sim_ep, label="Simulated cases (episode date)")
            plt.xlabel(f"Day since {start_date}")
            plt.ylabel("Daily cases")
            plt.title(f"{title_prefix}: daily incidence")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        if show_cumulative:
            plt.figure(figsize=(11, 5))
            plt.plot(x, np.cumsum(obs), label="Observed cumulative (episode date)")
            plt.plot(x, np.cumsum(sim_ep), label="Simulated cumulative (episode date)")
            plt.xlabel(f"Day since {start_date}")
            plt.ylabel("Cumulative cases")
            plt.title(f"{title_prefix}: cumulative incidence")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        if compare_diagnosis and hasattr(self, "daily_new_diagnoses"):
            sim_dx = np.asarray(self.daily_new_diagnoses, dtype=float)[:T]
            plt.figure(figsize=(11, 5))
            plt.plot(x, np.cumsum(obs), label="Observed cumulative (episode date)")
            plt.plot(x, np.cumsum(sim_dx), label="Simulated cumulative (diagnosis)")
            plt.xlabel(f"Day since {start_date}")
            plt.ylabel("Cumulative cases")
            plt.title(f"{title_prefix}: observed episode vs simulated diagnosis")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def plot_episode_strata(self, start_date="2022-05-01", by="sex", title_prefix="Episode-date cases"):
        """Plot simulated episode-date cases stratified by sex or sexual orientation."""
        import numpy as np
        import matplotlib.pyplot as plt

        ep = np.asarray(getattr(self, "daily_episode_cases", []), dtype=float)
        if ep.size == 0:
            raise AttributeError("Population has no daily_episode_cases.")
        T = ep.size
        x = np.arange(T)

        if by == "sex":
            arr = np.asarray(getattr(self, "daily_episode_cases_by_sex", []), dtype=float)
            if arr.size == 0:
                raise AttributeError("Population has no daily_episode_cases_by_sex.")
            arr = arr[:T, :]
            labels = ["Male", "Female"]
        elif by in ("ori", "orientation"):
            arr = np.asarray(getattr(self, "daily_episode_cases_by_ori", []), dtype=float)
            if arr.size == 0:
                raise AttributeError("Population has no daily_episode_cases_by_ori.")
            arr = arr[:T, :]
            labels = ["Hetero", "Bi", "Homo"]
        else:
            raise ValueError("by must be 'sex' or 'ori'")

        plt.figure(figsize=(11, 5))
        for j, lab in enumerate(labels):
            plt.plot(x, np.cumsum(arr[:, j]), label=f"{lab} (cumulative)")
        plt.xlabel(f"Day since {start_date}")
        plt.ylabel("Cumulative cases (episode date)")
        plt.title(f"{title_prefix} by {by}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Return a compact end-of-run summary (last daily_stats) plus key composition ratios.

        Ratios are computed from *episode-date* cumulative cases if the corresponding arrays exist.
        """
        out = dict(self.daily_stats[-1]) if self.daily_stats else {}

        # --- episode composition ratios (for calibration diagnostics) ---
        try:
            total = int(np.sum(self.daily_episode_cases)) if hasattr(self, "daily_episode_cases") else 0
        except Exception:
            total = 0

        if total > 0:
            # Sex (0=male, 1=female) among episode-date cases.
            if hasattr(self, "daily_episode_cases_by_sex"):
                sex_tot = np.sum(self.daily_episode_cases_by_sex, axis=0).astype(float)
                out["episode_female_ratio"] = float(sex_tot[1] / total)
                out["episode_male_ratio"] = float(sex_tot[0] / total)
                # Human-readable strings kept for backwards compatibility with older logs.
                out["female_ratio"] = f"{out['episode_female_ratio']:.2%}"
                out["male_ratio"] = f"{out['episode_male_ratio']:.2%}"

            # Sexual orientation (0=hetero,1=bi,2=homo) among episode-date cases.
            if hasattr(self, "daily_episode_cases_by_ori"):
                ori_tot = np.sum(self.daily_episode_cases_by_ori, axis=0).astype(float)
                out["episode_hetero_ratio"] = float(ori_tot[0] / total)
                out["episode_bi_ratio"] = float(ori_tot[1] / total)
                out["episode_homo_ratio"] = float(ori_tot[2] / total)
                out["heterosexual_ratio"] = f"{out['episode_hetero_ratio']:.2%}"
                out["bisexual_ratio"] = f"{out['episode_bi_ratio']:.2%}"
                out["homosexual_ratio"] = f"{out['episode_homo_ratio']:.2%}"

            # Sex-by-orientation case composition using the same denominator as the
            # observed stratified case table.  sexcat mapping:
            #   0=female_hetero, 1=male_hetero, 2=male_homo,
            #   3=female_homo,   4=male_bi,     5=female_bi.
            if hasattr(self, "daily_episode_cases_by_sexcat"):
                sc_tot = np.sum(self.daily_episode_cases_by_sexcat, axis=0).astype(float)
                known = float(np.sum(sc_tot))
                if known > 0:
                    fh, mh, mhomo, fhomo, mbi, fbi = sc_tot
                    out["episode_female_hetero"] = int(fh)
                    out["episode_male_hetero"] = int(mh)
                    out["episode_male_homo"] = int(mhomo)
                    out["episode_female_homo"] = int(fhomo)
                    out["episode_male_bi"] = int(mbi)
                    out["episode_female_bi"] = int(fbi)
                    out["episode_female_known_ratio"] = float((fh + fhomo + fbi) / known)
                    out["episode_homo_male_ratio"] = float(mhomo / known)
                    out["episode_bi_male_ratio"] = float(mbi / known)
                    out["episode_hetero_known_ratio"] = float((fh + mh) / known)
                    out["episode_bi_known_ratio"] = float((mbi + fbi) / known)
                    out["episode_homo_known_ratio"] = float((mhomo + fhomo) / known)
        else:
            # Keep keys present for easier downstream logging
            out.setdefault("female_ratio", "0.0%")
            out.setdefault("male_ratio", "0.0%")
            out.setdefault("heterosexual_ratio", "0.0%")
            out.setdefault("bisexual_ratio", "0.0%")
            out.setdefault("homosexual_ratio", "0.0%")
        # --- cumulative infections by orientation (NOT episode; based on daily new exposures) ---
        # These are the main calibration metrics because they use the same denominator
        # as the observed known-orientation targets:
        #   hetero + bi + homo among cases with known orientation.
        try:
            cum_h = int(np.sum(self.orientation_daily.get('hetero', [])))
            cum_b = int(np.sum(self.orientation_daily.get('bi', [])))
            cum_m = int(np.sum(self.orientation_daily.get('homo', [])))
            cum_tot = cum_h + cum_b + cum_m
            out["cum_inf_hetero"] = cum_h
            out["cum_inf_bi"] = cum_b
            out["cum_inf_homo"] = cum_m
            if cum_tot > 0:
                out["cum_inf_hetero_ratio"] = cum_h / cum_tot
                out["cum_inf_bi_ratio"] = cum_b / cum_tot
                out["cum_inf_homo_ratio"] = cum_m / cum_tot

            # Sex-specific cumulative infection diagnostics.  These are not used by
            # the transmission engine; they are only for checking the observed
            # Male-Homo / Male-Bisexual / Female-Heterosexual columns.
            if hasattr(self, "sex_daily"):
                cum_male = int(np.sum(self.sex_daily.get("male", [])))
                cum_female = int(np.sum(self.sex_daily.get("female", [])))
                out["cum_inf_male"] = cum_male
                out["cum_inf_female"] = cum_female
                if (cum_male + cum_female) > 0:
                    out["cum_inf_female_ratio"] = cum_female / (cum_male + cum_female)

            if hasattr(self, "sexcat_daily"):
                mhomo = int(np.sum(self.sexcat_daily.get("male_homo", [])))
                mbi = int(np.sum(self.sexcat_daily.get("male_bi", [])))
                mhet = int(np.sum(self.sexcat_daily.get("male_hetero", [])))
                fhet = int(np.sum(self.sexcat_daily.get("female_hetero", [])))
                fhomo = int(np.sum(self.sexcat_daily.get("female_homo", [])))
                fbi = int(np.sum(self.sexcat_daily.get("female_bi", [])))
                known = mhomo + mbi + mhet + fhet + fhomo + fbi
                out["cum_inf_male_homo"] = mhomo
                out["cum_inf_male_bi"] = mbi
                out["cum_inf_male_hetero"] = mhet
                out["cum_inf_female_hetero"] = fhet
                out["cum_inf_female_homo"] = fhomo
                out["cum_inf_female_bi"] = fbi
                if known > 0:
                    out["cum_inf_homo_male_ratio"] = mhomo / known
                    out["cum_inf_bi_male_ratio"] = mbi / known
                    out["cum_inf_female_known_ratio"] = (fhet + fhomo + fbi) / known

            # Optional calibration-target differences in percentage points.
            # The target values are passed from main.py via params["calibration_targets"].
            targets = self.params.get("calibration_targets", {}) if hasattr(self, "params") else {}
            if targets and cum_tot > 0:
                def _diff_pp(current, key):
                    return 100.0 * (float(current) - float(targets[key])) if key in targets else None

                # Main calibration diagnostics use episode/case composition when available.
                # Cumulative infection ratios remain available as separate mechanism checks.
                out["calib_diff_pp_hetero"] = _diff_pp(
                    out.get("episode_hetero_known_ratio",
                            out.get("episode_hetero_ratio", out.get("cum_inf_hetero_ratio", 0.0))),
                    "target_hetero"
                )
                out["calib_diff_pp_bi"] = _diff_pp(
                    out.get("episode_bi_known_ratio", out.get("episode_bi_ratio", out.get("cum_inf_bi_ratio", 0.0))),
                    "target_bi"
                )
                out["calib_diff_pp_homo"] = _diff_pp(
                    out.get("episode_homo_known_ratio",
                            out.get("episode_homo_ratio", out.get("cum_inf_homo_ratio", 0.0))),
                    "target_homo"
                )
                if "target_female_known" in targets:
                    cur = out.get("episode_female_known_ratio", out.get("cum_inf_female_known_ratio", None))
                    if cur is not None:
                        out["calib_diff_pp_female_known"] = 100.0 * (float(cur) - float(targets["target_female_known"]))
                if "target_homo_male" in targets:
                    cur = out.get("episode_homo_male_ratio", out.get("cum_inf_homo_male_ratio", None))
                    if cur is not None:
                        out["calib_diff_pp_homo_male"] = 100.0 * (float(cur) - float(targets["target_homo_male"]))
                if "target_bi_male" in targets:
                    cur = out.get("episode_bi_male_ratio", out.get("cum_inf_bi_male_ratio", None))
                    if cur is not None:
                        out["calib_diff_pp_bi_male"] = 100.0 * (float(cur) - float(targets["target_bi_male"]))
        except Exception:
            pass
        # Now you can check the probability_log to see the recorded probabilities
        # --- success-only TX probability diagnostics ---
        n = int(self.txlog_i)
        if n > 0:
            layer = self.txlog_layer[:n]
            p = self.txlog_p[:n].astype(float)

            # counts by layer
            uniq = np.unique(layer)
            for lc in uniq:
                m = (layer == lc)
                pp = p[m]
                print(f"TXLOG layer={int(lc)}: n={int(m.sum())}, "
                      f"mean={pp.mean():.4g}, p50={np.median(pp):.4g}, "
                      f"p90={np.quantile(pp, 0.9):.4g}, p99={np.quantile(pp, 0.99):.4g}, max={pp.max():.4g}")
        # --- add after existing TXLOG layer stats in summary() ---
        n = int(self.txlog_i)
        if n > 0:
            layer = self.txlog_layer[:n]
            src = self.txlog_src[:n]
            dst = self.txlog_dst[:n]
            p = self.txlog_p[:n].astype(float)

            # 你的默认 mapping: {"stable":1, "oneoff":2, "sex_party":3, "household_close":4}
            sex_party_code = 3
            m = (layer == sex_party_code)
            if np.any(m):
                s = src[m];
                d = dst[m];
                pp = p[m]
                gs = self.genders[s];
                gd = self.genders[d]

                mm = (gs == 0) & (gd == 0)
                ff = (gs == 1) & (gd == 1)
                mf = (gs != gd)  # MIX 的 proxy（至少对“成功传播”来说非常直观）

                def _stat(name, mask):
                    k = int(mask.sum())
                    if k == 0:
                        print(f"TXLOG sex_party {name}: n=0")
                        return
                    x = pp[mask]
                    print(f"TXLOG sex_party {name}: n={k}, mean={x.mean():.4g}, p50={np.median(x):.4g}, "
                          f"p90={np.quantile(x, 0.9):.4g}, p99={np.quantile(x, 0.99):.4g}, max={x.max():.4g}")

                _stat("MM", mm)
                _stat("FF", ff)
                _stat("MF", mf)

        return out
