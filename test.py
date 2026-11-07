
import numba as nb
import numpy as np

@nb.njit(parallel=True, nogil=True, fastmath=True, cache=True)  # 性能期：并行 + 无边界检查
def build_oneoff_edges_fast_needs(active: np.ndarray,     # int64 今日活跃 uid
                                  ages: np.ndarray,       # int16
                                  genders: np.ndarray,    # int8/0,1
                                  sexcat: np.ndarray,     # uint8(0..5)
                                  sex_role: np.ndarray,   # int8
                                  compat: np.ndarray,     # (6,6) bool
                                  need: np.ndarray,       # int32/64，预先生成好的每个 uid 的需求量
                                  p_anal_rec: float,
                                  p_anal_ins: float,
                                  p_vag: float,
                                  p_oral: float,
                                  p_skin: float,
                                  age_gap: int = 10,
                                  keep_scale: float = 1.0 # 0~1
                                  ):
    """
    基于预先提供的 need_i，减少并行随机数压力。
    双遍：计数(已由 need 提供) -> 前缀和 -> 实际填充(并记录 filled_i) -> 可选压缩。
    索引一律 int64；概率 float32。
    """
    m = active.size
    counts_need = need.astype(np.int64)

    # -------- prefix-sum ----------
    offsets = np.empty(m + 1, dtype=np.int64)
    total = 0
    for i in range(m):
        offsets[i] = total
        total += counts_need[i]
    offsets[m] = total

    if total == 0:
        return (np.empty(0, np.int64),
                np.empty(0, np.int64),
                np.empty(0, np.float32),
                np.empty(0, np.float32))

    src = np.empty(total, dtype=np.int64)
    dst = np.empty(total, dtype=np.int64)
    p_uv = np.empty(total, dtype=np.float32)
    p_vu = np.empty(total, dtype=np.float32)

    counts_filled = np.zeros(m, dtype=np.int64)

    # -------- fill (并行) ----------
    for i in nb.prange(m):
        uid = active[i]
        need_i = counts_need[i]
        if need_i <= 0:
            continue

        base = offsets[i]
        filled = 0
        tries = 0
        # 拒绝采样上限给足一点（在小 need 情况也不会太大）
        max_tries = need_i * 12 + 24

        # 小缓存避免 (uid, v) 重复；need_i 一般很小，这样足够
        chosen = np.empty(need_i, dtype=np.int64)
        chosen_size = 0

        age_u = int(ages[uid])
        sc_u  = int(sexcat[uid])

        while (filled < need_i) and (tries < max_tries):
            tries += 1
            j = np.random.randint(0, m)
            v = active[j]
            if v == uid:
                continue

            # 年龄差
            if abs(age_u - int(ages[v])) > age_gap:
                continue
            # 兼容
            if not compat[sc_u, int(sexcat[v])]:
                continue
            # 行为缩放
            if keep_scale < 1.0 and (np.random.random() >= keep_scale):
                continue
            # 去重
            dup = False
            for t in range(chosen_size):
                if chosen[t] == v:
                    dup = True
                    break
            if dup:
                continue

            idx = base + filled
            src[idx] = uid
            dst[idx] = v

            gu = genders[uid]; gv = genders[v]
            if (gu == 0) and (gv == 0):  # MSM
                r1 = sex_role[uid]; r2 = sex_role[v]
                if r1 == 1 and r2 == 0:
                    p_uv[idx] = p_anal_rec; p_vu[idx] = p_anal_ins
                elif r1 == 0 and r2 == 1:
                    p_uv[idx] = p_anal_ins; p_vu[idx] = p_anal_rec
