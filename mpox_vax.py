# mpox_vax.py  -----------------------------------------------------------
import numpy as np

import matplotlib
matplotlib.use("QtAgg")  # or "QtAgg" for Matplotlib >=3.8
import matplotlib.pyplot as plt

from datetime import timedelta, date

# ----------------------------------------------------------------------
# 1. DAILY DOSE SCHEDULE  (length = 224 d ≈ 22-May-2022 → 31-Dec-2022)
# ----------------------------------------------------------------------
weekly = np.array([
    # first dose   second dose   (epi-wk 20 starts 22-May-2022)
    [49,     0], [82,     0], [145,     0], [168,     0], [1397,   36],
    [2170,  36], [5210,   67], [15547, 158], [35820, 520], [54716, 797],
    [81382,1380], [103974,4634], [83987,12363], [72187,25142],
    [56149,36867], [36828,49091], [35773,56791], [26961,60935],
    [21493,44481], [16616,30958], [13326,24409], [11638,19290],
    [9198,15092],  [7569,11370],  [6360, 9813],  [5683, 9181],
    [2858, 4138],  [4332, 6917],  [3282, 4870],  [2458, 3462],
    [4,       7],  [0,       0]                           # 24-31 Dec
])
daily = np.repeat(weekly / 7, 7, axis=0).astype(int)
days_total = len(daily)

# sex split — rows = male, female (first-dose / second-dose)
sex_split = np.array([[0.915, 0.085],
                      [0.940, 0.060]])

# ----------------------------------------------------------------------
def _init_vax_attrs(self):
    if getattr(self, "_vax_inited", False):
        return
    self.vax_phase  = np.zeros(self.n, dtype=int)   # 0=none,1=post-D1,2=post-D2
    self.vax_day    = np.full(self.n, -1, dtype=int)
    self.r_sus_base = self.r_sus.copy()             # ← keep original age profile
    self._vax_inited = True

def _daily_vaccination(self):
    """Give today’s first & second doses, following the schedule above."""
    if self.day >= days_total:
        return

    first_today, second_today = daily[self.day]
    male   = np.where(self.genders == "male")[0]
    female = np.where(self.genders == "female")[0]

    # -------- first doses -------------------------------------------------
    if first_today:
        m_quota = int(first_today * sex_split[0, 0])
        f_quota = first_today - m_quota
        elig_m  = male[self.mcp_flags[male] & (self.vax_phase[male] == 0)]
        elig_f  = female[self.mcp_flags[female] & (self.vax_phase[female] == 0)]
        give_m  = np.random.choice(elig_m, min(m_quota, len(elig_m)), replace=False)
        give_f  = np.random.choice(elig_f, min(f_quota, len(elig_f)), replace=False)
        give    = np.concatenate((give_m, give_f))
        self.vax_phase[give] = 1
        self.vax_day[give]   = self.day

    # -------- second doses (≥28 d after D1) -------------------------------
    if second_today:
        eligible = np.where((self.vax_phase == 1) &
                            (self.day - self.vax_day >= 28))[0]
        m_quota = int(second_today * sex_split[1, 0])
        f_quota = second_today - m_quota
        elig_m  = eligible[self.genders[eligible] == "male"]
        elig_f  = eligible[self.genders[eligible] == "female"]
        give_m  = np.random.choice(elig_m, min(m_quota, len(elig_m)), replace=False)
        give_f  = np.random.choice(elig_f, min(f_quota, len(elig_f)), replace=False)
        give    = np.concatenate((give_m, give_f))
        self.vax_phase[give] = 2
        self.vax_day[give]   = self.day

def _update_vax_protection(self):
    mask = (self.vax_phase > 0) & (self.day - self.vax_day >= 14)
    self.r_sus[:] = self.r_sus_base          # restore baseline first

    d1 = mask & (self.vax_phase == 1)
    d2 = mask & (self.vax_phase == 2)
    self.r_sus[d1] *= (1 - 0.37)             # 37 % eff. after 1st
    self.r_sus[d2] *= (1 - 0.69)             # 69 % eff. after 2nd

# ----------------------------------------------------------------------
def _wrap_step(Pop):
    """Monkey-patch Pop.step exactly once."""
    if getattr(Pop, "_vax_patched", False):
        return
    orig_step = Pop.step
    def step_with_vax(self, *a, **kw):
        _init_vax_attrs(self)
        _daily_vaccination(self)
        _update_vax_protection(self)
        return orig_step(self, *a, **kw)
    Pop.step = step_with_vax
    Pop._vax_patched = True

# public helper ---------------------------------------------------------
def patch(pop):
    """
    Enable vaccination logic for *this* Population instance.
    Safe to call multiple times.
    """
    _wrap_step(pop.__class__)
    _init_vax_attrs(pop)

# ----------------------------------------------------------------------
def plot_vax(pop):
    """Quick-look cumulative dose curves."""
    days = np.arange(pop.day)
    cum1 = np.cumsum(daily[:pop.day, 0])
    cum2 = np.cumsum(daily[:pop.day, 1])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(days, cum1, label="cumulative dose 1")
    ax.plot(days, cum2, label="cumulative dose 2")
    ax.set_xlabel("simulation day")
    ax.set_ylabel("people")
    ax.set_title("Vaccination roll-out")
    ax.legend()
    plt.tight_layout()
    #plt.show()
__all__ = ["patch", "plot_vax"]
# ----------------------------------------------------------------------
