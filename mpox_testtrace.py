# mpox_testtrace.py  – array-based test & trace patch
# ----------------------------------------------------
import numpy as np
import types
import matplotlib
matplotlib.use("QtAgg")  # or "QtAgg" for Matplotlib >=3.8
import matplotlib.pyplot as plt

# -------- tunables ----------------------------------
TEST_PROB_SYMP      = 0.25   # daily P(test) once symptomatic
LAB_TURNAROUND_DAYS = 2
ISOLATION_DAYS      = 21
QUARANTINE_DAYS     = 7
TRACE_PROB_EDGE     = 0.20
# ----------------------------------------------------


def patch(pop):
    """
    Monkey-patch one Population instance with testing + contact tracing.
    Safe to call multiple times (only patches once).
    """
    if getattr(pop, '_tt_patched', False):
        return

    # ── add extra state arrays ───────────────────────────────────────
    n = pop.n
    pop.test_pending_day = np.full(n, -1, dtype=np.int32)
    pop.iso_end_day      = np.full(n, -1, dtype=np.int32)
    pop.quar_end_day     = np.full(n, -1, dtype=np.int32)

    pop.isolated    = np.zeros(n, dtype=bool)
    pop.quarantined = np.zeros(n, dtype=bool)

    pop._tt_daily = []
    pop._tt_patched = True

    # ── helper: run one day of T&T logic ─────────────────────────────
    def _run_tt(self):
        d = self.day

        # 1) release finished iso / quar
        self.isolated[  (d >= self.iso_end_day)  & self.isolated]    = False
        self.quarantined[(d >= self.quar_end_day) & self.quarantined] = False

        # 2) symptomatic people decide to test
        need_test = (self.symptomatic &
                     (self.test_pending_day < 0) &
                     (~self.isolated))
        take_test = need_test & (np.random.rand(self.n) < TEST_PROB_SYMP)
        self.test_pending_day[take_test] = d + LAB_TURNAROUND_DAYS

        # 3) lab results today
        got_result = (self.test_pending_day == d)
        positives  = got_result                    # PCR ~100 % sens/spec
        self.test_pending_day[got_result] = -1

        # isolate positives
        if positives.any():
            pos_idx = np.where(positives)[0]
            self.isolated[pos_idx] = True
            self.iso_end_day[pos_idx] = d + ISOLATION_DAYS

            # 4) trace contacts on *today’s* edge lists
            #
            # self._today_edges() returns a list of (src, dst) tuples that
            # step() already used for transmission
            #
            for src, dst in self._today_edges():
                # edges are undirected, stored once
                pos_mask = positives[src] | positives[dst]
                if not pos_mask.any():
                    continue

                # neighbours of positives on this layer
                traced = np.where(pos_mask)[0]
                contacts = np.concatenate([dst[traced], src[traced]])

                # each contact quarantined with probability TRACE_PROB_EDGE
                if contacts.size:
                    keep = (np.random.rand(contacts.size) < TRACE_PROB_EDGE)
                    contacts = contacts[keep]
                    if contacts.size:
                        self.quarantined[contacts] = True
                        self.quar_end_day[contacts] = d + QUARANTINE_DAYS

        # 5) daily bookkeeping
        self._tt_daily.append(dict(
            day=d,
            tests=int(take_test.sum()),
            pos=int(positives.sum()),
            iso=int(self.isolated.sum()),
            quar=int(self.quarantined.sum())
        ))

    # ── wrap original step() ─────────────────────────────────────────
    orig_step = pop.step

    def step_with_tt(self, *a, **kw):
        _run_tt(self)       # testing & tracing first
        orig_step(*a, **kw) # normal disease dynamics next

    pop.step = types.MethodType(step_with_tt, pop)


# -------------------------------------------------------------------
#  simple plot helper
# -------------------------------------------------------------------
def plot_tests(pop):
    if not getattr(pop, '_tt_patched', False):
        raise RuntimeError('Population instance not patched by mpox_testtrace.')
    data = pop._tt_daily
    if not data:
        print('No test-trace data recorded yet.')
        return
    days  = [d['day']   for d in data]
    tests = [d['tests'] for d in data]
    pos   = [d['pos']   for d in data]

    plt.figure(figsize=(8,4))
    plt.plot(days, tests, label='tests / day')
    plt.plot(days, pos,   label='positive results')
    plt.xlabel('Day'); plt.ylabel('count')
    plt.title('Testing volume & positive results')
    plt.legend();
    plt.tight_layout()
