"""
main.py - Kaggriculture agent: thin wiring layer over the modular port.

Production agent modules:
    src/constants.py - game constants + board layout
    src/economy.py   - MarketEngine (all market orders) + tunables
    src/config.py    - all tunables (single source of truth)
    src/planner.py   - FarmPlanner (job queue + per-unit dispatch)

Equivalence with the proven baseline behavior is enforced at runtime by the
parity suites (frozen oracle):
    market, strategy, navigation, and the full-pipeline planner replay.
Any drift of the modules vs the oracle fails loudly.

Kaggle packaging: a single-file submission needs this module's logic
bundled with its src/ package (see docs/AGENTS.md).
"""

from src.economy import MarketEngine
from src.planner import FarmPlanner
from src.decision_engine import AdaptiveAgent as _AdaptiveAgent


class Agent:
    """Parses observations and delegates to the modular engines."""

    def __init__(self, plan=None):
        self._plan = plan
        self.reset()

    def reset(self):
        self.day = 0
        self.hour = 0
        self.market = MarketEngine(plan=self._plan)
        self.planner = FarmPlanner(plan=self._plan)

    # Live views kept for introspection / regression tests.
    @property
    def ordered_today(self):
        return self.market.ordered_today

    @property
    def sold_today(self):
        return self.market.sold_today

    @property
    def animals_ordered(self):
        return self.market.animals_ordered

    @property
    def unit_task(self):
        return self.planner.unit_task

    def __call__(self, obs):
        if obs.get("step", 0) == 0:
            self.reset()
        day = obs.get("day", 0)
        self.hour = obs.get("hour", 0)
        player = obs["player"]
        me = obs["farms"][player]
        private = obs["private"]
        market = obs["market"]
        money = me["money"]
        prices = market.get("prices", {})
        shed = private["shed"]
        seeds = private["seeds"]
        tiles = me["tiles"]
        units = [list(me["farmer"])] + [list(h) for h in me["hands"]]
        invs = private["inventories"]

        orders = self.market.build_orders(day, self.hour, me, shed, seeds, prices, money, invs,
                                          unlocked_shops=list((obs.get("town") or {}).get("unlocked_shops") or []))
        market_ctx = {
            "day": day,
            "hour": self.hour,
            "prices": prices,
            "inventory": market.get("inventory", {}),
            "shops": list((obs.get("town") or {}).get("unlocked_shops") or []),
            "milk_crash": getattr(self.market, "milk_crash_active", False),
        }
        self.planner.on_day(day)
        jobs = self.planner.collect_jobs(day, tiles, shed, seeds, invs,
                                         self.market.animals_ordered,
                                         market_ctx=market_ctx)
        actions = []
        used_jobs = set()
        for u_idx, pos in enumerate(units):
            inv = invs[u_idx] if u_idx < len(invs) else {}
            actions.append(self.planner.unit_action(pos, u_idx, jobs, used_jobs, inv, shed))
        return {"farmer": actions[0], "hands": actions[1:], "market": orders}


# Ensemble: bull (6/8) vs crash (3/10) — oracle +5,672 on 8 seeds
BULL_PLAN = {"COW": 6, "SHEEP": 8, "GOOSE": 0, "MELON": 4, "STRAWBERRY": 22, "WHEAT": 18, "TOMATO": 10}
CRASH_PLAN = {"COW": 3, "SHEEP": 10, "GOOSE": 0, "MELON": 4, "STRAWBERRY": 22, "WHEAT": 18, "TOMATO": 10}

class EnsembleAgent:
    def __init__(self):
        self.bull = Agent(plan=BULL_PLAN)
        self.crash = Agent(plan=CRASH_PLAN)
        self.chosen = None
        self._adaptive_bull = _AdaptiveAgent(inner_factory=lambda: Agent(plan=BULL_PLAN), mode="veto")
        self._adaptive_crash = _AdaptiveAgent(inner_factory=lambda: Agent(plan=CRASH_PLAN), mode="veto")
        self.active = None
        self.milk_hist = []
    def __call__(self, obs):
        day = obs.get("day", 0)
        if obs.get("step",0)==0:
            self.chosen=None
            self.active=None
            self.milk_hist=[]
            self.bull.reset()
            self.crash.reset()
        prices = obs.get("market",{}).get("prices",{})
        milk = prices.get("MILK",160)
        # track daily price (once per day)
        if not self.milk_hist or (day > len(self.milk_hist)-1):
            self.milk_hist.append(float(milk))
            if len(self.milk_hist) > 4:
                self.milk_hist.pop(0)
        if self.chosen is None and day >= 3:
            wool = prices.get("WOOL",200)
            inv = obs.get("market",{}).get("inventory",{})
            slope_crash = False
            if len(self.milk_hist) >= 3:
                d1 = self.milk_hist[-1]-self.milk_hist[-2]
                d2 = self.milk_hist[-2]-self.milk_hist[-3]
                # accelerating decline while still high = early crash warning
                if milk < 170 and d1 < 0 and d1 < d2:
                    slope_crash = True
            crash = (milk < 150) or (wool < 180) or (inv.get("MILK",10000) > 10300) or slope_crash
            self.chosen = "crash" if crash else "bull"
            self.active = self._adaptive_crash if self.chosen=="crash" else self._adaptive_bull
        if self.active is None:
            # before day3 use bull as default
            return self._adaptive_bull(obs)
        return self.active(obs)

_raw_agent = Agent()
# Veto-mode wrapper: uses MarketModel projected_price to hold over-supplied goods
# 32 seeds: raw 111,155 -> veto 112,023 (+868) trained on 500 ladder replays
_adaptive = _AdaptiveAgent(inner_factory=Agent, mode="veto")
agent = EnsembleAgent()  # production Top3 ensemble
agent_raw = _raw_agent
agent_veto = _adaptive
agent_ensemble = agent


def agent_function(obs):
    return agent(obs)

def agent_veto_function(obs):
    return _adaptive(obs)