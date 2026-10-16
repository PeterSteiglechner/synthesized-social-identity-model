#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9})
import string
from matplotlib.lines import Line2D
import os
os.makedirs("data", exist_ok=True)
os.makedirs("figs", exist_ok=True)

#%%
# params
n = 100
nA = 90
nB = n-nA
opA_hi = 0.9
opB_lo = opA_hi
e_init_min = 0
e_init_max = 1

crit_difference_max=0.2
alpha = 0.2
gamma = 0.5
delta_charge = 0.05
delta_drain = 0.05
delta_charge_identity = 0.5
beta = 0.01

T = 3000

#%%
def snap(seed, t, groups, opinions, identityS, selfEsteem, threat):
    return pd.DataFrame({"seed":seed, "t":t, "id":range(n), "group":groups, "op":opinions, "identity_standard":identityS, "self_esteem":selfEsteem, "threat":threat})


#%%
seeds = np.arange(100)
scenarios = [
    (True, True, True),
    (False, True, True),
    (True, False, True),
    (True, True, False),
]
for useSIA, useIT, useIPT in scenarios:
    res = []
    for seed in seeds: 
        np.random.seed(seed)
        groups = np.array(["A"] * nA + ["B"] * nB)
        opinions =  np.concatenate([np.random.uniform(0,opA_hi, nA), np.random.uniform(opB_lo,1,nB)])
        identityS =  np.concatenate([[opinions[groups=="A"].mean()] * nA, [opinions[groups=="B"].mean()] * nB]) if useSIA else opinions.copy()
        selfesteem = np.random.uniform(e_init_min, e_init_max, size=n)
            
        res.append(snap(seed, 0, groups, opinions, identityS, selfesteem, np.nan))
        agentIds= list(range(n))
        for t in range(1,T+1):
            # agentIds.shuffle()
            observed_ops = opinions[np.random.permutation(n)]          # or choice, excluding self
            threatening = (abs(observed_ops - identityS) > (crit_difference_max * selfesteem))
            # ALTERNATIVE: threat = False if self-esteem high and diff=0,  
            # threatening = np.random.random(n) <= (1-selfesteem) * abs(observed_ops - identityS)
            nonthreatening = ~threatening
            
            # fast loop
            opinions[nonthreatening] += alpha * (observed_ops[nonthreatening] - opinions[nonthreatening]) 
            selfesteem[nonthreatening] += delta_charge 
            if useIPT:
                opinions[threatening] +=  gamma * (identityS[threatening] - opinions[threatening]) 
            selfesteem[threatening] -= delta_drain 
            selfesteem = np.clip(selfesteem, a_min=0, a_max=1)
            opinions = np.clip(opinions, a_min=0, a_max=1)
            
            # slow loop
            if useIT: 
                # CHANGE: slow loop with prob: 1-self-esteem
                # identityChange = np.random.random(n) < (1-selfesteem)
                identityChange = selfesteem <= 1e-4
                identityS[identityChange] += beta * (observed_ops[identityChange] - identityS[identityChange]) 
                selfesteem[identityChange] += delta_charge_identity
                identityS = np.clip(identityS, a_min=0, a_max=1)
                selfesteem = np.clip(selfesteem, a_min=0, a_max=1)
                    
            if t%10 == 0 :
                res.append(snap(seed, t, groups, opinions, identityS, selfesteem, threatening))

    # ---------------------
    # -----   Store
    # ---------------------
    condTitle = "Baseline Simualtion (all theories)" if (useSIA*useIPT*useIT)==1 else ("Baseline without SIA" if not useSIA else ("Baseline without IPT" if not useIPT else ("Baseline without ICT" if not useIT else ("?"))))
    cond = "" if (useSIA*useIPT*useIT)==1 else ("_noSIA" if not useSIA else ("_noIPT" if not useIPT else ("_noIT" if not useIT else ("?"))))
    print("finished: ",condTitle, "; stored as ",f"data/baseline{cond}.csv")
    df = pd.concat(res)
    df.to_csv(f"data/baseline{cond}.csv")


# %%
# ---------------------
# -----   Visualise
# ---------------------
for useSIA, useIT, useIPT in scenarios:

    cond = "" if (useSIA*useIPT*useIT)==1 else ("_noSIA" if not useSIA else ("_noIPT" if not useIPT else ("_noIT" if not useIT else ("?"))))
    condTitle = "Baseline Simualtion (all theories)" if (useSIA*useIPT*useIT)==1 else ("Baseline without SIA" if not useSIA else ("Baseline without IPT" if not useIPT else ("Baseline without ICT" if not useIT else ("?"))))

    df = pd.read_csv(f"data/baseline{cond}.csv")

    cmap = {"A": "tomato", "B": "cornflowerblue"}
    cols = ["op", "identity_standard", "self_esteem"]
    labels = ["Opinion", "Social Identity Content", "Self Esteem"]
    resolution = 1

    # ---------------------
    # -----   Ensemble Mean
    # ---------------------
    fig, axs = plt.subplots(1, 3, sharex=True, figsize=(16/2.54, 7/2.54))
    # filter once, then collapse to one row per (seed, group, t)
    d = df[(df.t % resolution == 0)]
    d = d.groupby(["seed", "group", "t"], as_index=False)[cols].mean().sort_values("t")

    # individual seeds: faint
    for (s, g), sub in d.groupby(["seed", "group"]):
        for ax, col in zip(axs, cols):
            ax.plot(sub["t"], sub[col], color=cmap[g], alpha=0.1, lw=1)

    # mean across seeds: solid, drawn on top
    m = d.groupby(["group", "t"], as_index=False)[cols].mean()
    for g, sub in m.groupby("group"):
        for ax, col in zip(axs, cols):
            ax.plot(sub["t"], sub[col], color=cmap[g], lw=2, zorder=3, ls="--")

    for ax, lab in zip(axs, labels):
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1)
        # ax.set_xscale("log")

    axs[0].legend(handles=[Line2D([], [], color=c, lw=2, label=g) for g, c in cmap.items()])
    fig.suptitle(condTitle, x=0.5, ha="center")
    for ax, l in zip(axs, string.ascii_uppercase):
        ax.text(0.025,0.975,l, transform=ax.transAxes, fontsize=12, fontdict={"weight":"bold"}, va="top", ha="left")
    fig.tight_layout()
    fig.savefig(f"figs/baseline{cond}.png", dpi=600)

    # ---------------------
    # -----   Individual Run
    # ---------------------

    s = 48

    fig, axs = plt.subplots(1, 3, sharex=True, figsize=(16/2.54, 7/2.54))

    d = df[(df.seed==s) & (df.t % resolution == 0)]
    d = d.groupby(["id", "group", "t"], as_index=False)[cols].mean().sort_values("t")

    # individuals
    for (i,g), sub in d.groupby(["id", "group"]):
        for ax, col in zip(axs, cols):
            ax.plot(sub["t"], sub[col], color=cmap[g], alpha=0.2, lw=1)

    # mean : solid, drawn on top
    m = d.groupby(["group", "t"], as_index=False)[cols].mean()
    for g, sub in m.groupby("group"):
        for ax, col in zip(axs, cols):
            ax.plot(sub["t"], sub[col], color=cmap[g], lw=2, zorder=3, ls="--")

    for ax, lab in zip(axs, labels):
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1)
        # ax.set_xscale("log")

    axs[0].legend(handles=[Line2D([], [], color=c, lw=2, label=g) for g, c in cmap.items()])
    fig.suptitle(f"Example Simulation: {condTitle}", x=0.5, ha="center")
    for ax, l in zip(axs, string.ascii_uppercase):
        ax.text(0.025,0.975,l, transform=ax.transAxes, fontsize=12, fontdict={"weight":"bold"}, va="top", ha="left")

    fig.tight_layout()
    fig.savefig(f"figs/baseline{cond}_exampleSeed{s}.png", dpi=600)
# %%


# %%
