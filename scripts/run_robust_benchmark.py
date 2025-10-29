import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_digits
from stable_cart import RobustPrefixHonestClassifier  # from your package
# optionally: from stable_cart import LessGreedyHybridRegressor, BootstrapVariancePenalizedRegressor

def strat_boot(X, y, rs):
    idx = []
    for c in np.unique(y):
        ci = np.where(y==c)[0]
        b = rs.choice(ci, size=len(ci), replace=True)
        idx.append(b)
    idx = np.concatenate(idx); rs.shuffle(idx)
    return X[idx], y[idx]

def winsorize(X, q=(0.01,0.99)):
    lo = np.quantile(X, q[0], axis=0); hi = np.quantile(X, q[1], axis=0)
    return np.minimum(np.maximum(X, lo), hi), lo, hi

def run_one(name, X, y, B=20, rs=np.random.RandomState(42)):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    Xtr, lo, hi = winsorize(Xtr); Xte = np.minimum(np.maximum(Xte, lo), hi)
    min_leaf = max(2, int(np.sqrt(len(Xtr))/2))
    out = []
    def eval_model(get_m, label, extra):
        P=np.zeros((B,len(Xte))); A=np.zeros(B); C=np.zeros(B)
        for b in range(B):
            Xb, yb = strat_boot(Xtr, ytr, rs)
            m = get_m(rs.randint(0,10**6)); m.fit(Xb, yb)
            pb = m.predict_proba(Xte)[:,1]
            P[b]=pb; A[b]=roc_auc_score(yte,pb); C[b]=accuracy_score(yte,(pb>=0.5).astype(int))
        out.append(dict(dataset=name, model=label,
                        oos_prob_variance=float(P.var(axis=0,ddof=1).mean()),
                        auc=float(A.mean()), acc=float(C.mean()), **extra))
    # Baseline CART
    eval_model(lambda seed: DecisionTreeClassifier(max_depth=6, min_samples_leaf=min_leaf, random_state=seed),
               "CART", {})
    # Robust
    for L in [1,2]:
        for m in [0.5,1.0,2.0]:
            for ef in [0.3,0.4,0.5]:
                eval_model(lambda seed, L=L,m=m,ef=ef: RobustPrefixHonestClassifier(
                    top_levels=L, max_depth=6, min_samples_leaf=min_leaf,
                    val_frac=0.2, est_frac=ef, m_smooth=m,
                    consensus_B=12, consensus_subsample_frac=0.8, consensus_max_bins=24,
                    random_state=seed),
                    "RobustPrefixHonestClassifier", dict(top_levels=L, m_smooth=m, est_frac=ef))
    return pd.DataFrame(out)

def main():
    rs = np.random.RandomState(2025)
    bc = load_breast_cancer(); X_bc,y_bc=bc.data,bc.target
    dg = load_digits(); X_dg,y_dg=dg.data,(dg.target==0).astype(int)
    frames = []
    for name, X, y in [("breast_cancer",X_bc,y_bc), ("digits_0vsrest",X_dg,y_dg)]:
        frames.append(run_one(name, X, y, B=20, rs=rs))
    df = pd.concat(frames, ignore_index=True)
    # annotate relative variance reduction vs CART per dataset
    def rel(g):
        base = float(g.loc[g.model=="CART","oos_prob_variance"].mean())
        g = g.copy()
        g["rel_var_reduction"] = 1.0 - g["oos_prob_variance"] / base if base>0 else np.nan
        return g
    df = df.groupby("dataset", group_keys=False).apply(rel)
    df.to_csv("robust_benchmark_results.csv", index=False)
    print(df.head(12))

if __name__ == "__main__":
    main()