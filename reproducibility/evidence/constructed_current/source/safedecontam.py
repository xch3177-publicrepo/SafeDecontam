#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeDecontam 管线的忠实复现(按论文 §IV–V),用于真实数据研究。批量向量化实现。
论文未完全指定的自由度(论文新章节中如实披露):
  - BM25 归一化 = BM25(b,d) / BM25(b,b)(每查询自归一,值域[0,1])
  - 真实文本约束词典(见 REL_LEXICON/UNIT_WORDS)与结论语境规则(CONCL_CUES)
"""
import math, re
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize as l2norm

TOKEN_RE = re.compile(r"[a-z0-9$%.,'/-]+")
NUM_RE = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?|\d+/\d+)(?![\w])")
SIM_COLS = ["ngram4", "shingle3", "bm25n", "wcos", "ccos", "lsa"]
CON_COLS = ["a_num", "a_rel", "a_tgt", "a_ent", "a_ans"]
CHRONO = ["predates"]

def tokens(t): return TOKEN_RE.findall(t.lower())
def word_ngrams(toks, n): return set(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))

# ---------------- 相似度层(训练集拟合;批量计算) ----------------
class SimilarityLayer:
    k1, b = 1.5, 0.75
    def fit(self, train_texts):
        self.tfidf_w = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit(train_texts)
        self.tfidf_c = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True).fit(train_texts)
        Xw = self.tfidf_w.transform(train_texts)
        self.svd = TruncatedSVD(n_components=min(64, Xw.shape[1]-1, Xw.shape[0]-1), random_state=0).fit(Xw)
        docs = [tokens(t) for t in train_texts]
        self.avgdl = float(np.mean([len(d) for d in docs])) if docs else 1.0
        self.N = len(docs); df = Counter()
        for d in docs: df.update(set(d))
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.default_idf = math.log(1 + (self.N + 0.5) / 0.5)
        return self
    def _bm25(self, q_toks, d_toks):
        tf = Counter(d_toks); dl = len(d_toks) or 1; s = 0.0
        for w in set(q_toks):
            f = tf.get(w, 0)
            if not f: continue
            s += self.idf.get(w, self.default_idf) * f * (self.k1+1) / (f + self.k1*(1 - self.b + self.b*dl/self.avgdl))
        return s
    def batch(self, b_texts, d_texts):
        n = len(b_texts); out = {}
        Bw, Dw = self.tfidf_w.transform(b_texts), self.tfidf_w.transform(d_texts)
        Bc, Dc = self.tfidf_c.transform(b_texts), self.tfidf_c.transform(d_texts)
        out["wcos"] = np.asarray(l2norm(Bw).multiply(l2norm(Dw)).sum(1)).ravel()
        out["ccos"] = np.asarray(l2norm(Bc).multiply(l2norm(Dc)).sum(1)).ravel()
        Bl, Dl = self.svd.transform(Bw), self.svd.transform(Dw)
        def _n(M):
            nn = np.linalg.norm(M, axis=1, keepdims=True); nn[nn == 0] = 1; return M / nn
        out["lsa"] = (_n(Bl) * _n(Dl)).sum(1)
        ng4, sh3, bmn = np.zeros(n), np.zeros(n), np.zeros(n)
        for i, (b, d) in enumerate(zip(b_texts, d_texts)):
            tb, td = tokens(b), tokens(d)
            g4 = word_ngrams(tb, 4)
            ng4[i] = len(g4 & word_ngrams(td, 4)) / len(g4) if g4 else 0.0
            s3b, s3d = word_ngrams(tb, 3), word_ngrams(td, 3)
            sh3[i] = len(s3b & s3d) / len(s3b | s3d) if (s3b | s3d) else 0.0
            ss = self._bm25(tb, tb)
            bmn[i] = min(1.0, self._bm25(tb, td) / ss) if ss > 0 else 0.0
        out.update(ngram4=ng4, shingle3=sh3, bm25n=bmn)
        return pd.DataFrame(out)

# ---------------- 观察约束适配器(真实文本) ----------------
REL_LEXICON = {"more","less","than","times","twice","half","each","per","total","sum","difference",
    "product","quotient","remainder","left","remaining","altogether","increase","decrease","add",
    "subtract","multiply","divide","split","share","every","all","some","none","not","no","never",
    "except","only","both","if","then","because","either","neither","must","cannot","least","most",
    "first","next","after","before","until","becomes","goal","wants","needs"}
UNIT_WORDS = {"dollars","dollar","cents","eggs","hours","hour","minutes","days","weeks","miles",
    "meters","feet","pounds","kg","grams","liters","cups","boxes","pieces","points","percent","%",
    "$","students","people","children","books","apples","oranges","cards","pages","chapters",
    "questions","problems"}
STOP = {"the","a","an","of","to","in","on","for","and","or","is","are","was","were","it","this",
    "that","with","as","at","by","from","be","been","his","her","their","what","which","how"}
CONCL_CUES = re.compile(r"(####|answer\s*(is|:)|correct\s+(option|answer)|=\s*[\d,./]+\s*$|"
                        r"therefore|so\s+the|in\s+total|makes?\b|earns?\b|is\s+the\s+answer)", re.I | re.M)

def norm_num(s):
    s = str(s).replace(",", "").strip()
    try:
        if "/" in s:
            a, b = s.split("/"); return round(float(a)/float(b), 6)
        return round(float(s), 6)
    except Exception: return None

def extract_numbers(t): return [norm_num(m) for m in NUM_RE.findall(t) if norm_num(m) is not None]

def multiset_recall(ref, cand):
    if not ref: return 1.0
    rc, cc = Counter(ref), Counter(cand)
    return sum(min(v, cc[k]) for k, v in rc.items()) / sum(rc.values())

def set_recall(ref, cand):
    return 1.0 if not ref else len(ref & cand) / len(ref)

def question_target(b_text):
    sents = re.split(r"(?<=[.!?])\s+", b_text.strip())
    qs = [s for s in sents if "?" in s]
    tgt = qs[-1] if qs else (sents[-1] if sents else "")
    return {w for w in tokens(tgt) if w not in STOP and not w.isdigit()}

def entities_of(t):
    caps = {w.lower() for w in re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z']+)", t)}
    units = {w for w in tokens(t) if w in UNIT_WORDS}
    return caps | units

def answer_support(b_answer, b_input_nums, d_text):
    if not str(b_answer).strip(): return 0.0
    ans = str(b_answer).strip(); an = norm_num(ans)
    if an is not None:
        for m in NUM_RE.finditer(d_text):
            if norm_num(m.group(1)) != an: continue
            ctx = d_text[max(0, m.start()-90):m.end()+90]
            cue = bool(CONCL_CUES.search(ctx)); in_tail = m.start() > 0.6*len(d_text)
            if (cue or in_tail) and not (an in b_input_nums and not cue):
                return 1.0
        return 0.0
    i = d_text.lower().find(ans.lower())
    if i >= 0:
        ctx = d_text[max(0, i-90):i+len(ans)+90]
        if CONCL_CUES.search(ctx) or i > 0.6*len(d_text): return 1.0
    return 0.0

def constraint_frame(df):
    rows = []
    for _, r in df.iterrows():
        b, d = r.benchmark_text, r.candidate_text
        b_nums = extract_numbers(b)
        tb, td = set(tokens(b)), set(tokens(d))
        rows.append(dict(
            a_num=multiset_recall(b_nums, extract_numbers(d)),
            a_rel=set_recall({w for w in tb if w in REL_LEXICON}, {w for w in td if w in REL_LEXICON}),
            a_tgt=set_recall(question_target(b), td),
            a_ent=set_recall(entities_of(b), entities_of(d)),
            a_ans=answer_support(r.get("benchmark_answer", ""), set(b_nums), d),
            predates=1.0 if str(r.get("candidate_date", "")) < str(r.get("benchmark_release", "9999")) else 0.0))
    return pd.DataFrame(rows, index=df.index)

# ---------------- 分组切分 / 精确校准 / 指标 ----------------
def grouped_split(groups, seed, frac=(0.4, 0.3, 0.3)):
    rng = np.random.RandomState(seed)
    g = np.array(sorted(set(groups))); rng.shuffle(g)
    n = len(g); n_tr, n_ca = int(round(frac[0]*n)), int(round(frac[1]*n))
    return set(g[:n_tr]), set(g[n_tr:n_tr+n_ca]), set(g[n_tr+n_ca:])

def cp_upper(k, n, alpha):
    if n == 0: return 1.0
    return 1.0 if k >= n else float(beta_dist.ppf(1 - alpha, k + 1, n - k))

def group_max_scores(df, z):
    out = {}
    d = df.assign(z=z)
    for (fam, gid), sub in d[d.decision_label == "retain"].groupby(["family", "group_id"]):
        out.setdefault(fam, {})[gid] = float(sub.z.max())
    return out

def select_threshold(cal_df, z_cal, eps, alpha=0.05):
    fams = sorted(cal_df.family.unique()); H = len(fams)
    gm = group_max_scores(cal_df, z_cal)
    qs_by_fam = {f: np.array(sorted(gm.get(f, {}).values())) for f in fams}
    pos = np.sort(z_cal[(cal_df.decision_label == "remove").values])
    best = None
    for tau in np.unique(np.concatenate([z_cal, [np.inf]])):
        ok = True
        for f in fams:
            qs = qs_by_fam[f]; n = len(qs)
            k = int(n - np.searchsorted(qs, tau, side="left"))
            if cp_upper(k, n, alpha / H) > eps: ok = False; break
        if not ok: continue
        cr = float((pos >= tau).mean()) if len(pos) else 0.0
        dmg = sum(int(len(qs_by_fam[f]) - np.searchsorted(qs_by_fam[f], tau)) for f in fams)
        key = (cr, -dmg, tau if np.isfinite(tau) else 1e18)
        if best is None or key > best[0]: best = (key, tau)
    return best[1] if best else np.inf

def eval_at(df, z, tau):
    rm = z >= tau
    y = (df.decision_label == "remove").values
    cr = float(rm[y].mean()) if y.sum() else float("nan")
    cdr = float(rm[~y].mean()) if (~y).sum() else float("nan")
    gm = group_max_scores(df, z)
    n_g = sum(len(v) for v in gm.values())
    dmg = sum(int(q >= tau) for v in gm.values() for q in v.values())
    fp, tp = int(rm[~y].sum()), int(rm[y].sum())
    u95 = {f: cp_upper(int((np.array(list(v.values())) >= tau).sum()), len(v), 0.05/max(1,len(gm)))
           for f, v in gm.items()}
    return dict(CR=100*cr, CDR=100*cdr, GCDR=100*(dmg/n_g if n_g else float("nan")), CRR=100*(1-cdr),
                FRPC=(fp/tp if tp else float("nan")), damaged_groups=dmg, total_groups=n_g,
                removed_clean_pairs=fp, total_clean_pairs=int((~y).sum()),
                removed_contam=tp, total_contam=int(y.sum()),
                U95_worst=100*max(u95.values()) if u95 else float("nan"),
                U95_by_family={k: 100*v for k, v in u95.items()})

def bootstrap_cr_ci(df, z, tau, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    d = df.assign(rm=(z >= tau))
    pos = d[d.decision_label == "remove"]
    by_g = {g: sub.rm.values for g, sub in pos.groupby("group_id")}
    gids = np.array(list(by_g))
    stats = []
    for _ in range(n_boot):
        take = rng.choice(gids, size=len(gids), replace=True)
        stats.append(np.concatenate([by_g[g] for g in take]).mean())
    return 100*float(np.percentile(stats, 2.5)), 100*float(np.percentile(stats, 97.5))

# ---------------- 端到端准备与评估 ----------------
def prepare(pairs, seed=42579):
    """分组切分 + 训练集拟合相似度层 + 一次性特征化(供所有变体复用)。"""
    pairs = pairs.copy()
    tr, ca, te = {}, {}, {}
    for fam, sub in pairs.groupby("family"):
        tr[fam], ca[fam], te[fam] = grouped_split(sub.group_id, seed)
    pairs["part"] = ["train" if g in tr[f] else ("cal" if g in ca[f] else "test")
                     for f, g in zip(pairs.family, pairs.group_id)]
    frames, X = {}, {}
    train = pairs[pairs.part == "train"]
    sim = SimilarityLayer().fit(list(train.benchmark_text) + list(train.candidate_text))
    for p in ["train", "cal", "test"]:
        df = pairs[pairs.part == p].reset_index(drop=True)
        Xs = sim.batch(list(df.benchmark_text), list(df.candidate_text))
        Xc = constraint_frame(df).reset_index(drop=True)
        Xall = pd.concat([Xs, Xc], axis=1)
        for c in ["a_num", "a_rel", "a_tgt", "a_ent"]:   # 精确一致指示(真实文本适配器扩展)
            Xall[c + "_x"] = (Xall[c] >= 0.999).astype(float)
        frames[p], X[p] = df, Xall
    return dict(frames=frames, X=X, sim=sim,
                splits={f: (len(tr[f]), len(ca[f]), len(te[f])) for f in tr})

def fit_score(prep, feat_cols, model="logreg"):
    Xtr = prep["X"]["train"][feat_cols]
    scaler = StandardScaler().fit(Xtr)
    ytr = (prep["frames"]["train"].decision_label == "remove").astype(int).values
    clf = (LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000) if model == "logreg"
           else MLPClassifier(hidden_layer_sizes=(16,), activation="relu", solver="lbfgs",
                              alpha=1e-3, max_iter=4000, random_state=0))
    clf.fit(scaler.transform(Xtr), ytr)
    z = {p: clf.predict_proba(scaler.transform(prep["X"][p][feat_cols]))[:, 1] for p in ["cal", "test"]}
    return z, clf, scaler

def calibrated_result(prep, z_cal, z_test, eps=0.05, alpha=0.05, with_ci=False):
    cal, test = prep["frames"]["cal"], prep["frames"]["test"]
    tau = select_threshold(cal, z_cal, eps, alpha)
    out = dict(tau=float(tau), eps=eps, cal=eval_at(cal, z_cal, tau), test=eval_at(test, z_test, tau))
    if with_ci: out["test_CR_ci95"] = bootstrap_cr_ci(test, z_test, tau)
    return out

EXACT_COLS = ["a_num_x", "a_rel_x", "a_tgt_x", "a_ent_x"]

def mlp_logits(clf, Xs):
    """MLP 的未压缩 margin(sigmoid 前的 logit)。predict_proba 在 |logit|>~37 时浮点饱和为 0/1,
    使干净组最大分与正例并列于 1.0、精确校准不可行;在 logit 上做阈值即可恢复可分性,
    且任何温度缩放都与之保序等价。"""
    h = np.maximum(0, Xs @ clf.coefs_[0] + clf.intercepts_[0])
    return (h @ clf.coefs_[1] + clf.intercepts_[1]).ravel()

def fit_score_margin(prep, feat_cols):
    Xtr = prep["X"]["train"][feat_cols]
    scaler = StandardScaler().fit(Xtr)
    ytr = (prep["frames"]["train"].decision_label == "remove").astype(int).values
    clf = MLPClassifier(hidden_layer_sizes=(16,), activation="relu", solver="lbfgs",
                        alpha=1e-3, max_iter=4000, random_state=0)
    clf.fit(scaler.transform(Xtr), ytr)
    z = {p: mlp_logits(clf, scaler.transform(prep["X"][p][feat_cols])) for p in ["cal", "test"]}
    return z, clf, scaler

def variant_scores(prep, variant):
    """variant: 'core'|'core_noind'|'mlp'|'no_constraints'|'no_chronology'|单特征名|'fusion'"""
    if variant in ("core", "core_noind", "mlp", "no_constraints", "no_chronology"):
        cols = {"core": SIM_COLS + CON_COLS + EXACT_COLS + CHRONO,
                "core_noind": SIM_COLS + CON_COLS + CHRONO,
                "mlp": SIM_COLS + CON_COLS + EXACT_COLS + CHRONO,
                "no_constraints": SIM_COLS + CHRONO,
                "no_chronology": SIM_COLS + CON_COLS + EXACT_COLS}[variant]
        z, _, _ = fit_score(prep, cols, model=("mlp" if variant == "mlp" else "logreg"))
        return z["cal"], z["test"]
    if variant == "mlp_margin":
        z, _, _ = fit_score_margin(prep, SIM_COLS + CON_COLS + EXACT_COLS + CHRONO)
        return z["cal"], z["test"]
    if variant == "fusion":
        f = lambda p: prep["X"][p][["wcos", "ccos", "lsa", "bm25n"]].mean(1).values
        return f("cal"), f("test")
    return prep["X"]["cal"][variant].values, prep["X"]["test"][variant].values
