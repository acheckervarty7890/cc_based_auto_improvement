#!/usr/bin/env python
"""Does removing the red-team region FARTHEST from the eval distribution help, and does
that depend on the probe architecture?

Earlier finding (linear_then_softmax, hu_harm / gemma-3-27b L32, 10 weight-init seeds):
deleting the far region entirely was WORTH +0.0076 mean eval AUROC over keeping it
(8/10 seeds, paired t p=0.018), while deleting an equal, label-matched number of rows
from the NEAREST region cost -0.0083 (3/10 seeds, p=0.057). Far-minus-near was +0.0158
in 10/10 seeds (p=0.0002). This script asks whether that survives a change of probe
architecture, and whether it survives a clustering that is not a single k-means fit.

Three things are done differently from that first pass:

  1. CONSENSUS CLUSTERING. The far/near regions came from one KMeans fit at one k with
     one random_state, so "the far cluster" was partly an artifact of that fit. Here an
     ensemble of fits (several k, several seeds, each on a random subsample) is reduced
     to a co-association matrix and cut with average linkage. Stability is reported
     (ARI of each ensemble member against the consensus, and per-cluster mean
     co-association) so a cluster that only exists at one k is visible as such.
  2. A 10% BUDGET, NOT A CLUSTER BOUNDARY. The far set is built by taking whole
     consensus clusters farthest-first until it reaches 10% of the red-team set, topping
     up from the last cluster by per-sample distance if that cluster would overshoot. So
     far and near remove exactly the same number of rows and the comparison is not
     confounded by how big the farthest cluster happened to be.
  3. EVERY ARCHITECTURE tuberlens exposes, over the same clustering — the clustering is
     a property of the activations, which are architecture-independent, so one clustering
     serves all arms.

Run matrix per architecture (the same design that produced the numbers above):

    full_s{S}            no removal, weight-init seed S                  -> seed-noise floor
    far_s{S}             remove the 10% farthest, seed S                 -> SEED axis, paired with full
    neardraw_d{D}_s{S0}  remove 10% drawn from the nearest region, draw D -> DRAW axis (which-rows null)
    near_s{S}            remove the fixed 10% nearest (--fixed-near-arm)
    neardraw_d0_s{S}     draw 0 at every seed (--neardraw-seed-cross)
    rand_d{D}_s{S0}      remove 10% drawn from anywhere (--with-random-arm)

The two default arms sit on different axes on purpose. `far` varies the seed against a
fixed removal set, so full-vs-far is PAIRED and the seed noise differences out. The draws
vary the rows at a fixed seed, giving the null the far effect has to clear: an effect
inside the draw spread says only that removing 74 rows moved something, not that the far
region did. Reading far_minus_full without that null is how a 0.016 shift gets mistaken
for a finding when the which-rows sd is 0.016 too.

SEED HANDLING — the one subtle thing here. ``retrain_probe``'s ``seed`` argument drives
THREE things at once: the content-deterministic train/val split, the base activation
cache key, and ``seed_everything()`` immediately before ``ProbeFactory.build``. Varying
it would therefore change the data split AND miss the activation cache, forcing
gemma-3-27b forward passes. So ``seed=42`` is held fixed and only the weight init is
varied, by patching ``agentic_redteam.evaluation.seed_everything`` — which ``retrain.py``
imports at call time, so the patch takes effect. Confirm with --check-seed-patch.

NO MODEL IS EVER LOADED. Both the red-team activations (per conversation) and the eval
activations (per split) come from cache. Fill those caches from Kaggle first, either with
--fetch-kaggle or by running scripts/fetch_kaggle_train_activations.py and
scripts/fetch_kaggle_eval_activations.py by hand.

Typical cloud-box invocation:

    python scripts/arch_cluster_ablation.py --fetch-kaggle --work-dir results/arch_abl

Stages run in order and each is skipped if its output already exists, so re-running after
an interruption resumes. Use --stages to run a subset.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
POS_LABEL = "harmful_to_human"
NEG_LABEL = "not_harmful_to_human"

# tuberlens' get_activations clips every sequence at this many tokens, so a stored
# n_tokens of exactly ACT_TOKEN_CAP means the conversation was cut off. See
# --exclude-truncated.
ACT_TOKEN_CAP = 1024

# Every ProbeType tuberlens exposes. `sklearn`, `difference_of_means` and `lda` are
# closed-form/one-pass and take seconds; the seven Adam-trained ones dominate runtime.
ALL_ARCHS = [
    "linear_then_softmax", "linear_then_mean", "linear_then_max",
    "linear_then_rolling_max", "linear_then_last", "attention", "pre_mean",
    "difference_of_means", "lda", "sklearn",
]
# Architectures whose fit has no stochastic weight init: seeding them changes nothing,
# so they are run at a single seed and the seed axis is reported as degenerate.
DETERMINISTIC_ARCHS = {"difference_of_means", "lda", "sklearn"}

STAGES = ("fetch", "pool", "cluster", "run", "analyze")


# --------------------------------------------------------------------------- #
# stage 1: pooling
# --------------------------------------------------------------------------- #
def _redteam_cache_key(messages: list[dict], *, combine: bool, convert: bool) -> str:
    """Reproduce ``retrain._redteam_activation_cache_path``'s key for one conversation."""
    basis = json.dumps(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(
        f"model={MODEL_NAME}|layer={LAYER}|combine={combine}|convert={convert}|{basis}".encode()
    ).hexdigest()[:32]


def _pair_partners(meta: list[dict], paths: dict, args) -> dict[int, int]:
    """Map each row index to its contrastive partner's row index (see --intact-pairs-only).

    The postprocessed dump carries only ``{id, inputs, label}`` with sequentially
    reassigned ids, so the original->generated link is gone by the time we read it. The
    contrastive cache is where it survives: every cached record holds both the generated
    conversation (``inputs``) and the conversation it was generated FROM
    (``original_messages``). Hashing each with the same content key `stage_pool` used to
    name the activation blobs joins both sides back to their rows.

    Cached rows are matched by content, so entries written for conversations that a later
    ``filter_dataset`` pass dropped simply find no row and are ignored — the cache is a
    superset of any one iteration's dump (488 records vs. 439 pairs at iter3).
    """
    src = paths["probe_dir"] / "contrastive_cache.jsonl"
    if not src.is_file():
        raise SystemExit(
            f"--intact-pairs-only needs the contrastive cache at {src}, which records "
            f"which generated conversation came from which original. Without it there is "
            f"no way to tell a pair from an orphan."
        )
    key_of = {m["key"]: i for i, m in enumerate(meta)}
    partner: dict[int, int] = {}
    unmatched = 0
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line).get("record")
        if isinstance(rec, str):          # older rows stored the dict's repr, not JSON
            rec = ast.literal_eval(rec)
        gk = _redteam_cache_key(rec["inputs"], combine=args.combine, convert=args.convert)
        ok = _redteam_cache_key(rec["original_messages"], combine=args.combine,
                                convert=args.convert)
        gi, oi = key_of.get(gk), key_of.get(ok)
        if gi is None or oi is None:
            unmatched += 1
            continue
        partner[gi], partner[oi] = oi, gi
    print(f"pairs: {len(partner)//2} pairs recovered from {src.name} "
          f"({unmatched} cache rows matched no conversation in this iteration)")
    return partner


def _masked_mean(acts, mask):
    """[B,T,H] activations + [B,T] mask -> [B,H] float32.

    Mean over real tokens is the right summary here because it is how the probe family
    aggregates: every LinearThenAgg computes a per-token logit and pools it, so two
    conversations with the same masked-mean activation get the same score from a
    linear head regardless of length.
    """
    import torch
    a = acts.to(torch.float32)
    m = mask.to(torch.float32).unsqueeze(-1)
    return (a * m).sum(1) / m.sum(1).clamp(min=1)


def stage_pool(args, paths: dict) -> None:
    import torch

    out = paths["pooled"]
    if (out / "redteam_pooled.npy").exists() and (out / "eval_pooled.npy").exists():
        print("pool: already done, skipping")
        return
    out.mkdir(parents=True, exist_ok=True)

    blob_dir = paths["base_cache"] / f"redteam_acts_{MODEL_NAME.replace('/', '_')}_L{LAYER}"
    if not blob_dir.is_dir():
        raise SystemExit(
            f"pool: no red-team activation blobs at {blob_dir}\n"
            f"      run with --fetch-kaggle, or scripts/fetch_kaggle_train_activations.py"
        )

    src = paths["probe_dir"] / f"redteam_postprocessed_iter{args.iteration}.jsonl"
    rows = []
    seen = set()
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        msgs = r["inputs"] if isinstance(r["inputs"], list) else json.loads(r["inputs"])
        key = _redteam_cache_key(msgs, combine=args.combine, convert=args.convert)
        if key in seen:      # the dump can repeat a conversation across ids; the cache
            continue         # is keyed by content, so it is one activation either way
        seen.add(key)
        rows.append({"key": key, "label": r["label"], "messages": msgs})
    print(f"pool: {len(rows)} unique conversations from {src.name}", flush=True)

    missing = [r["key"] for r in rows if not (blob_dir / f"{r['key']}.pt").is_file()]
    if missing:
        raise SystemExit(
            f"pool: {len(missing)} of {len(rows)} conversations have no cached activation "
            f"(e.g. {missing[0]}). The cache is for a different iteration or different "
            f"combine/convert flags — fetch the iter{args.iteration} dataset."
        )

    rt = None
    for i, r in enumerate(rows):
        b = torch.load(blob_dir / f"{r['key']}.pt", map_location="cpu", weights_only=False)
        v = _masked_mean(b["activations"], b["attention_mask"])[0].numpy()
        if rt is None:
            rt = np.zeros((len(rows), v.shape[0]), dtype=np.float32)
        rt[i] = v
        r["n_tokens"] = int(b["attention_mask"].sum())
        if (i + 1) % 200 == 0:
            print(f"  pooled {i+1}/{len(rows)}", flush=True)

    np.save(out / "redteam_pooled.npy", rt)
    (out / "redteam_meta.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    ev_vecs, ev_meta = [], []
    for p in sorted(paths["eval_cache"].glob("*-acts_full.pt")):
        split = p.name.split("-acts_full.pt")[0]
        split_jsonl = paths["eval_dir"] / f"{split}.jsonl"
        if not split_jsonl.is_file():
            print(f"  eval {split}: no split JSONL, skipping")
            continue
        b = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        acts, mask = b["activations"], b["attention_mask"]
        labels = [json.loads(l)["labels"] for l in split_jsonl.read_text().splitlines() if l.strip()]
        if len(labels) != acts.shape[0]:
            raise SystemExit(f"pool: {split} blob has {acts.shape[0]} rows, "
                             f"JSONL has {len(labels)} — stale cache.")
        chunks = [_masked_mean(acts[s:s + 32], mask[s:s + 32]).numpy()
                  for s in range(0, acts.shape[0], 32)]
        ev_vecs.append(np.concatenate(chunks, 0))
        ev_meta += [{"split": split, "label": lb} for lb in labels]
        print(f"  eval {split}: {len(labels)} rows", flush=True)
        del b, acts, mask

    if not ev_vecs:
        raise SystemExit(f"pool: no eval blobs in {paths['eval_cache']}")
    np.save(out / "eval_pooled.npy", np.concatenate(ev_vecs, 0))
    (out / "eval_meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in ev_meta))
    print(f"pool: wrote {out}")


# --------------------------------------------------------------------------- #
# stage 2: consensus clustering
# --------------------------------------------------------------------------- #
def _zscore(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _consensus(X: np.ndarray, *, k_range, n_runs, subsample, rng_seed, name):
    """Ensemble k-means -> co-association matrix -> average-linkage partition.

    Each ensemble member sees a random ``subsample`` of the rows at a k drawn from
    ``k_range``, so a boundary that only exists at one k, or only when a particular row
    is present, cannot survive into the consensus. ``C[i,j]`` is the fraction of members
    in which i and j were BOTH sampled and landed together, so the denominator is
    per-pair rather than the run count.
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    n = len(X)
    Z = _zscore(X)
    co = np.zeros((n, n), dtype=np.float32)
    cnt = np.zeros((n, n), dtype=np.float32)
    rng = np.random.default_rng(rng_seed)
    members = []
    m_size = max(2, int(round(subsample * n)))

    print(f"\n{name}: consensus over {n_runs} fits, k in {list(k_range)}, "
          f"{subsample:.0%} subsample ({m_size}/{n} rows)", flush=True)
    for r in range(n_runs):
        k = k_range[r % len(k_range)]
        sel = rng.choice(n, m_size, replace=False)
        km = KMeans(n_clusters=k, n_init=10,
                    random_state=int(rng.integers(1 << 31))).fit(Z[sel])
        assign = np.full(n, -1, dtype=np.int32)
        assign[sel] = km.labels_
        present = assign >= 0
        cnt[np.ix_(present, present)] += 1
        for c in range(k):
            m = np.where(assign == c)[0]
            if len(m):
                co[np.ix_(m, m)] += 1
        members.append(assign)
        if (r + 1) % 10 == 0:
            print(f"  fit {r+1}/{n_runs}", flush=True)

    C = co / np.maximum(cnt, 1.0)
    np.fill_diagonal(C, 1.0)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)

    best = None
    for k in k_range:
        lab = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                      linkage="average").fit_predict(D)
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(D, lab, metric="precomputed")
        sizes = sorted(Counter(lab).values(), reverse=True)
        print(f"  consensus k={k:2d} sil={s:+.4f} sizes={sizes}", flush=True)
        if best is None or s > best[0]:
            best = (s, k, lab)
    if best is None:
        raise SystemExit(f"{name}: consensus produced no usable partition")
    sil, k_final, labels = best

    aris = [adjusted_rand_score(a[a >= 0], labels[a >= 0]) for a in members]
    intra = {}
    for c in sorted(set(labels)):
        m = np.where(labels == c)[0]
        intra[int(c)] = float(C[np.ix_(m, m)].mean()) if len(m) > 1 else 1.0
    print(f"  -> k={k_final} (sil={sil:+.4f}); ensemble ARI vs consensus "
          f"mean={np.mean(aris):.3f} min={np.min(aris):.3f} max={np.max(aris):.3f}")
    print(f"  -> mean within-cluster co-association: "
          + " ".join(f"c{c}={v:.2f}" for c, v in sorted(intra.items())))
    return labels, {"k": int(k_final), "silhouette": float(sil),
                    "ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris)),
                    "ari_max": float(np.max(aris)), "intra_coassoc": intra}


def _cosd(a, b):
    return float(1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _distinctive_terms(meta, idx_a, idx_b, top=12):
    """Words most characteristic of set A vs set B, by smoothed log-odds ratio.

    Add-1 smoothed so a term appearing in one set only doesn't produce an infinite score,
    and restricted to terms seen at least 5 times overall so the list isn't noise.
    """
    import re
    from collections import Counter as C

    def counts(idx):
        c = C()
        for i in idx:
            text = " ".join(m["content"] for m in meta[i]["messages"]).lower()
            c.update(set(re.findall(r"[a-z][a-z'-]{2,}", text)))  # per-doc presence
        return c

    ca, cb = counts(idx_a), counts(idx_b)
    na, nb = len(idx_a), len(idx_b)
    scores = {}
    for w in set(ca) | set(cb):
        if ca[w] + cb[w] < 5:
            continue
        pa = (ca[w] + 1) / (na + 2)
        pb = (cb[w] + 1) / (nb + 2)
        scores[w] = np.log(pa / (1 - pa)) - np.log(pb / (1 - pb))
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [w for w, _ in ranked[:top]], [w for w, _ in ranked[-top:]][::-1]


def _report_region_differences(meta, tok, dist, far, near, near_pool, near_draws):
    """Describe how the far and near regions differ, now that nothing is matched.

    Selection is by distance alone, so every other difference between the two regions is a
    free variable. Anything large here is a candidate alternative explanation for whatever
    the ablation shows, and belongs in the write-up alongside the AUROC numbers.
    """
    corpus = list(range(len(meta)))
    near_union = sorted(set().union(*(set(d) for d in near_draws))) if near_draws else near_pool
    groups = {"far": far, "near": near, "near(pool)": near_pool,
              "near(drawn)": near_union, "corpus": corpus}

    def stat(idx):
        t = tok[idx]
        nmsg = np.array([len(meta[i]["messages"]) for i in idx], dtype=float)
        return {
            "n": len(idx),
            "pos_frac": float(np.mean([meta[i]["label"] == "positive" for i in idx])),
            "tok_median": float(np.median(t)), "tok_mean": float(t.mean()),
            "tok_p10": float(np.percentile(t, 10)), "tok_p90": float(np.percentile(t, 90)),
            "at_1024_cap": float(np.mean(t >= 1024)),
            "n_messages_median": float(np.median(nmsg)),
            "dist_mean": float(dist[idx].mean()),
        }

    prof = {k: stat(v) for k, v in groups.items()}
    print("\n--- how the regions differ (nothing is matched; distance is the only criterion) ---")
    print(f"{'':<13}{'n':>6}{'%pos':>7}{'medTok':>8}{'p10':>7}{'p90':>7}{'@cap':>7}"
          f"{'msgs':>6}{'dist':>8}")
    for k, s in prof.items():
        print(f"{k:<13}{s['n']:>6}{s['pos_frac']:>7.0%}{s['tok_median']:>8.0f}"
              f"{s['tok_p10']:>7.0f}{s['tok_p90']:>7.0f}{s['at_1024_cap']:>7.0%}"
              f"{s['n_messages_median']:>6.0f}{s['dist_mean']:>8.4f}")

    hi, lo = _distinctive_terms(meta, far, near)
    print(f"  vocabulary skewed toward FAR : {', '.join(hi)}")
    print(f"  vocabulary skewed toward NEAR: {', '.join(lo)}")
    prof["distinctive_far"], prof["distinctive_near"] = hi, lo
    return prof


def stage_cluster(args, paths: dict) -> None:
    out = paths["work"] / "clusters.json"
    if out.exists() and not args.recluster:
        print("cluster: clusters.json exists, skipping (--recluster to redo)")
        return

    P = paths["pooled"]
    rt_all = np.load(P / "redteam_pooled.npy")
    ev = np.load(P / "eval_pooled.npy")
    meta_all = [json.loads(l) for l in (P / "redteam_meta.jsonl").read_text().splitlines()]
    ev_meta = [json.loads(l) for l in (P / "eval_meta.jsonl").read_text().splitlines()]
    k_range = list(range(args.k_min, args.k_max + 1))

    # ---- optionally drop the rows the 1024-token activation cap cut off ----
    # A truncated row's pooled vector averages only its first 1024 tokens, so where it
    # lands in activation space is partly an artifact of where the cap fell rather than of
    # what the conversation is — and truncation is not spread evenly, so it concentrates in
    # whichever region happens to be long-winded and quietly becomes a second selection
    # criterion alongside distance. `n_tokens` is min(true_len, 1024), so `>= cap` flags
    # exactly the truncated rows; verified against a full re-tokenization of this corpus
    # (72/878 flagged, no row's true length is exactly 1024, so no false positives).
    # Everything below is computed in LOCAL index space over the kept rows and mapped back
    # through `keep_idx` at the end, so every index written to clusters.json still points
    # into the unfiltered redteam_meta.jsonl.
    if args.exclude_truncated:
        keep_idx = [i for i, m in enumerate(meta_all) if m["n_tokens"] < ACT_TOKEN_CAP]
        dropped = [i for i in range(len(meta_all)) if i not in set(keep_idx)]
        dl = Counter(meta_all[i]["label"] for i in dropped)
        print(f"cluster: excluding {len(dropped)} truncated rows "
              f"({dl['positive']} pos / {dl['negative']} neg); "
              f"{len(keep_idx)} of {len(meta_all)} rows remain")
    else:
        keep_idx, dropped = list(range(len(meta_all))), []

    # ---- optionally keep only rows whose contrastive partner is also present ----
    # The dump is emitted in matched pairs (an original and its opposite-class rewrite);
    # `generate_contrastive_dataset` drops a source whose generation failed, so every row
    # starts with a partner. The truncation filter above breaks that: it removes 71
    # generated rows against 1 original, leaving 70 orphans and shifting the corpus from
    # 439/439 to 372 neg / 434 pos. An arm that removes an orphan is a different kind of
    # intervention from one that removes a whole pair, so leaving them in makes
    # "pairs removed" a second, uncontrolled criterion alongside distance.
    if args.intact_pairs_only:
        partner = _pair_partners(meta_all, paths, args)
        present = set(keep_idx)
        intact = [i for i in keep_idx if partner.get(i, -1) in present]
        orph = [i for i in keep_idx if i not in set(intact)]
        ol = Counter(meta_all[i]["label"] for i in orph)
        keep_idx = intact
        dropped = sorted(set(dropped) | set(orph))
        print(f"cluster: excluding {len(orph)} rows orphaned by the filter "
              f"({ol['positive']} pos / {ol['negative']} neg); {len(keep_idx)} rows = "
              f"{len(keep_idx)//2} intact pairs remain")

    rt = rt_all[keep_idx]
    meta = [meta_all[i] for i in keep_idx]

    lab_rt, stab_rt = _consensus(rt, k_range=k_range, n_runs=args.ensemble_runs,
                                 subsample=args.subsample, rng_seed=args.cluster_seed,
                                 name=f"RED-TEAM (n={len(rt)})")
    lab_ev, stab_ev = _consensus(ev, k_range=k_range, n_runs=args.ensemble_runs,
                                 subsample=args.subsample, rng_seed=args.cluster_seed + 1,
                                 name=f"EVAL (n={len(ev)})")

    ev_cent = np.stack([ev[lab_ev == c].mean(0) for c in sorted(set(lab_ev))])
    print(f"\nEVAL consensus clusters (k={len(ev_cent)}):")
    for i, c in enumerate(sorted(set(lab_ev))):
        sel = [m for m, l in zip(ev_meta, lab_ev) if l == c]
        print(f"  ev{i}: n={len(sel):4d} splits={dict(Counter(m['split'] for m in sel).most_common())} "
              f"harmful={sum(m['label'] == POS_LABEL for m in sel)}/{len(sel)}")

    # Distance to the NEAREST eval cluster, not to the eval grand centroid: the eval set
    # is multi-modal, so its grand centroid is a point no eval data occupies and every
    # red-team cluster looks equidistant from it.
    per_sample_d = np.array([min(_cosd(rt[i], ev_cent[j]) for j in range(len(ev_cent)))
                             for i in range(len(rt))])

    clusters = []
    for c in sorted(set(lab_rt)):
        m = np.where(lab_rt == c)[0]
        cent = rt[m].mean(0)
        d = [_cosd(cent, ev_cent[j]) for j in range(len(ev_cent))]
        lc = Counter(meta[i]["label"] for i in m)
        clusters.append({
            "c": int(c), "n": len(m), "min_cos": min(d), "nearest_ev": int(np.argmin(d)),
            "mean_sample_cos": float(per_sample_d[m].mean()),
            "n_pos": lc.get("positive", 0), "n_neg": lc.get("negative", 0),
            "intra_coassoc": stab_rt["intra_coassoc"][int(c)],
            "members": [int(i) for i in m],
        })

    print(f"\nRED-TEAM consensus clusters (k={len(clusters)}), farthest eval cluster first:")
    print(f"{'c':>3}{'n':>6}{'minCos':>9}{'->ev':>6}{'perSampCos':>12}{'pos/neg':>11}{'stab':>7}")
    for r in sorted(clusters, key=lambda r: -r["min_cos"]):
        print(f"{r['c']:>3}{r['n']:>6}{r['min_cos']:>9.4f}{r['nearest_ev']:>6}"
              f"{r['mean_sample_cos']:>12.4f}{r['n_pos']:>6}/{r['n_neg']:<4}"
              f"{r['intra_coassoc']:>7.2f}")

    # ---- far set: whole clusters farthest-first up to the budget ----
    tok = np.array([m["n_tokens"] for m in meta], dtype=float)
    target = int(round(args.target_frac * len(rt)))
    far, far_from = [], []
    if args.class_balanced_removal:
        # Farthest-first WITHIN each class, so removal leaves the training set's class
        # balance untouched and neither far-vs-full nor near-vs-full is confounded by it.
        cap = {"positive": target // 2, "negative": target - target // 2}
        got: Counter = Counter()
        for r in sorted(clusters, key=lambda r: -r["min_cos"]):
            took = 0
            for i in sorted(r["members"], key=lambda i: -per_sample_d[i]):
                lb = meta[i]["label"]
                if got[lb] < cap[lb]:
                    far.append(i)
                    got[lb] += 1
                    took += 1
            if took:
                far_from.append({"c": r["c"], "took": took, "of": r["n"], "whole": False})
            if len(far) >= target:
                break
    else:
        for r in sorted(clusters, key=lambda r: -r["min_cos"]):
            if len(far) >= target:
                break
            room = target - len(far)
            if len(r["members"]) <= room:
                far += r["members"]
                far_from.append({"c": r["c"], "took": len(r["members"]), "of": r["n"],
                                 "whole": True})
            else:
                # This cluster would overshoot: take its own farthest members so the
                # budget is exact and far/near remove the same count.
                top = sorted(r["members"], key=lambda i: -per_sample_d[i])[:room]
                far += top
                far_from.append({"c": r["c"], "took": len(top), "of": r["n"], "whole": False})
    far = sorted(far)
    n_pos = sum(meta[i]["label"] == "positive" for i in far)
    n_neg = len(far) - n_pos
    print(f"\nFAR set: {len(far)} rows ({args.target_frac:.0%} of {len(rt)}), "
          f"{n_pos} pos / {n_neg} neg, median {np.median(tok[far]):.0f} tokens "
          f"(corpus median {np.median(tok):.0f}), from "
          + ", ".join(f"c{d['c']}({d['took']}/{d['of']}{'' if d['whole'] else ' partial'})"
                      for d in far_from))

    # ---- near set: the exact mirror of the far construction ----
    # Whole clusters nearest-first to the same budget, topped up from the last cluster by
    # per-sample distance (nearest first). This is the arm that makes far-vs-near a clean
    # contrast: one fixed set on each side, same size, built by the same rule, differing
    # only in the sign of the distance ordering. The sampled near_pool below answers a
    # different question — how much of any near effect is just which rows got drawn.
    near, near_from = [], []
    for r in sorted(clusters, key=lambda r: r["min_cos"]):
        if len(near) >= target:
            break
        room = target - len(near)
        if len(r["members"]) <= room:
            near += r["members"]
            near_from.append({"c": r["c"], "took": len(r["members"]), "of": r["n"],
                              "whole": True})
        else:
            top = sorted(r["members"], key=lambda i: per_sample_d[i])[:room]
            near += top
            near_from.append({"c": r["c"], "took": len(top), "of": r["n"], "whole": False})
    near = sorted(near)
    near_pos = sum(meta[i]["label"] == "positive" for i in near)
    print(f"NEAR set: {len(near)} rows, {near_pos} pos / {len(near)-near_pos} neg, "
          f"median {np.median(tok[near]):.0f} tokens, "
          f"mean dist {per_sample_d[near].mean():.4f} (far {per_sample_d[far].mean():.4f}), "
          f"from " + ", ".join(f"c{d['c']}({d['took']}/{d['of']}"
                               f"{'' if d['whole'] else ' partial'})" for d in near_from))

    # ---- near pool: nearest clusters, grown to a multiple of the removal budget ----
    # Distance is the ONLY selection criterion on either side — no class-balance or length
    # matching is imposed. That is what keeps the near pool genuinely near: an earlier
    # version matched the far set's 4-pos/84-neg composition, and satisfying that demand
    # dragged the pool out to 72% of the corpus, at which point "near" meant "not far".
    # Sizing on the budget alone stops it at the nearest few clusters. Whatever else
    # differs between the two regions is measured afterwards and reported, not controlled.
    near_pool, near_pool_from = [], []
    for r in sorted(clusters, key=lambda r: r["min_cos"]):
        near_pool += r["members"]
        near_pool_from.append(r["c"])
        if len(near_pool) >= args.near_pool_mult * target:
            break
    if len(near_pool) < target:
        raise SystemExit(f"cluster: near pool has {len(near_pool)} rows, need {target}")
    sup = Counter(meta[i]["label"] for i in near_pool)
    print(f"NEAR pool: {len(near_pool)} rows ({len(near_pool)/len(rt):.0%} of the corpus) "
          f"from c" + ", c".join(str(c) for c in near_pool_from)
          + f" — {sup['positive']} pos / {sup['negative']} neg, "
          f"median {np.median(tok[near_pool]):.0f} tokens, "
          f"mean dist {per_sample_d[near_pool].mean():.4f} vs far "
          f"{per_sample_d[far].mean():.4f}")

    def draws_from(pool, seed):
        """`args.draws` unconstrained random removal sets of `target` rows from `pool`."""
        rng = np.random.default_rng(seed)
        return [sorted(int(x) for x in rng.choice(pool, target, replace=False))
                for _ in range(args.draws)]

    near_draws = draws_from(near_pool, args.draw_seed)
    rand_draws = (draws_from(list(range(len(rt))), args.draw_seed + 1)
                  if args.with_random_arm else [])

    ov = [len(set(a) & set(b)) / len(a) for a, b in itertools.combinations(near_draws, 2)]
    print(f"NEAR draws: {len(near_draws)} x {len(near_draws[0])} rows, unconstrained; "
          f"pairwise overlap mean={np.mean(ov):.1%} max={np.max(ov):.1%}")
    if np.mean(ov) > 0.5:
        print("  WARNING: draws overlap >50% — raise --near-pool-mult or the draw-to-draw "
              "spread will understate the true sampling variance.")

    profile = _report_region_differences(meta, tok, per_sample_d, far, near, near_pool,
                                         near_draws)

    # Local (kept-row) indices -> indices into the unfiltered redteam_meta.jsonl, so
    # stage_run can keep reading the one meta file it always has.
    def g(idx):
        return [keep_idx[i] for i in idx]

    json.dump({
        "n_redteam": len(rt), "n_eval": len(ev), "target": target,
        "target_frac": args.target_frac,
        "exclude_truncated": bool(args.exclude_truncated),
        "excluded": dropped, "kept_idx": keep_idx, "n_redteam_all": len(meta_all),
        "labels_redteam": [int(x) for x in lab_rt], "labels_eval": [int(x) for x in lab_ev],
        "stability_redteam": stab_rt, "stability_eval": stab_ev,
        "clusters": [dict(c, members=g(c["members"])) for c in clusters],
        "far": g(far), "far_from": far_from, "far_n_pos": n_pos, "far_n_neg": n_neg,
        "near": g(near), "near_from": near_from,
        "near_n_pos": near_pos, "near_n_neg": len(near) - near_pos,
        "near_pool": g(near_pool), "near_pool_from": near_pool_from,
        "near_draws": [g(d) for d in near_draws], "rand_draws": [g(d) for d in rand_draws],
        "per_sample_min_cos": per_sample_d.tolist(),
        "diagnostics": {
            "class_balanced_removal": args.class_balanced_removal,
            "near_pool_frac_of_corpus": len(near_pool) / len(rt),
            "draw_overlap_mean": float(np.mean(ov)), "draw_overlap_max": float(np.max(ov)),
            "far_near_overlap": len(set(far) & set(near)),
        },
        "region_profile": profile,
    }, open(out, "w"))
    print(f"\ncluster: wrote {out}")


# --------------------------------------------------------------------------- #
# stage 3: retrain + eval
# --------------------------------------------------------------------------- #
def _to_record(m: dict) -> dict:
    """Wrap a pooled-meta row back into the AttemptRecord shape retrain_probe reads."""
    j = POS_LABEL if m["label"] == "positive" else NEG_LABEL
    return {"sample": {"messages": m["messages"]}, "probe_score": 0.5,
            "probe_predicts_positive": j == NEG_LABEL, "judge_label": j,
            "judge_reason": "", "judge_confidence": 10, "success": True,
            "attacker_model": "arch_cluster_ablation", "run_id": "arch_abl",
            "round": 0, "iteration": 0,
            "error_type": "false_negative" if j == POS_LABEL else "false_positive",
            "pos_class_label": POS_LABEL, "neg_class_label": NEG_LABEL}


def stage_run(args, paths: dict) -> None:
    import pandas as pd

    import agentic_redteam.evaluation as _ev
    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import retrain_probe

    real_seed_everything = _ev.seed_everything

    P = paths["pooled"]
    meta = [json.loads(l) for l in (P / "redteam_meta.jsonl").read_text().splitlines()]
    cl = json.load(open(paths["work"] / "clusters.json"))
    res_dir = paths["work"] / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Rows the clustering excluded are excluded from EVERY arm, `full` included — they are
    # out of the corpus, not a removal being tested, so leaving them in `full` would make
    # the baseline a different dataset from the arms it is the baseline for.
    excluded = cl.get("excluded", [])
    if excluded:
        print(f"run: {len(excluded)} truncated rows excluded from every arm "
              f"(corpus is {len(meta)-len(excluded)} of {len(meta)} rows)")

    jobs = []
    for arch in args.archs:
        seeds = [args.seeds[0]] if arch in DETERMINISTIC_ARCHS else args.seeds
        for s in seeds:
            jobs.append((arch, f"full_s{s}", [], s))
            jobs.append((arch, f"far_s{s}", cl["far"], s))
            # The fixed near set is the deterministic mirror of far. It is off by default
            # because the sampled `neardraw` arm answers the same question better: a
            # single fixed near set cannot distinguish "this region matters" from "these
            # 74 rows happened to matter", which is exactly what the draw spread measures.
            if args.fixed_near_arm and cl.get("near"):
                jobs.append((arch, f"near_s{s}", cl["near"], s))
        # `neardraw`, not `near` — `arm` is parsed off the name prefix, so the sampled
        # arm needs a name the fixed near set can't be confused with.
        for d, drop in enumerate(cl["near_draws"]):
            jobs.append((arch, f"neardraw_d{d}_s{seeds[0]}", drop, seeds[0]))
        # Draw 0 re-run across the other seeds: the only cell where the seed and draw axes
        # cross, which is what makes init noise and which-rows noise directly comparable.
        # Off by default — with `seeds` long enough, `full` already estimates init noise
        # on identical data, and these jobs cost one per extra seed for a second estimate.
        if args.neardraw_seed_cross:
            for s in seeds[1:]:
                jobs.append((arch, f"neardraw_d0_s{s}", cl["near_draws"][0], s))
        for d, drop in enumerate(cl["rand_draws"]):
            jobs.append((arch, f"rand_d{d}_s{seeds[0]}", drop, seeds[0]))

    todo = [j for j in jobs if not (res_dir / f"{j[0]}__{j[1]}.csv").exists()]
    print(f"\nrun: {len(jobs)} jobs across {len(args.archs)} architecture(s); "
          f"{len(jobs)-len(todo)} already done, {len(todo)} to go", flush=True)
    if args.dry_run:
        for a, n, drop, s in todo[:20]:
            print(f"  {a:24s} {n:16s} drop={len(drop):3d} seed={s}")
        if len(todo) > 20:
            print(f"  ... and {len(todo)-20} more")
        return

    t_start = time.time()
    failed = []
    for n_done, (arch, name, drop, seed) in enumerate(todo, 1):
        csv = res_dir / f"{arch}__{name}.csv"
        dropset = set(drop) | set(excluded)
        keep = [m for i, m in enumerate(meta) if i not in dropset]
        jsonl = res_dir / f".rt_{arch}__{name}.jsonl"
        jsonl.write_text("".join(json.dumps(_to_record(m), ensure_ascii=False) + "\n"
                                 for m in keep))
        probe = res_dir / f".p_{arch}__{name}.pkl"
        t0 = time.time()
        try:
            # Only the weight init varies; retrain_probe's own seed stays 42 so the
            # train/val split and every activation cache key are unchanged.
            _ev.seed_everything = lambda _s, _S=seed: real_seed_everything(_S)
            try:
                retrain_probe(
                    jsonl_path=jsonl,
                    base_probe_path=paths["probe_dir"] / "probe_iter0.pkl",
                    base_training_data_path=paths["base_data"],
                    new_probe_path=probe,
                    probe_spec=arch,
                    preprocessing=None, min_judge_confidence=0,
                    test_size=args.test_size, split_field=None, seed=args.split_seed,
                    base_activation_cache_dir=paths["base_cache"],
                    combine_consecutive_messages=args.combine,
                    convert_tool_to_assistant=args.convert,
                    verbose=False,
                )
            finally:
                _ev.seed_everything = real_seed_everything
            df = evaluate_probe(
                probe_path=probe, eval_dataset_dir=paths["eval_dir"],
                activations_cache_dir=paths["eval_cache"], max_samples=None,
                seed=args.split_seed, combine_consecutive_messages=args.combine,
                convert_tool_to_assistant=args.convert,
            )
        except Exception as e:
            failed.append((arch, name, repr(e)))
            print(f"  [{n_done}/{len(todo)}] {arch:24s} {name:16s} FAILED: "
                  f"{type(e).__name__}: {e}", flush=True)
            (res_dir / f"{arch}__{name}.FAILED").write_text(f"{type(e).__name__}: {e}\n")
            continue
        finally:
            probe.unlink(missing_ok=True)
            jsonl.unlink(missing_ok=True)

        arm = name.split("_d")[0].split("_s")[0]
        df = df.assign(arch=arch, run=name, arm=arm, n_kept=len(keep),
                       n_dropped=len(drop), n_excluded=len(excluded), init_seed=seed)
        df.to_csv(csv, index=False)
        r = df[df.dataset == "mean"].iloc[0]
        el = time.time() - t_start
        eta = el / n_done * (len(todo) - n_done)
        print(f"  [{n_done}/{len(todo)}] {arch:24s} {name:16s} n={len(keep):4d} "
              f"seed={seed:<4d} AUROC={r.auroc:.4f} acc={r.accuracy:.4f} "
              f"({time.time()-t0:.0f}s, ETA {eta/60:.0f}m)", flush=True)

    if failed:
        print(f"\nrun: {len(failed)} job(s) failed:")
        for a, n, e in failed:
            print(f"  {a} {n}: {e}")


# --------------------------------------------------------------------------- #
# stage 4: analysis
# --------------------------------------------------------------------------- #
def stage_analyze(args, paths: dict) -> None:
    import pandas as pd
    from scipy import stats

    res_dir = paths["work"] / "results"
    files = sorted(res_dir.glob("*__*.csv"))
    if not files:
        raise SystemExit("analyze: no results yet")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df.to_csv(paths["work"] / "all_splits.csv", index=False)
    mean = df[df.dataset == "mean"].copy()
    mean.to_csv(paths["work"] / "all_means.csv", index=False)

    rows = []
    # Two axes, reported separately because they measure different noise sources:
    #   SEED  — full vs far over the same weight-init seeds, paired and t-tested. The
    #           pairing is what makes it sensitive; an unpaired comparison is swamped by
    #           the seed spread, which is the same order as the effect.
    #   DRAW  — the neardraw replicates at one fixed seed. Their spread is the "which rows
    #           did you happen to remove" null, and `far_pctile` locates the far arm in it.
    #           An effect inside that null is not evidence about the far REGION, only that
    #           removing 74 rows moved something.
    print(f"\n{'arch':<24}{'n_seed':>7}{'full':>9}{'far':>9}{'far-full':>10}{'p':>8}"
          f"{'w/l':>7}{'seed_sd':>9}{'draw_mean':>10}{'draw_sd':>9}{'far_pct':>8}")
    for arch in sorted(mean.arch.unique()):
        a = mean[mean.arch == arch]
        full = a[a.run.str.startswith("full_")].set_index("init_seed").auroc
        far = a[a.run.str.startswith("far_")].set_index("init_seed").auroc
        # `neardraw_d{D}_s{S}`; the older `near_d{D}_` spelling is accepted so results
        # written before the rename still aggregate.
        draw = a[a.run.str.contains(r"^near(?:draw)?_d\d+_", regex=True)].auroc
        seeds = sorted(set(full.index) & set(far.index))
        if not seeds:
            continue
        f_, r_ = full[seeds].values, far[seeds].values
        d = r_ - f_
        p_ff = stats.ttest_rel(r_, f_).pvalue if len(seeds) > 1 else float("nan")
        wins = int((d > 0).sum())
        seed_sd = float(np.std(np.concatenate([f_, r_ - d.mean()]), ddof=1)) \
            if len(seeds) > 1 else 0.0
        # Where the far arm sits in the draw null, compared at the draw arm's own seed.
        dm = float(draw.mean()) if len(draw) else float("nan")
        dsd = float(draw.std(ddof=1)) if len(draw) > 1 else float("nan")
        pct = float((draw < r_.mean()).mean() * 100) if len(draw) else float("nan")
        rows.append({"arch": arch, "n_seeds": len(seeds),
                     "full": f_.mean(), "full_sd": f_.std(ddof=1) if len(seeds) > 1 else 0.0,
                     "far": r_.mean(), "far_sd": r_.std(ddof=1) if len(seeds) > 1 else 0.0,
                     "far_minus_full": d.mean(), "p_far_vs_full": p_ff,
                     "far_beats_full": f"{wins}/{len(seeds)}", "seed_sd": seed_sd,
                     "neardraw_mean": dm, "neardraw_sd": dsd, "n_draws": int(len(draw)),
                     "far_pctile_in_draws": pct})
        print(f"{arch:<24}{len(seeds):>7}{f_.mean():>9.4f}{r_.mean():>9.4f}"
              f"{d.mean():>+10.4f}{p_ff:>8.4f}{wins:>4}/{len(seeds)}{seed_sd:>9.4f}"
              f"{dm:>10.4f}{dsd:>9.4f}{pct:>7.0f}%")

    summary = pd.DataFrame(rows)
    summary.to_csv(paths["work"] / "arch_summary.csv", index=False)

    per_split = (df[df.dataset != "mean"]
                 .pivot_table(index=["arch", "dataset"], columns="arm",
                              values="auroc", aggfunc="mean"))
    per_split.to_csv(paths["work"] / "per_split.csv")
    print(f"\nwrote {paths['work']/'arch_summary.csv'}, "
          f"{paths['work']/'per_split.csv'}, {paths['work']/'all_means.csv'}")

    if summary.empty:
        return
    best = summary.sort_values("far_minus_full", ascending=False).iloc[0]
    print(f"\nlargest far-minus-full: {best.arch} {best.far_minus_full:+.4f} "
          f"(p={best.p_far_vs_full:.4f}, {best.far_beats_full}), "
          f"{best.far_pctile_in_draws:.0f}th pctile of its own near-draw null")
    print("Read far_minus_full against BOTH seed_sd and neardraw_sd: an effect smaller "
          "than either is not resolved by this design.")


# --------------------------------------------------------------------------- #
def stage_fetch(args, paths: dict) -> None:
    py = sys.executable
    cmds = [
        [py, str(REPO_ROOT / "scripts" / "fetch_kaggle_train_activations.py"),
         "--cache-dir", str(paths["base_cache"]), "--arm", args.arm,
         "--iteration", str(args.iteration)],
        [py, str(REPO_ROOT / "scripts" / "fetch_kaggle_eval_activations.py"),
         "--concept", "hu_harm", "--cache-dir", str(paths["eval_cache"])],
    ]
    for cmd in cmds:
        print(f"\n$ {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


def _check_seed_patch(paths, args) -> int:
    """Prove the weight-init patch actually changes the probe (and only that).

    Worth running once on a new box: if ``retrain.py`` ever stopped importing
    ``seed_everything`` at call time, the patch would silently no-op and every "seed"
    would produce the identical probe — which is exactly the vacuous control this
    design exists to avoid.
    """
    import agentic_redteam.evaluation as _ev
    from agentic_redteam.retrain import retrain_probe
    from agentic_redteam.evaluation import evaluate_probe

    real = _ev.seed_everything
    P = paths["pooled"]
    meta = [json.loads(l) for l in (P / "redteam_meta.jsonl").read_text().splitlines()]
    tmp = paths["work"] / "seedcheck"
    tmp.mkdir(parents=True, exist_ok=True)
    jsonl = tmp / "rt.jsonl"
    jsonl.write_text("".join(json.dumps(_to_record(m), ensure_ascii=False) + "\n" for m in meta))

    scores = {}
    for seed in (42, 101):
        probe = tmp / f"p_{seed}.pkl"
        _ev.seed_everything = lambda _s, _S=seed: real(_S)
        try:
            retrain_probe(jsonl_path=jsonl,
                          base_probe_path=paths["probe_dir"] / "probe_iter0.pkl",
                          base_training_data_path=paths["base_data"], new_probe_path=probe,
                          probe_spec=args.archs[0], preprocessing=None,
                          min_judge_confidence=0, test_size=args.test_size,
                          split_field=None, seed=args.split_seed,
                          base_activation_cache_dir=paths["base_cache"],
                          combine_consecutive_messages=args.combine,
                          convert_tool_to_assistant=args.convert, verbose=False)
        finally:
            _ev.seed_everything = real
        df = evaluate_probe(probe_path=probe, eval_dataset_dir=paths["eval_dir"],
                            activations_cache_dir=paths["eval_cache"], max_samples=None,
                            seed=args.split_seed, combine_consecutive_messages=args.combine,
                            convert_tool_to_assistant=args.convert)
        scores[seed] = float(df[df.dataset == "mean"].iloc[0].auroc)
        probe.unlink(missing_ok=True)
        print(f"  init seed {seed}: AUROC={scores[seed]:.6f}")
    jsonl.unlink(missing_ok=True)
    if scores[42] == scores[101]:
        print("\nFAIL: both seeds gave the identical probe — the patch is a no-op, so a "
              "'seed' axis would be vacuous. Check that retrain.py still imports "
              "seed_everything at call time.")
        return 1
    print("\nOK: the seeds produce different probes, so the seed axis is real.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path, default=REPO_ROOT / "results_arch_cluster_ablation")
    ap.add_argument("--probe-dir", type=Path,
                    default=REPO_ROOT / "probes/hu_harm_gemma27b_deepseekv4pro_batch")
    ap.add_argument("--base-training-data", type=Path,
                    default=REPO_ROOT / "data/hu_harm_llama70b_50.jsonl")
    ap.add_argument("--eval-dataset-dir", type=Path, default=REPO_ROOT / "eval_dataset_hu_ha")
    ap.add_argument("--base-activation-cache-dir", type=Path, default=None,
                    help="default <work-dir>/base_activations")
    ap.add_argument("--eval-activation-cache-dir", type=Path, default=None,
                    help="default <work-dir>/eval_activations")
    ap.add_argument("--arm", default="deepseekv4pro", help="attacker arm on Kaggle")
    ap.add_argument("--iteration", type=int, default=3)

    ap.add_argument("--archs", nargs="*", default=ALL_ARCHS,
                    help=f"probe architectures to test (default: all {len(ALL_ARCHS)})")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 101, 202, 303, 404],
                    help="weight-init seeds")
    ap.add_argument("--draws", type=int, default=10, help="random draws from the near pool")
    ap.add_argument("--target-frac", type=float, default=0.10,
                    help="fraction of the red-team set each arm removes")
    ap.add_argument("--with-random-arm", action="store_true",
                    help="also remove --draws random sets from anywhere (control)")
    ap.add_argument("--near-pool-mult", type=float, default=3.0,
                    help="grow the near pool to this multiple of the removal budget. Nothing "
                         "is matched, so the pool is sized only to keep the draws distinct: "
                         "3x gives ~33%% expected pairwise overlap while keeping the pool to "
                         "the nearest few clusters.")
    ap.add_argument("--class-balanced-removal", action="store_true",
                    help="take the farthest target/2 of EACH class instead of the farthest "
                         "clusters wholesale, so every arm leaves the training class "
                         "balance untouched. Also makes the near draws far less "
                         "overlapping when the far region is class-skewed.")

    ap.add_argument("--fixed-near-arm", action="store_true",
                    help="also run the deterministic near_s{S} arm (the mirror of far). "
                         "Off by default: the sampled neardraw arm supersedes it.")
    ap.add_argument("--neardraw-seed-cross", action="store_true",
                    help="also re-run near draw 0 at every other seed, crossing the seed "
                         "and draw axes. Off by default; costs one job per extra seed.")
    ap.add_argument("--intact-pairs-only", action="store_true",
                    help="keep only rows whose contrastive partner is also in the corpus, "
                         "so every arm removes whole pairs rather than orphaning halves. "
                         "Needs <probe-dir>/contrastive_cache.jsonl.")
    ap.add_argument("--exclude-truncated", action="store_true",
                    help="drop rows the 1024-token activation cap cut off (n_tokens >= "
                         "1024) before clustering, and from every run arm including full")
    ap.add_argument("--k-min", type=int, default=4)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--ensemble-runs", type=int, default=45)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--cluster-seed", type=int, default=7)
    ap.add_argument("--draw-seed", type=int, default=2024)
    ap.add_argument("--recluster", action="store_true")

    ap.add_argument("--split-seed", type=int, default=42,
                    help="retrain_probe's own seed (train/val split + cache keys). "
                         "Changing it invalidates the downloaded activation cache.")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-combine-consecutive-messages", dest="combine",
                    action="store_false", default=True)
    ap.add_argument("--no-convert-tool-to-assistant", dest="convert",
                    action="store_false", default=True)

    ap.add_argument("--stages", nargs="*", default=None, choices=STAGES,
                    help=f"default: {' '.join(STAGES[1:])} (fetch only with --fetch-kaggle)")
    ap.add_argument("--fetch-kaggle", action="store_true",
                    help="download the activation caches from Kaggle first")
    ap.add_argument("--check-seed-patch", action="store_true",
                    help="verify the weight-init patch is not a no-op, then exit")
    ap.add_argument("--dry-run", action="store_true", help="list the run matrix, run nothing")
    args = ap.parse_args(argv)

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    paths = {
        "work": work,
        "pooled": work / "pooled",
        "probe_dir": args.probe_dir.resolve(),
        "base_data": args.base_training_data.resolve(),
        "eval_dir": args.eval_dataset_dir.resolve(),
        "base_cache": (args.base_activation_cache_dir or work / "base_activations").resolve(),
        "eval_cache": (args.eval_activation_cache_dir or work / "eval_activations").resolve(),
    }
    paths["base_cache"].mkdir(parents=True, exist_ok=True)
    paths["eval_cache"].mkdir(parents=True, exist_ok=True)

    bad = [a for a in args.archs if a not in ALL_ARCHS]
    if bad:
        raise SystemExit(f"unknown architecture(s) {bad}; choose from {ALL_ARCHS}")

    try:
        import torch
        print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}"
              + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
        from tuberlens.config import global_settings as g
        print(f"tuberlens DEVICE={g.DEVICE} DTYPE={g.DTYPE}  "
              f"<- all arms must run on the SAME device: DTYPE differs by device, so "
              f"CPU and GPU results are not comparable")
    except Exception as e:
        print(f"(could not report device: {e})")

    stages = args.stages or [s for s in STAGES if s != "fetch"]
    if args.fetch_kaggle and "fetch" not in stages:
        stages = ["fetch"] + list(stages)

    if "fetch" in stages:
        stage_fetch(args, paths)
    if "pool" in stages:
        stage_pool(args, paths)
    if args.check_seed_patch:
        return _check_seed_patch(paths, args)
    if "cluster" in stages:
        stage_cluster(args, paths)
    if "run" in stages:
        stage_run(args, paths)
    if "analyze" in stages and not args.dry_run:
        stage_analyze(args, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
