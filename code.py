import os, zipfile, shutil, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print("TensorFlow version:", tf._version_)
WINDOW_SIZE     = 1024
STRIDE          = 128   
BATCH_SIZE      = 32
EPOCHS          = 40
WARMUP_EPOCHS   = 5
LEARNING_RATE   = 1e-3
CLIP_NORM       = 1.0
PATIENCE        = 7
ECONS_SUBSAMPLE = 32
LAMBDA_VALUES   = [0.0, 0.1, 0.5, 1.0]

zip_path     = "/content/archive.zip"
extract_path = "/content/eeg_v4"
if os.path.exists(extract_path): shutil.rmtree(extract_path)
os.makedirs(extract_path)
with zipfile.ZipFile(zip_path) as z: z.extractall(extract_path)

root = Path(extract_path)
cands = []
for p in root.rglob("*"):
    if p.is_dir():
        subs = [x for x in p.iterdir() if x.is_dir()]
        tc   = sum(len(list(sd.glob("*.txt"))) for sd in subs)
        if len(subs) >= 2 and tc > 0:
            cands.append((p, tc))
cands.sort(key=lambda x: x[1], reverse=True)
dataset_root = cands[0][0]
print("Dataset root:", dataset_root)

positive_folders = {"S", "s"}
X_raw, y_raw = [], []
print("\nClass folders:")
for d in sorted(dataset_root.iterdir()):
    if not d.is_dir(): continue
    files = sorted(d.glob("*.txt"))
    print(f"  {d.name}: {len(files)} files")
    if not files: continue
    lbl = 1 if d.name in positive_folders else 0
    for fp in files:
        X_raw.append(np.ravel(np.loadtxt(fp, dtype=np.float32)))
        y_raw.append(lbl)

X_raw = np.array(X_raw, dtype=object)
y_raw = np.array(y_raw, dtype=np.int32)
print(f"\nTotal: {len(X_raw)} signals  (S={np.sum(y_raw==1)}, nonS={np.sum(y_raw==0)})")

idx = np.arange(len(X_raw))
tr_idx, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y_raw)
va_idx, te_idx = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=y_raw[tmp])

def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-8)

def make_sets(ids):
    return [zscore(np.asarray(X_raw[i], np.float32)) for i in ids], y_raw[ids]

X_tr, y_tr = make_sets(tr_idx)
X_va, y_va = make_sets(va_idx)
X_te, y_te = make_sets(te_idx)

def create_window_pairs(signals, labels, ws=1024, st=128):
    xt_l, xt1_l, yt_l = [], [], []
    for sig, lbl in zip(signals, labels):
        sig = np.asarray(sig, np.float32)
        if len(sig) < ws:
            sig = np.pad(sig, (0, ws - len(sig)))
        wins = [sig[s:s+ws] for s in range(0, len(sig)-ws+1, st)]
        if len(wins) < 2: continue
        wins = np.array(wins, np.float32)
        for i in range(len(wins) - 1):
            xt_l.append(wins[i])
            xt1_l.append(wins[i+1])
            yt_l.append(lbl)
    return (np.array(xt_l,  np.float32)[..., None],
            np.array(xt1_l, np.float32)[..., None],
            np.array(yt_l,  np.float32))

xtr_t,  xtr_t1,  ytr_t  = create_window_pairs(X_tr, y_tr, WINDOW_SIZE, STRIDE)
xva_t,  xva_t1,  yva_t  = create_window_pairs(X_va, y_va, WINDOW_SIZE, STRIDE)
xte_t,  xte_t1,  yte_t  = create_window_pairs(X_te, y_te, WINDOW_SIZE, STRIDE)

print(f"\nWindow pairs (stride={STRIDE}, overlap={100*(1-STRIDE/WINDOW_SIZE):.0f}%):")
print(f"  train={xtr_t.shape}, val={xva_t.shape}, test={xte_t.shape}")
print(f"  train seizure pairs: {int(np.sum(ytr_t==1))}, nonseizure: {int(np.sum(ytr_t==0))}")

def res_block(x, f, k, drop=0.15):
    h  = layers.Conv1D(f, k, padding="same")(x)
    h  = layers.BatchNormalization()(h)
    h  = layers.Activation("elu")(h)
    h  = layers.Conv1D(f, k, padding="same")(h)
    h  = layers.BatchNormalization()(h)
    sc = layers.Conv1D(f, 1, padding="same")(x) if x.shape[-1] != f else x
    if x.shape[-1] != f:
        sc = layers.BatchNormalization()(sc)
    h  = layers.Add()([h, sc])
    h  = layers.Activation("elu")(h)
    h  = layers.MaxPooling1D(2)(h)
    h  = layers.Dropout(drop)(h)
    return h

def build_model(input_shape=(1024, 1)):
    """
    Returns two models:
    - full_model: input → sigmoid (for classification)
    - repr_model: input → z (64-dim vector from GAP, before the final Dense layer)
    Both share the same parameters.
    """
    inp = layers.Input(shape=input_shape, name="eeg_input")
    x   = layers.Conv1D(16, 11, padding="same", name="stem_conv")(inp)
    x   = layers.BatchNormalization(name="stem_bn")(x)
    x   = layers.Activation("elu", name="stem_act")(x)
    x   = layers.MaxPooling1D(2, name="stem_pool")(x)
    x   = res_block(x, 16, 7, 0.10)
    x   = res_block(x, 32, 5, 0.15)
    x   = res_block(x, 64, 3, 0.20)
    z   = layers.GlobalAveragePooling1D(name="gap")(x)          # ← representation z
    h   = layers.Dense(64, activation="elu", name="dense1")(z)
    h   = layers.Dropout(0.25, name="drop_head")(h)
    out = layers.Dense(1, activation="sigmoid", name="output")(h)

    full_model = models.Model(inp, out,  name="full_model")
    repr_model = models.Model(inp, z,    name="repr_model")  
    return full_model, repr_model

# ══════════════════════════════════════════════
# HYBRID E_cons
# Combines:
# (A) cosine similarity on representations z (semantic)
# (B) cosine similarity on GxI relative to input (classic explainability)
# Formula: E_cons = α * cos(z_t, z_t1) + (1-α) * cos(GxI_t, GxI_t1)
# with α=0.6 → representations dominate, GxI contributes
# ══════════════════════════════════════════════
ALPHA_REPR = 0.6   # weight of representations vs input gradients

def l2_norm(v, eps=1e-8):
    """L2-normalization over the last dimension."""
    flat = tf.reshape(v, [tf.shape(v)[0], -1])
    return flat / (tf.norm(flat, axis=1, keepdims=True) + eps)

def compute_econs_hybrid(full_model, repr_model, xt, xt1):
    """
    Hybrid E_cons on a mini-batch of pairs (xt, xt+1).

    Component A – intermediate representations z:
      cos(z_t, z_{t+1}) where z = GAP layer output (64-dim)
      This is similar to a contrastive/self-supervised loss.

    Component B – Gradient × Input relative to output:
      cos(GxI_t, GxI_{t+1}) L2-normalized
      This is the classic saliency map from the paper.

    Returns: econs [B], comp_repr [B], comp_gxi [B]
    """
    # ── Component A: representations ──
    z_t  = repr_model(xt,  training=True)   # [B, 64]
    z_t1 = repr_model(xt1, training=True)   # [B, 64]
    z_t_n  = l2_norm(z_t)
    z_t1_n = l2_norm(z_t1)
    cos_repr = tf.reduce_sum(z_t_n * z_t1_n, axis=1)  # [B]

    # ── Component B: Gradient × Input ──
    with tf.GradientTape() as tape:
        tape.watch(xt)
        score_t = tf.squeeze(full_model(xt, training=True), 1)
    g_t = tape.gradient(score_t, xt)
    gxi_t = g_t * xt  # [B, T, 1]

    with tf.GradientTape() as tape:
        tape.watch(xt1)
        score_t1 = tf.squeeze(full_model(xt1, training=True), 1)
    g_t1 = tape.gradient(score_t1, xt1)
    gxi_t1 = g_t1 * xt1  # [B, T, 1]

    gxi_t_n  = l2_norm(gxi_t)
    gxi_t1_n = l2_norm(gxi_t1)
    cos_gxi  = tf.reduce_sum(gxi_t_n * gxi_t1_n, axis=1)  # [B]

    econs = ALPHA_REPR * cos_repr + (1.0 - ALPHA_REPR) * cos_gxi
    return econs, cos_repr, cos_gxi

def best_threshold(probs, labels):
    best_t, best_ba = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        pred = (probs >= t).astype(int)
        if len(np.unique(pred)) < 2: continue
        cm_ = confusion_matrix(labels, pred)
        if cm_.shape != (2, 2): continue
        tn_, fp_, fn_, tp_ = cm_.ravel()
        ba = 0.5 * (tp_/(tp_+fn_+1e-8) + tn_/(tn_+fp_+1e-8))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t

def run_experiment(lam, xtr_t, xtr_t1, ytr_t,
                          xva_t, xva_t1, yva_t,
                          xte_t, xte_t1, yte_t):

    print(f"\n{'═'*60}")
    print(f"  EXPERIMENT  λ = {lam}")
    print(f"{'═'*60}")

    full_model, repr_model = build_model((WINDOW_SIZE, 1))

    cw_arr = compute_class_weight("balanced",
                                  classes=np.unique(ytr_t.astype(int)),
                                  y=ytr_t.astype(int))
    cw = {0: float(cw_arr[0]), 1: float(cw_arr[1])}
    print(f"  Class weights: {cw}")

    total_steps = int(np.ceil(len(xtr_t) / BATCH_SIZE)) * EPOCHS
    lr_sched = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=total_steps,
        alpha=1e-5 / LEARNING_RATE
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=CLIP_NORM)
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    train_ds = (tf.data.Dataset
                .from_tensor_slices((xtr_t, xtr_t1, ytr_t))
                .shuffle(len(xtr_t), seed=SEED)
                .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset
              .from_tensor_slices((xva_t, xva_t1, yva_t))
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    test_ds = (tf.data.Dataset
               .from_tensor_slices((xte_t, xte_t1, yte_t))
               .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    def current_lambda(epoch):
        if lam == 0.0: return 0.0
        return lam * min(1.0, epoch / max(WARMUP_EPOCHS, 1))

    def train_step(xt, xt1, yt, lam_eff):
        yt = tf.cast(tf.reshape(yt, [-1, 1]), tf.float32)
        sw = tf.cast(
            tf.where(tf.equal(tf.cast(yt[:,0], tf.int32), 1), cw[1], cw[0]),
            tf.float32
        )
        with tf.GradientTape() as tape:
            pt   = full_model(xt, training=True)
            cls  = tf.reduce_mean(bce(yt, pt) * sw)

            if lam_eff > 0.0:
                k = min(ECONS_SUBSAMPLE, tf.shape(xt)[0].numpy())
                ec, c_repr, c_gxi = compute_econs_hybrid(
                    full_model, repr_model, xt[:k], xt1[:k]
                )
                ec_mean  = tf.reduce_mean(ec)
                cons     = 1.0 - ec_mean
                total    = cls + lam_eff * cons
            else:
                ec_mean = tf.constant(0.0)
                cons    = tf.constant(0.0)
                c_repr  = tf.constant([0.0])
                c_gxi   = tf.constant([0.0])
                total   = cls

        grads = tape.gradient(total, full_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, full_model.trainable_variables))
        return float(cls), float(total), float(ec_mean)

    def val_step(xt, xt1, yt, lam_eff):
        yt = tf.cast(tf.reshape(yt, [-1, 1]), tf.float32)
        pt = full_model(xt, training=False)
        cls = tf.reduce_mean(bce(yt, pt))

        if lam_eff > 0.0:
            k = min(ECONS_SUBSAMPLE, tf.shape(xt)[0].numpy())
            ec, _, _ = compute_econs_hybrid(
                full_model, repr_model, xt[:k], xt1[:k]
            )
            ec_mean = tf.reduce_mean(ec)
            total   = cls + lam_eff * (1.0 - ec_mean)
        else:
            ec_mean = tf.constant(0.0)
            total   = cls

        return float(cls), float(total), float(ec_mean), pt

    hist = {"tr_cls":[], "tr_ec":[], "tr_tot":[],
            "va_cls":[], "va_ec":[], "va_tot":[], "va_auc":[], "lam_eff":[]}

    best_score, best_weights, wait = -np.inf, None, 0
    val_probs_last, val_true_last  = [], []

    for epoch in range(1, EPOCHS + 1):
        lam_eff = current_lambda(epoch)

        tc_l, te_l, tt_l = [], [], []
        for xt, xt1, yt in train_ds:
            c, tot, em = train_step(xt, xt1, yt, lam_eff)
            tc_l.append(c); te_l.append(em); tt_l.append(tot)

        vc_l, ve_l, vt_l, vp_l, vtr_l = [], [], [], [], []
        for xt, xt1, yt in val_ds:
            c, tot, em, pt = val_step(xt, xt1, yt, lam_eff)
            vc_l.append(c); ve_l.append(em); vt_l.append(tot)
            vp_l.extend(tf.squeeze(pt, 1).numpy().tolist())
            vtr_l.extend(yt.numpy().tolist())

        va  = roc_auc_score(np.array(vtr_l), np.array(vp_l))
        vem = np.mean(ve_l)
        score = va + 0.02 * vem

        hist["tr_cls"].append(np.mean(tc_l))
        hist["tr_ec"].append(np.mean(te_l))
        hist["tr_tot"].append(np.mean(tt_l))
        hist["va_cls"].append(np.mean(vc_l))
        hist["va_ec"].append(vem)
        hist["va_tot"].append(np.mean(vt_l))
        hist["va_auc"].append(va)
        hist["lam_eff"].append(lam_eff)

        print(f"  Ep {epoch:02d}/{EPOCHS} | λ={lam_eff:.2f} | "
              f"tr_loss={hist['tr_tot'][-1]:.4f} | tr_Ec={hist['tr_ec'][-1]:.4f} | "
              f"va_loss={hist['va_tot'][-1]:.4f} | va_Ec={vem:.4f} | va_AUC={va:.4f}")

        val_probs_last = vp_l
        val_true_last  = vtr_l

        if score > best_score:
            best_score   = score
            best_weights = full_model.get_weights()
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("  Early stopping.")
                break

    full_model.set_weights(best_weights)
    thresh = best_threshold(np.array(val_probs_last), np.array(val_true_last))
    print(f"  Optimal threshold: {thresh:.2f}")
    tp_all, tt_all = [], []
    ec_all, ec_s, ec_ns = [], [], []
    cr_s, cr_ns   = [], []   
    cg_s, cg_ns   = [], []   

    for xt, xt1, yt in test_ds:
        pt = tf.squeeze(full_model(xt, training=False), 1).numpy()
        tp_all.extend(pt.tolist())
        tt_all.extend(yt.numpy().tolist())

        k = min(ECONS_SUBSAMPLE, xt.shape[0])
        ec_v, cr_v, cg_v = compute_econs_hybrid(
            full_model, repr_model, xt[:k], xt1[:k]
        )
        for e, cr, cg, l in zip(ec_v.numpy(), cr_v.numpy(), cg_v.numpy(), yt.numpy()[:k]):
            ec_all.append(float(e))
            if l == 1:
                ec_s.append(float(e)); cr_s.append(float(cr)); cg_s.append(float(cg))
            else:
                ec_ns.append(float(e)); cr_ns.append(float(cr)); cg_ns.append(float(cg))

    tp_all = np.array(tp_all)
    tt_all = np.array(tt_all)
    pred   = (tp_all >= thresh).astype(int)

    acc  = accuracy_score(tt_all, pred)
    cm_  = confusion_matrix(tt_all, pred)
    rauc = roc_auc_score(tt_all, tp_all)
    tn_, fp_, fn_, tp_ = cm_.ravel()
    sens = tp_ / (tp_ + fn_ + 1e-8)
    spec = tn_ / (tn_ + fp_ + 1e-8)

    me_all = float(np.mean(ec_all)) if ec_all else 0.0
    me_s   = float(np.mean(ec_s))   if ec_s   else 0.0
    me_ns  = float(np.mean(ec_ns))  if ec_ns  else 0.0
    mr_s   = float(np.mean(cr_s))   if cr_s   else 0.0
    mg_s   = float(np.mean(cg_s))   if cg_s   else 0.0

    print(f"\n  ── TEST (λ={lam}, thresh={thresh:.2f}) ──")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  Sensitivity : {sens*100:.2f}%")
    print(f"  Specificity : {spec*100:.2f}%")
    print(f"  AUC         : {rauc*100:.2f}%")
    print(f"  E_cons(all) : {me_all:.4f}")
    print(f"  E_cons(S)   : {me_s:.4f}   [cos_repr={mr_s:.4f}, cos_gxi={mg_s:.4f}]")
    print(f"  E_cons(nonS): {me_ns:.4f}")
    print(f"  Confusion:\n{cm_}")
    fpr_, tpr_, _ = roc_curve(tt_all, tp_all)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(fpr_, tpr_, lw=2, label=f"AUC={rauc:.3f}")
    axs[0].plot([0,1],[0,1],"--", lw=1)
    axs[0].set_title(f"ROC (λ={lam})"); axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR")
    axs[0].legend(loc="lower right")
    sns.heatmap(cm_, annot=True, fmt="d", cmap="viridis", cbar=False,
                xticklabels=["nonS","S"], yticklabels=["nonS","S"], ax=axs[1])
    axs[1].set_title(f"Confusion Matrix (λ={lam})")
    axs[1].set_xlabel("Predicted"); axs[1].set_ylabel("True")
    plt.tight_layout()
    fname = f"roc_cm_v4_lam{str(lam).replace('.','')}.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight"); plt.show()
    print(f"  Saved: {fname}")

    return {
        "lambda":      lam,
        "accuracy":    round(acc,  4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "auc":         round(rauc, 4),
        "econs_all":   round(me_all, 4),
        "econs_S":     round(me_s,   4),
        "econs_nonS":  round(me_ns,  4),
        "history":     hist,
        "cm":          cm_.tolist(),
    }

print("\n" + "═"*60)
print("  VERIFICATION: Hybrid E_cons on a few pairs (untrained model)")
print("═"*60)
fm_test, rm_test = build_model((WINDOW_SIZE, 1))
test_pairs = xtr_t[:16], xtr_t1[:16]
ec_pre, cr_pre, cg_pre = compute_econs_hybrid(fm_test, rm_test,
                                               tf.constant(test_pairs[0]),
                                               tf.constant(test_pairs[1]))
print(f"  Hybrid E_cons (untrained): mean={ec_pre.numpy().mean():.4f}, "
      f"cos_repr={cr_pre.numpy().mean():.4f}, cos_gxi={cg_pre.numpy().mean():.4f}")
print("  → cos_repr > 0 even untrained if windows are semantically similar")

all_results = []
for lam in LAMBDA_VALUES:
    res = run_experiment(lam,
                         xtr_t, xtr_t1, ytr_t,
                         xva_t, xva_t1, yva_t,
                         xte_t, xte_t1, yte_t)
    all_results.append(res)

sep = "═" * 72
print(f"\n{sep}")
print("  FINAL TABLE – Hybrid E_cons (repr + GxI)")
print(sep)
print(f"  {'λ':>5} | {'Acc':>7} | {'Sens':>7} | {'Spec':>7} | {'AUC':>7} | "
      f"{'E_cons':>8} | {'E_cons(S)':>9} | {'E_cons(nS)':>10}")
print("  " + "─"*68)
for r in all_results:
    print(f"  {r['lambda']:>5} | {r['accuracy']:>7.3f} | {r['sensitivity']:>7.3f} | "
          f"{r['specificity']:>7.3f} | {r['auc']:>7.3f} | {r['econs_all']:>8.4f} | "
          f"{r['econs_S']:>9.4f} | {r['econs_nonS']:>10.4f}")
print(f"{sep}\n")

df = pd.DataFrame([{k: v for k,v in r.items() if k not in ("history","cm")}
                   for r in all_results])
df.to_csv("eeg_results_v4.csv", index=False)
print("Saved: eeg_results_v4.csv")

fig, axes = plt.subplots(3, len(LAMBDA_VALUES), figsize=(5*len(LAMBDA_VALUES), 10))
for col, r in enumerate(all_results):
    h = r["history"]; ep = range(1, len(h["tr_tot"])+1); lm = r["lambda"]
    axes[0,col].plot(ep, h["tr_tot"], label="train")
    axes[0,col].plot(ep, h["va_tot"], label="val")
    axes[0,col].set_title(f"Total Loss (λ={lm})"); axes[0,col].legend()
    axes[1,col].plot(ep, h["tr_ec"], label="train")
    axes[1,col].plot(ep, h["va_ec"], label="val")
    axes[1,col].set_title(f"E_cons (λ={lm})"); axes[1,col].legend()
    axes[2,col].plot(ep, h["va_auc"])
    axes[2,col].set_title(f"Val AUC (λ={lm})")
    axes[2,col].set_ylim(0.85, 1.01)
plt.tight_layout()
plt.savefig("training_curves_v4.png", dpi=200, bbox_inches="tight"); plt.show()
print("Saved: training_curves_v4.png")

lams  = [r["lambda"]   for r in all_results]
accs  = [r["accuracy"] for r in all_results]
aucs  = [r["auc"]      for r in all_results]
ecs   = [r["econs_all"]for r in all_results]
ecs_s = [r["econs_S"]  for r in all_results]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax1.plot(lams, accs,  "o-",  color="steelblue",  label="Accuracy", lw=2)
ax1.plot(lams, aucs,  "s--", color="dodgerblue", label="AUC",      lw=2)
ax2.plot(lams, ecs,   "^-",  color="tomato",     label="E_cons",   lw=2)
ax2.plot(lams, ecs_s, "v--", color="orangered",  label="E_cons(S)",lw=2)
ax1.set_xlabel("λ"); ax1.set_ylabel("Accuracy / AUC"); ax2.set_ylabel("E_cons")
ax1.set_title("Trade-off: Classification performance vs Hybrid E_cons")
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc="lower left")
plt.tight_layout()
plt.savefig("tradeoff_v4.png", dpi=200, bbox_inches="tight"); plt.show()
print("Saved: tradeoff_v4.png")
