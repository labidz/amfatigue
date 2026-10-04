"""Experiment configurations. 'fast' is what runs in this workspace; 'full' is for the paper."""
CONFIGS = {
    "debug": dict(n_train=60, n_cal=40, n_test=30, n_anom=30, seeds=[0], steps=30, batch=0, gh_train=3, gh_pred=5,
                  K=6, ens=2, ks=[0, 3, 5], pac_draws=20, conf_reps=5, aft_steps=50, cox_steps=30, real_steps=30,
                  real_folds=3),
    "fast": dict(n_train=300, n_cal=200, n_test=200, n_anom=200, seeds=[0, 1, 2], steps=400, batch=150,
                 gh_train=5, gh_pred=13, K=12, ens=3, ks=[0, 1, 2, 3, 5, 8], pac_draws=200, conf_reps=50,
                 aft_steps=1500, cox_steps=800, real_steps=3000, real_folds=5),
    "full": dict(n_train=600, n_cal=450, n_test=450, n_anom=450, seeds=list(range(10)), steps=2000, batch=200,
                 gh_train=9, gh_pred=25, K=24, ens=5, ks=[0, 1, 2, 3, 5, 8, 12], pac_draws=1000, conf_reps=200,
                 aft_steps=4000, cox_steps=2000, real_steps=6000, real_folds=10),
}
ALPHA = 0.10  # B-basis content 90 %
DELTA = 0.05  # B-basis confidence 95 %
ND = 2e6  # design life (FKM / Eurocode reference)
C0 = 1e7  # censoring cap for life bounds
C0_FORENSIC = 1e6
N_FS = 1e7  # life at which fatigue strength is compared (E1)
