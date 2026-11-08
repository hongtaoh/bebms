#!/usr/bin/env python3
"""
Longitudinal validation of a model's ADNI stages against clinical follow-up.

The question a reviewer asked: is there any quantitative evidence, on real data,
that an inferred disease stage means anything? Here is the test, which is the
same design SuStaIn used (Young et al., Nature Communications 2018, Table 3):

    Fit the model to BASELINE cross-sectional data only. Then ask whether the
    baseline stage predicts how soon a participant is later diagnosed with AD.

No ordering and no stage is ever derived from longitudinal data. The follow-up
visits supply only the outcome -- who converted, and when. If the stage axis is
ordered correctly, participants placed later at baseline should convert sooner.

This copy lives in the bebms repo so the BEBMS and SuStaIn assignments can be
checked without reaching into another repository. It is identical to
subtrace_iclr_clean/adni_survival_validation.py.

Needs only ADNIMERGE.csv and a per-participant assignments CSV with columns:
    experiment, participant, predicted_subtype, predicted_stage_continuous
where `participant` is the 0-based row index of adni.csv. It does NOT need
adni.csv, id_dx.json or dx_array.npy.

Usage, from this directory:

    python adni_survival_validation.py --results notebooks/baseline_assignments.csv
    python adni_survival_validation.py --results ... --stage-col predicted_stage_direct
    python adni_survival_validation.py --results ... --include-cn

Outputs, written to --out (default results/survival/):
    survival_table.csv          one row per participant: stage, subtype, time, event
    cox_per_experiment.csv      Cox hazard ratios per method
    cox_subtype_pairwise.csv    pairwise subtype comparisons
    rank_checks.csv             distribution-free checks (no lifelines needed)
    summary.txt                 human-readable summary

Cox regression needs `lifelines`. Without it the script still runs and reports
the distribution-free checks, which need only scipy.
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# These must match process_adni.py exactly, or the participant indices in
# results/adni/all_patient_subtype_stage.csv will not line up.
META = ["PTID", "DX_bl", "VISCODE", "COLPROT"]
BIOMARKERS = [
    "MMSE_bl", "Ventricles_bl", "WholeBrain_bl", "MidTemp_bl", "Fusiform_bl",
    "Entorhinal_bl", "Hippocampus_bl", "ADAS13_bl", "PTAU_bl", "TAU_bl",
    "ABETA_bl", "RAVLT_immediate_bl", "ICV_bl",
]
DIAGNOSES = ["CN", "EMCI", "LMCI", "AD"]

AD_LABELS = {"Dementia", "AD"}          # how ADNIMERGE spells an AD diagnosis
COVARIATES = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4"]


# ----------------------------------------------------------------------------
# 1. Recover which real ADNI participant each index refers to
# ----------------------------------------------------------------------------
def recover_ptid_map(raw_path):
    """
    process_adni.py overwrites the real PTID with range(len(df)), so the saved
    results identify people only by position. Replaying the identical filter
    here reproduces that ordering and recovers the mapping.
    """
    df = pd.read_csv(raw_path, low_memory=False)[META + BIOMARKERS]
    df = df[df["VISCODE"] == "bl"]
    df = df[df["DX_bl"].isin(DIAGNOSES)]
    for col in BIOMARKERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=BIOMARKERS).reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)

    return pd.DataFrame({
        "participant": np.arange(len(df)),
        "real_ptid": df["PTID"].values,
        "dx_bl": df["DX_bl"].values,
    })


# ----------------------------------------------------------------------------
# 2. Who converted to AD, and when
# ----------------------------------------------------------------------------
def build_survival(raw_path, ptid_map):
    """One row per participant: time to AD diagnosis, or censoring time."""
    cols = ["PTID", "VISCODE", "DX", "Years_bl", "M"] + COVARIATES
    full = pd.read_csv(raw_path, low_memory=False)
    have = [c for c in cols if c in full.columns]
    full = full[have].copy()

    if "Years_bl" in full.columns:
        full["t"] = pd.to_numeric(full["Years_bl"], errors="coerce")
    elif "M" in full.columns:
        full["t"] = pd.to_numeric(full["M"], errors="coerce") / 12.0
    else:
        raise SystemExit("ADNIMERGE has neither Years_bl nor M; cannot build a time axis.")

    full = full[full["PTID"].isin(set(ptid_map["real_ptid"]))]
    full = full.dropna(subset=["t"])

    seen = sorted(set(full["DX"].dropna().unique()))
    print(f"  DX labels present in follow-up: {seen}")
    ad = AD_LABELS & set(seen)
    if not ad:
        raise SystemExit(f"No AD-like label among {seen}; edit AD_LABELS at the top.")
    print(f"  treating {sorted(ad)} as AD")

    base = full[full["VISCODE"] == "bl"].drop_duplicates("PTID").set_index("PTID")

    rows = []
    for ptid, g in full.groupby("PTID"):
        g = g.sort_values("t")
        conv = g[g["DX"].isin(ad)]
        if len(conv):
            t_event, event = float(conv["t"].iloc[0]), 1
        else:
            t_event, event = float(g["t"].max()), 0
        rec = {"real_ptid": ptid, "time_years": t_event, "event": event,
               "n_visits": len(g)}
        if ptid in base.index:
            for c in COVARIATES:
                if c in base.columns:
                    rec[c] = base.loc[ptid, c]
        rows.append(rec)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. Distribution-free checks (scipy only)
# ----------------------------------------------------------------------------
def rank_checks(df, stage_col):
    """Does a later baseline stage go with sooner conversion?"""
    from scipy import stats

    out = []
    for exp, g in df.groupby("experiment"):
        conv = g[g["event"] == 1]
        row = {"experiment": exp, "n": len(g), "n_converted": len(conv)}

        # Among people who converted: later stage should mean shorter time.
        if len(conv) >= 5:
            rho, p = stats.spearmanr(conv[stage_col], conv["time_years"])
            row["spearman_stage_vs_time"] = round(float(rho), 4)
            row["spearman_p"] = float(p)

        # Split at the median stage: does the later half convert more often?
        med = g[stage_col].median()
        hi, lo = g[g[stage_col] > med], g[g[stage_col] <= med]
        if len(hi) and len(lo):
            row["conv_rate_above_median_stage"] = round(hi["event"].mean(), 4)
            row["conv_rate_below_median_stage"] = round(lo["event"].mean(), 4)
            table = [[int(hi["event"].sum()), int((1 - hi["event"]).sum())],
                     [int(lo["event"].sum()), int((1 - lo["event"]).sum())]]
            try:
                _, p = stats.fisher_exact(table)
                row["fisher_p"] = float(p)
            except Exception:
                pass
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# 4. Cox proportional hazards, as in SuStaIn Table 3
# ----------------------------------------------------------------------------
def cox_per_experiment(df, stage_col, use_covariates=True):
    """
    Cox PH per experiment, following SuStaIn Table 3.

    Subtype labels are arbitrary and unordered, so subtype must NOT enter the
    model as a numeric column: fitting a single coefficient to labels 0,1,2
    assumes the risk ordering happens to match the label ordering and will hide
    a real effect otherwise. We dummy-code instead, and report a likelihood
    ratio test of "subtype adds nothing beyond stage" (the global question),
    alongside the per-subtype hazard ratios relative to the reference subtype.
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test  # noqa: F401
    except ImportError:
        return None, None

    from scipy import stats as sps

    per_exp, pairwise = [], []
    for exp, g in df.groupby("experiment"):
        base = pd.DataFrame({
            "T": g["time_years"].values,
            "E": g["event"].values,
            "stage": g[stage_col].values,
        })
        subtype = pd.Series(g["predicted_subtype"].values, name="subtype").astype(int)
        if use_covariates:
            for c in COVARIATES:
                if c in g.columns and g[c].notna().mean() > 0.8:
                    v = g[c]
                    if v.dtype == object:
                        v = (v.astype(str).str.strip().str.lower() == "male").astype(float)
                    base[c] = pd.to_numeric(v, errors="coerce").values

        ok = base.notna().all(axis=1) & (base["T"] > 0)
        base, subtype = base[ok].reset_index(drop=True), subtype[ok.values].reset_index(drop=True)
        if base["E"].sum() < 10 or len(base) < 30:
            per_exp.append({"experiment": exp, "note": "too few events",
                            "n": len(base), "n_events": int(base["E"].sum())})
            continue

        rec = {"experiment": exp, "n": len(base), "n_events": int(base["E"].sum())}
        try:
            # ---- stage-only model (the headline) ----
            m_stage = CoxPHFitter().fit(base, duration_col="T", event_col="E")
            s = m_stage.summary
            rec.update({
                "stage_HR": round(float(s.loc["stage", "exp(coef)"]), 4),
                "stage_p": float(s.loc["stage", "p"]),
                "stage_CI_low": round(float(s.loc["stage", "exp(coef) lower 95%"]), 4),
                "stage_CI_high": round(float(s.loc["stage", "exp(coef) upper 95%"]), 4),
                "concordance": round(float(m_stage.concordance_index_), 4),
            })

            # ---- does subtype add anything beyond stage? (dummy-coded) ----
            levels = sorted(subtype.unique())
            if len(levels) > 1:
                ref = levels[0]
                d = base.copy()
                for lv in levels[1:]:
                    d[f"subtype_{lv}"] = (subtype == lv).astype(float).values
                m_full = CoxPHFitter().fit(d, duration_col="T", event_col="E")
                lr = 2 * (m_full.log_likelihood_ - m_stage.log_likelihood_)
                dof = len(levels) - 1
                rec["subtype_LRT_chi2"] = round(float(lr), 4)
                rec["subtype_LRT_df"] = dof
                rec["subtype_LRT_p"] = float(sps.chi2.sf(max(lr, 0.0), dof))
                rec["subtype_reference"] = int(ref)
                rec["concordance_with_subtype"] = round(float(m_full.concordance_index_), 4)
                fs = m_full.summary
                for lv in levels[1:]:
                    k = f"subtype_{lv}"
                    if k in fs.index:
                        rec[f"HR_subtype{lv}_vs_{ref}"] = round(float(fs.loc[k, "exp(coef)"]), 4)
                        rec[f"p_subtype{lv}_vs_{ref}"] = float(fs.loc[k, "p"])

                # ---- pairwise two-group models, as SuStaIn did ----
                for i in range(len(levels)):
                    for j in range(i + 1, len(levels)):
                        a, b = levels[i], levels[j]
                        mask = subtype.isin([a, b]).values
                        dd = base[mask].copy()
                        dd["is_b"] = (subtype[mask] == b).astype(float).values
                        if dd["E"].sum() < 10 or len(dd) < 25:
                            continue
                        try:
                            mm = CoxPHFitter().fit(dd, duration_col="T", event_col="E")
                            ss = mm.summary
                            pairwise.append({
                                "experiment": exp, "pair": f"{b} vs {a}",
                                "n": len(dd), "n_events": int(dd["E"].sum()),
                                "HR": round(float(ss.loc["is_b", "exp(coef)"]), 4),
                                "p": float(ss.loc["is_b", "p"]),
                                "stage_HR": round(float(ss.loc["stage", "exp(coef)"]), 4),
                            })
                        except Exception:
                            pass
        except Exception as e:
            rec["note"] = f"fit failed: {type(e).__name__}: {e}"
        per_exp.append(rec)

    return pd.DataFrame(per_exp), (pd.DataFrame(pairwise) if pairwise else None)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "ADNIMERGE.csv"))
    ap.add_argument("--results",
                    default=os.path.join(HERE, "notebooks", "baseline_assignments.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "survival"))
    ap.add_argument("--stage-col", default="predicted_stage_continuous",
                    choices=["predicted_stage_continuous", "predicted_stage_direct",
                             "predicted_stage_ordering"])
    ap.add_argument("--include-cn", action="store_true",
                    help="also include baseline-CN participants (few convert)")
    ap.add_argument("--no-covariates", action="store_true")
    args = ap.parse_args()

    for p in (args.raw, args.results):
        if not os.path.isfile(p):
            raise SystemExit(f"missing required file: {p}")
    os.makedirs(args.out, exist_ok=True)

    print("=" * 72)
    print("SUBTRACE ADNI stage vs. clinical conversion to AD")
    print("=" * 72)

    print("\n[1/4] recovering participant index -> real ADNI PTID")
    ptid_map = recover_ptid_map(args.raw)
    print(f"  recovered {len(ptid_map)} baseline participants")
    ptid_map.to_csv(os.path.join(args.out, "ptid_map.csv"), index=False)

    res = pd.read_csv(args.results)
    n_res = res["participant"].nunique()
    if n_res != len(ptid_map):
        raise SystemExit(
            f"MISMATCH: results have {n_res} participants but the replayed filter "
            f"produced {len(ptid_map)}. The preprocessing in process_adni.py must have "
            f"changed since the results were generated; the index->PTID mapping would "
            f"be wrong, so stopping rather than reporting a bogus association."
        )
    print(f"  OK: matches the {n_res} participants in the results file")

    print("\n[2/4] building the survival table from follow-up visits")
    surv = build_survival(args.raw, ptid_map)
    surv = surv.merge(ptid_map, on="real_ptid", how="inner")
    print(f"  {len(surv)} participants with follow-up")
    print(f"  converted to AD: {int(surv['event'].sum())}")
    print(f"  median follow-up: {surv['time_years'].median():.2f} years")

    df = res.merge(surv, on="participant", how="inner")

    keep = ["EMCI", "LMCI"] + (["CN"] if args.include_cn else [])
    before = df["participant"].nunique()
    df = df[df["dx_bl"].isin(keep)]
    print(f"\n[3/4] restricting to baseline {'/'.join(keep)}: "
          f"{df['participant'].nunique()} of {before} participants")
    print(f"  (baseline-AD participants are excluded: nothing left to predict)")
    print(f"  events in this subset: {int(df[df['experiment'] == 'exp1']['event'].sum())}")

    df.to_csv(os.path.join(args.out, "survival_table.csv"), index=False)

    print(f"\n[4/4] analysis, stage column = {args.stage_col}")
    lines = []
    rc = rank_checks(df, args.stage_col)
    rc.to_csv(os.path.join(args.out, "rank_checks.csv"), index=False)
    print("\n--- distribution-free checks ---")
    print(rc.to_string(index=False))
    lines.append("Distribution-free checks\n" + rc.to_string(index=False))

    cox, pw = cox_per_experiment(df, args.stage_col, use_covariates=not args.no_covariates)
    if cox is None:
        msg = ("\nlifelines is not installed, so the Cox models were skipped.\n"
               "  pip install lifelines\n"
               "then re-run. The checks above do not need it.")
        print(msg)
        lines.append(msg)
    else:
        cox.to_csv(os.path.join(args.out, "cox_per_experiment.csv"), index=False)
        show = [c for c in ["experiment", "n", "n_events", "stage_HR", "stage_p",
                            "stage_CI_low", "stage_CI_high", "concordance",
                            "subtype_LRT_chi2", "subtype_LRT_df", "subtype_LRT_p",
                            "concordance_with_subtype", "note"] if c in cox.columns]
        print("\n--- Cox proportional hazards (as in SuStaIn Table 3) ---")
        print(cox[show].to_string(index=False))
        lines.append("\nCox proportional hazards\n" + cox[show].to_string(index=False))
        if pw is not None:
            pw.to_csv(os.path.join(args.out, "cox_subtype_pairwise.csv"), index=False)
            print("\n--- pairwise subtype comparisons (SuStaIn's S-C / C-T / S-T design) ---")
            print(pw.to_string(index=False))
            lines.append("\nPairwise subtype comparisons\n" + pw.to_string(index=False))
        if "subtype_LRT_p" in cox.columns:
            L = cox.dropna(subset=["subtype_LRT_p"])
            nsig = int((L["subtype_LRT_p"] < 0.05).sum())
            v = (f"\nSubtype adds information beyond stage (likelihood ratio test, "
                 f"dummy-coded) in {nsig}/{len(L)} experiments at p<0.05.")
            print(v); lines.append(v)
        if "stage_HR" in cox.columns:
            ok = cox.dropna(subset=["stage_HR"])
            sig = ok[ok["stage_p"] < 0.05]
            verdict = (
                f"\nStage hazard ratio > 1 in {int((ok['stage_HR'] > 1).sum())}/{len(ok)} "
                f"experiments; significant at p<0.05 in {len(sig)}/{len(ok)}.\n"
                f"A hazard ratio above 1 means a later predicted baseline stage goes with\n"
                f"a sooner clinical AD diagnosis, which is the expected direction."
            )
            print(verdict)
            lines.append(verdict)

    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote outputs to {args.out}")


if __name__ == "__main__":
    main()
