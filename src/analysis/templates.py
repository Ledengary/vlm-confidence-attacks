def certificate_caption(n_ref, n_total, na_estimator, na_model, na_set, na_g0, na_floor):
    return (
        r"\caption{\textbf{The uniform amplitude certificate, refuted in "
        + f"{n_ref} of {n_total}"
        + r" estimator-by-cell profiles and unattainable in the remaining one.} "
        + r"$\gamma^{\ast}$ (with bootstrap 95\% CI) is the amplitude below which a certified readout provably keeps discriminating above the cell's operative floor, reported as the continuous crossing of the clean separation curve; clean sep.\ is the mean clean pairwise separation and $G(0)$ the clean AUROC, both over all correct-incorrect pairs; disp.\ is the mean per-item displacement the aimed attack witnesses and max the largest, both on the feasibility-restricted set of $n$ feas.\ items; frac$>\gamma^{\ast}$ is the fraction of feasible items whose witness exceeds $\gamma^{\ast}$, each falsifying a uniform certificate at that level; refuted marks whether a feasible witness exceeds $\gamma^{\ast}$. For "
        + f"{na_estimator} on {na_model}/{na_set}"
        + r" the clean AUROC $G(0)="
        + na_g0
        + r"$ already lies below the operative floor $"
        + na_floor
        + r"$, so no positive amplitude is certifiable; that row shows n/a for both $\gamma^{\ast}$ and refuted, since the certificate is unattainable rather than refuted.}"
    )


def sci_tex(x):
    s = f"{x:.1e}"
    mant, exp = s.split("e")
    return mant + r"\times10^{" + str(int(exp)) + "}"


def decomp_full_caption(max_recon_err):
    return (
        r"\caption{\textbf{The exact decomposition for all eighty-four estimator-by-cell profiles.} "
        + r"$w$ is the same-answer pair weight, $A_{\mathrm{same}}$ and $A_{\mathrm{cross}}$ the attacked discrimination on same-answer and cross-answer pairs, $A_{\mathrm{adv}}$ the reconstructed total, and $\rho$ the same-answer reachable-range overlap. The reconstruction error against the independently stored $A_{\mathrm{adv}}$ is at most $"
        + sci_tex(max_recon_err)
        + r"$.}"
    )


def phenomenon_caption(n_total):
    return (
        r"\caption{\textbf{The phenomenon in full:} clean and worst-case-oracle attacked AUROC for all "
        + str(n_total)
        + r" estimator-by-cell profiles, against the operative answer-string floor. `$<$floor' marks attacked at or below the floor; `$<$chance' marks attacked below 0.5. Attacked is the witnessed-union oracle (the strongest per-item candidate selection).}"
    )


def phenomenon_footer(n_chance, n_total, exc_est, exc_model, exc_set, exc_attacked, n_floor):
    return (
        r"\multicolumn{8}{@{}l}{\footnotesize Attacked $<$ chance in "
        + f"{n_chance} of {n_total}"
        + r" (the exception is "
        + f"{exc_est}/{exc_model}/{exc_set}, {exc_attacked}"
        + r"); attacked $\le$ floor in all "
        + f"{n_floor}"
        + r".}\\"
    )


def ladder_full_footnote(wb_excl, wb_bitexact, wb_min, wb_max):
    return (
        r"\vspace{3pt}"
        + "\n"
        + r"\noindent{\footnotesize All AUROC are worst-case-oracle attacked verifier confidence. ``mean, excl.\ under-powered white-box'' drops the one under-powered Molmo-verifier white-box pair (Gemma$\to$Molmo): white-box "
        + wb_excl
        + r". ``bit-exact verifier'' drops the three Molmo-verifier pairs, the only non-bit-exact differentiable paths (Table~\ref{tab:app-parity}): white-box "
        + wb_bitexact
        + r", range "
        + wb_min
        + r" to "
        + wb_max
        + r". In the notes column, score $n{=}100$ marks the three InternVL-verifier pairs, ``near-chance clean'' the two Molmo-verifier pairs on POPE, and ``parity'' the Molmo-verifier differentiable-path gap.}"
    )


def ladder_parity_footnote(mean_gap):
    return (
        r"\vspace{3pt}"
        + "\n"
        + r"\noindent{\footnotesize The Molmo verifier's per-item mean gap is $\approx$"
        + mean_gap
        + r" against the 0.144 maximum. Every reported verifier score is recomputed with the official processor, so the gap affects only the white-box search, never a reported AUROC.}"
    )


def universal_footnote(max_gap, n_cells_word, n_contain_word, n_channels_word, pim_lo, pim_hi, ratio_lo, ratio_hi):
    return (
        r"\noindent{\footnotesize drop = clean minus perturbed AUROC. The largest universal-minus-random difference is $"
        + max_gap
        + r"$ over all "
        + n_cells_word
        + r" cells, and "
        + n_contain_word
        + " of "
        + n_channels_word
        + r" channels have uni$-$rand intervals containing 0. The per-image aimed attack removes $"
        + pim_lo
        + r"$ to $"
        + pim_hi
        + r"$ on SAPLMA in the same cells, "
        + ratio_lo
        + " to "
        + ratio_hi
        + r" times the largest universal-versus-random gap.}"
    )


def certificate_footer(n_ref, n_total, na_estimator, na_model, na_set):
    return [
        r"\multicolumn{11}{@{}l}{\footnotesize "
        + f"{n_ref} of {n_total}"
        + r" profiles refuted; 1 unattainable ("
        + f"{na_estimator}, {na_model}, {na_set}:"
        + r"}\\",
        r"\multicolumn{11}{@{}l}{\footnotesize $G(0)$ below the operative floor, so no positive amplitude is certifiable).}\\",
    ]
