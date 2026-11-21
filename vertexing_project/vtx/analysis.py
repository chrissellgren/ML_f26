"""Analysis functions from notebooks/analysis.ipynb, copied here so they can be imported:

    from vtx import analysis
    data_cut = analysis.cut(data)
    taus = analysis.select_taus(data)
    label = analysis.correct_vertex(data_cut)
    choices = analysis.tjva_andsumpt2(data, taus)   # per-tau SumPT2 / TJVA-lite v1 / v2 choices

Keep this file and the notebook in sync: develop in the notebook, then paste the final version here.
"""
import awkward as ak
import numpy as np
import pandas as pd


def cut(all, printing=False):
    """ Import entire data as awkward array, return data structure with only passing events.
    Event requirements: has at least one true hadronic tau with pT>20gev, |eta| < 2.5
    And has tau within dR < 0.2 of a reco jet with pT > 20 Gev  """
    if printing: print(f"Num events before cuts: {len(all)}")
    # mass hadronic tau, pt, eta cuts
    m_truthtau = all.truth_tau_from_hard
    m_hadronic = ~all.truth_tau_leptonic
    m_istau = m_truthtau & m_hadronic
    m_pt = (all.truth_tau_vis_pt > 20)  # Gev
    m_eta = (np.abs(all.truth_tau_vis_eta) < 2.5)

    passcut = m_istau & m_pt & m_eta
    atleast1 = (ak.sum(passcut, axis=-1) >= 1)

    # data only keeps events with at least one tau passing cuts
    data = all[atleast1]
    if printing: print(f"Num events after tau cuts: {len(data)}")

    ############# reco jet associated with tau checks

    tau_etas = data.truth_tau_vis_eta
    tau_phis = data.truth_tau_vis_phi
    jet_etas = data.jet_eta
    jet_phis = data.jet_phi

    # broadcast / insert axis to calculate deta across variable jet number per event
    detas = tau_etas[:, :, None] - jet_etas[:, None, :]
    # make sure to wrap (0, 2pi) to (-pi, pi)
    dphis = (tau_phis[:, :, None] - jet_phis[:, None, :] + np.pi) % (2*np.pi) - np.pi
    dR = np.sqrt(detas**2+dphis**2)

    # now check if a jet has dR < 0.2 to a tau and has pT > 20 Gev
    jet_pt = data.jet_pt[:, None, :]
    hasgoodjet = (dR < 0.2) & (jet_pt > 20)

    # find whether any jet in the event has a tau near it
    tau_nearjet = ak.any(hasgoodjet, axis=-1) & passcut[atleast1]
    # if any tau in the event has a good jet near it
    m_jettau = ak.any(tau_nearjet, axis=-1)

    data = data[m_jettau]
    if printing: print(f"Num events after jet cut: {len(data)}")
    return data


def select_taus(all):
    """Per-tau version of cut(): one row per evaluated tau.
    Same requirements as cut(): truth hadronic tau from the gamma*, vis pT > 20 GeV, |eta| < 2.5,
    with a reco jet (pT > 20 GeV) within dR < 0.2. If several jets match, take the closest.
    Returns a DataFrame with columns:
      event  : index of the event in `all` (use it to grab that event's tracks/vertices: all.trk_pt[taus.event])
      tau    : index into that event's truth_tau_* lists
      jet    : index into that event's jet_* lists
      jet_pt, jet_eta, jet_phi : the matched jet (the seed axis for TJVA-lite)
      tau_pt, nprong : truth visible pT and number of charged prongs (for efficiency vs pT / prong plots)"""
    # same tau cuts as in cut()
    passcut = (all.truth_tau_from_hard & ~all.truth_tau_leptonic
               & (all.truth_tau_vis_pt > 20) & (np.abs(all.truth_tau_vis_eta) < 2.5))

    # same dR computation as in cut(), but on all events so `event` indexes the input array
    detas = all.truth_tau_vis_eta[:, :, None] - all.jet_eta[:, None, :]
    dphis = (all.truth_tau_vis_phi[:, :, None] - all.jet_phi[:, None, :] + np.pi) % (2*np.pi) - np.pi
    dR = np.sqrt(detas**2 + dphis**2)
    hasgoodjet = (dR < 0.2) & (all.jet_pt[:, None, :] > 20)

    # a tau is evaluated if it passes the cuts and has a good jet; its jet is the closest good one
    keep = passcut & ak.any(hasgoodjet, axis=-1)
    jet_idx = ak.argmin(ak.where(hasgoodjet, dR, np.inf), axis=-1)

    # event and tau index for every tau, then keep only the evaluated ones
    event_idx = ak.broadcast_arrays(np.arange(len(all)), passcut)[0]
    tau_idx = ak.local_index(passcut, axis=1)
    event = ak.to_numpy(ak.flatten(event_idx[keep]))
    tau = ak.to_numpy(ak.flatten(tau_idx[keep]))
    jet = ak.to_numpy(ak.flatten(jet_idx[keep]))

    # pick one element per tau: flatten the jagged column, then jump to (start of event) + (index within event)
    def pick(x, i):
        start = np.concatenate([[0], np.cumsum(ak.to_numpy(ak.num(x)))])
        return ak.to_numpy(ak.flatten(x))[start[event] + i]
    return pd.DataFrame({
        "event": event, "tau": tau, "jet": jet,
        "jet_pt": pick(all.jet_pt, jet), "jet_eta": pick(all.jet_eta, jet), "jet_phi": pick(all.jet_phi, jet),
        "tau_pt": pick(all.truth_tau_vis_pt, tau), "nprong": pick(all.truth_tau_nprong, tau),
    })


def correct_vertex(data, purity=0):
    """Params: data (awk array) purity = fraction of tracks in best vertex which must be from the target process for the event to have a best vertex
    Returns: array of len (n,) of the index of the best vertex candidate (vertex with most HS process tracks pointing to it)
    ie the rv_index of the candidate holding the most tracks with trk_truth_vtx == 0,
    if those are > purity of its tracks; else -1."""

    # mask for ensuring tracks come from hs process and isn't an unassigned track (-1)
    m_tracks_hs = (data.trk_truth_vtx == 0) & (data.trk_delphes_vtx >= 0)
    hs_trk_vertices = data.trk_delphes_vtx[m_tracks_hs]  # vertex index of the tracks that are from HS

    # now find the mode of hs_trk_vertices on the -1 axis
    # Sort vertex indices within each event so identical values sit next to each other
    sorted_vtx = ak.sort(hs_trk_vertices, axis=-1)
    # Count the length of each consecutive run of identical indices
    counts = ak.run_lengths(sorted_vtx)
    # Group sorted_vtx by those run lengths and grab the single vertex value per run
    runs = ak.unflatten(ak.flatten(sorted_vtx), ak.flatten(counts))
    unique_vtx = ak.unflatten(ak.firsts(runs), ak.num(counts))
    # pick the vertex index whose run had the highest count
    best_run = ak.argmax(counts, axis=-1, keepdims=True)
    best_vtx = ak.firsts(unique_vtx[best_run], axis=-1)  # shape: (n,)

    # now check if the the best vertex has over half of its tracks from the true hs

    # how many tracks did the best vertex have?
    ntrks_frombestvertex = ak.sum((data.trk_delphes_vtx == best_vtx), axis=-1)
    n_hs_trks_inbestvtx = ak.sum((data.trk_delphes_vtx == best_vtx) & m_tracks_hs, axis=-1)

    merged_events = (n_hs_trks_inbestvtx / ntrks_frombestvertex < purity)
    best_vtx = ak.where(merged_events, -1, best_vtx)
    # events where no vertex holds any HS track come out as None: mark them -1 too
    return ak.fill_none(best_vtx, -1)


def wilson(k, n, z=1.0):
    """Wilson score interval for k successes out of n (z=1: 68%)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


# ---------------------------------------------------------------------------------------------
# Vertex choices per tau. tjva_andsumpt2() (Chris's original function) was split into:
#   sumpt2_choice()  -> SumPT2 choice, per event, copied to each tau
#   cone_tracks()    -> the tracks in each tau's dR < 0.2 cone (reused by v1, v2 and later the Transformer)
#   tjva_v1()        -> score f(v) with the Delphes track->vertex assignment
#   tjva_v2()        -> same as v1, plus a "rescue" of cone tracks that are in no vertex
# All choices are rv_index values; -2 means "no candidate vertex in the event".
# ---------------------------------------------------------------------------------------------

def sumpt2_choice(data, taus):
    """SumPT2: per event the rv_index with the largest rv_sumpt2, copied to each tau (array of len n_taus)."""
    ev = taus.event.values
    sumpt2_per_event = ak.firsts(data.rv_index[ak.argmax(data.rv_sumpt2, axis=1, keepdims=True)])
    return ak.to_numpy(ak.fill_none(sumpt2_per_event, -2))[ev]


def cone_tracks(data, taus, dr_max=0.2, pt_min=1.0):
    """Tracks in each tau's cone: dR(track, matched jet axis) < dr_max and pT > pt_min.
    Returns a dict of per-tau jagged arrays (one list of cone tracks per tau): pt, vtx (Delphes vertex, -1 = none),
    dz (z0) and eta (needed by the v2 rescue)."""
    ev = taus.event.values

    # from the per-tau table, pull the kinematics of the tracks in that tau event
    # and get kinematics of jets associated with that tau
    trk_pts  = data.trk_pt[ev]
    trk_etas = data.trk_eta[ev]
    trk_phis = data.trk_phi[ev]
    jet_eta = taus.jet_eta.values
    jet_phi = taus.jet_phi.values

    # cut out tracks not in the dR<0.2 cone from jet axis
    # calculate dR, subtract the jet values from each track value
    detas = trk_etas - jet_eta
    dphis = (trk_phis - jet_phi + np.pi) % (2*np.pi) - np.pi  # wrap (0, 2pi) to (-pi, pi)
    dR = np.sqrt(detas**2+dphis**2)
    m_trkincone = (dR < dr_max) & (trk_pts > pt_min)  # shape (n, ntrks per event)

    return {"pt":  trk_pts[m_trkincone],                    # used for calculating f(v)
            "vtx": data.trk_delphes_vtx[ev][m_trkincone],   # which vertex each track in cone points to
            "dz":  data.trk_dz[ev][m_trkincone],
            "eta": trk_etas[m_trkincone]}


def tjva_v1(cone, sumpt2_choice_per_tau):
    """TJVA-lite v1: per tau, the vertex holding the largest pT fraction f(v) of the cone tracks,
    using the Delphes track->vertex assignment (cone["vtx"]). Unassigned tracks (-1) only count in the denominator.
    Taus with no assigned cone track fall back to the SumPT2 choice.
    Returns (choice, max_f, fallback) as numpy arrays of len n_taus."""
    trk_pt_incone = cone["pt"]
    vtx_trkincone = cone["vtx"]

    # make a total list of all vertices that appear in any cone
    vertices = np.unique(ak.to_numpy(ak.flatten(vtx_trkincone)))
    # ignore vertex = -1 (tracks that haven't been assigned a vertex)
    vertices = vertices[vertices >= 0]

    # scan through vertices to calculate f(v) for each one
    sumpt_tot = ak.sum(trk_pt_incone, axis=-1)
    fs = []
    for v in vertices:
        # mask for all tracks from this vertex
        trk_thisvtx = (vtx_trkincone == v)
        sumpt_thisvtx = ak.sum(trk_pt_incone[trk_thisvtx], axis=-1)
        fs.append(ak.to_numpy(sumpt_thisvtx / ak.where(sumpt_tot > 0, sumpt_tot, 1)))
    f_vals = np.stack(fs, axis=1)   # shape (n_taus, n_vertex_values)

    max_f = f_vals.max(axis=1)
    fallback = max_f <= 0           # empty cone, or every cone track unassigned

    best_v = vertices[f_vals.argmax(axis=1)]
    # for empty cone taus, replace best vertex pick with the sumpt2 choice
    # logic of np.where(condition,x,y) :if condition, x; else y
    choice = np.where(fallback, sumpt2_choice_per_tau, best_v)
    return choice, max_f, fallback


def tjva_v2(cone, data, taus, sumpt2_choice_per_tau, window=1.0):
    """TJVA-lite v2 = v1 + one extra step before scoring:
      NEW (structure): needs the candidate vertex positions (data.rv_z, data.rv_index) in addition to the cone.
      NEW (function):  a cone track that the vertex finder left unassigned (vtx == -1) is attached to the candidate
                       vertex nearest in z, if |z0 - z_vertex| * sin(theta) < window (mm). This mimics ATLAS's
                       "Nonprompt" track-to-vertex association, which lets TJVA use tau tracks that are in no vertex fit.
      SAME as v1:      the scoring (f(v), argmax, SumPT2 fallback) -- it just calls tjva_v1 on the updated assignment.
    Returns (choice, max_f, fallback, n_rescued) with n_rescued = number of rescued cone tracks per tau."""
    
    new_vtx = rescue_unassigned(cone,data,taus)

    # SAME as v1 from here on
    choice, max_f, fallback = tjva_v1({**cone, "vtx": new_vtx}, sumpt2_choice_per_tau)
    return choice, max_f, fallback, ak.to_numpy(ak.sum(rescued, axis=-1))

def rescue_unassigned(cone,data,taus):
    """See docstring in tjva_v2 for function"""
    ev = taus.event.values
    rv_z = data.rv_z[ev]           # candidate vertex positions per tau, shape (n, nvertices)
    rv_index = data.rv_index[ev]

    # |z0 - z_vertex| * sin(theta) for every (cone track, candidate vertex) pair, shape (n, ncone, nvertices)
    sin_theta = 1 / np.cosh(cone["eta"])
    dist = np.abs(cone["dz"][:, :, None] - rv_z[:, None, :]) * sin_theta[:, :, None]

    # nearest candidate vertex for each cone track
    nearest = ak.argmin(dist, axis=-1, keepdims=True)
    nearest_vtx = ak.firsts(ak.broadcast_arrays(rv_index[:, None, :], dist)[0][nearest], axis=-1)
    nearest_dist = ak.min(dist, axis=-1)

    # rescue: only tracks that are unassigned AND close enough to a vertex; everything else keeps its assignment
    rescued = (cone["vtx"] < 0) & ak.fill_none(nearest_dist < window, False)
    new_vtx = ak.where(rescued, ak.fill_none(nearest_vtx, -1), cone["vtx"])
    return new_vtx

def tjva_andsumpt2(data, taus=None):
    """Per-tau choices of every baseline, as a DataFrame aligned row-by-row with `taus`."""
    if taus is None:
        taus = select_taus(data)  # allows to feed in taus if already done to avoid calling twice
    sp = sumpt2_choice(data, taus)
    cone = cone_tracks(data, taus)
    v1, max_f, fallback = tjva_v1(cone, sp)
    v2, max_f_v2, fallback_v2, n_rescued = tjva_v2(cone, data, taus, sp)
    return pd.DataFrame({"sumpt2_choice": sp, "tjva_choice": v1, "tjva_v2_choice": v2,
                         "max_f": max_f, "fallback": fallback, "fallback_v2": fallback_v2,
                         "n_rescued": n_rescued})


def efficiency(correct, weights=None):
    """(efficiency, lower, upper) with a 68% Wilson interval. With weights, uses the effective number of
    entries (sum w)^2 / sum w^2 so the error reflects the weighted sample."""
    correct = np.asarray(correct, float)
    w = np.ones_like(correct) if weights is None else np.asarray(weights, float)
    eff = np.sum(w * correct) / np.sum(w)
    n_eff = np.sum(w)**2 / np.sum(w**2)
    lo, hi = wilson(eff * n_eff, n_eff)
    return eff, lo, hi


def chosen_z(data, taus, choice):
    """z position (rv_z) of the chosen vertex for each tau (NaN if no choice). For the generator-vertex cross-check."""
    ev = taus.event.values
    z = ak.firsts(data.rv_z[ev][data.rv_index[ev] == choice])
    return ak.to_numpy(ak.fill_none(z, np.nan))
