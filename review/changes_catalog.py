"""
changes_catalog.py  —  every edit I made for the prediction & prediction_multiclass widgets,
                       written out plainly so you can review each one without opening the HTML.

There are only TWO logical changes:
   (A) make block SLQ (block size 4) the DEFAULT for the eigenspectrum density plots, and
   (B) actually implement block SLQ in the browser (it only existed on the GPU server before).

The algorithm itself is in  review/block_slq.py  (readable, runnable).  This file is just the
map: what changed, where, from what, to what, and why.  Line numbers are approximate (~).

Files touched:
   eos_widget_prediction/index_prediction.html            (the prediction widget)
   eos_widget_prediction/index_prediction_multiclass.html (the multiclass prediction widget)
   server.py                                              (the GPU backend that drives both)
   index.html                                             (the eos_lab widget — same treatment, for parity)

The edits are IDENTICAL in the two prediction HTML files (their JS was byte-for-byte the same
before I started, so the same change applied to both).
================================================================================================


CHANGE 1 — new function `_mgsBlock`  (added)
------------------------------------------------------------------------------------------------
  index_prediction.html            ~L1241
  index_prediction_multiclass.html ~L1238

  A two-pass Gram-Schmidt QR of `b` column vectors. JS has no built-in QR, so this hand-writes
  the "orthonormalize these b vectors" step that block Lanczos needs. In block_slq.py this is
  just `numpy.linalg.qr(...)`.


CHANGE 2 — new function `blockLanczosCore`  (added)
------------------------------------------------------------------------------------------------
  index_prediction.html            ~L1254
  index_prediction_multiclass.html ~L1251

  The block Lanczos recurrence: start from `b` random orthonormal vectors and build the
  block-tridiagonal matrix T. Port of server.py `_block_lanczos_core` (L3071).
  Readable version: `block_lanczos_tridiagonal` in block_slq.py.


CHANGE 3 — `slqDensity` gained a `block` argument  (modified)
------------------------------------------------------------------------------------------------
  index_prediction.html            ~L1281
  index_prediction_multiclass.html ~L1278

  BEFORE:  function slqDensity(hvpFn, p, nProbe, m, nGrid, seed) { ...only single-vector SLQ... }
  AFTER:   function slqDensity(hvpFn, p, nProbe, m, nGrid, seed, block) {
               if (block > 1)  ...block Lanczos path (new)...
               else            ...the original single-vector path (unchanged)...
           }
  So block=1 behaves exactly as before; block>1 uses the new block path. Readable versions:
  `standard_slq` (block=1) and `block_slq` (block>1) in block_slq.py.


CHANGE 4 — remember the chosen block size on the run-state object  (modified)
------------------------------------------------------------------------------------------------
  index_prediction.html            ~L4432   (the big `ST = { ... }` line)
  index_prediction_multiclass.html ~L4431

  Added one field:   slqBlock: Math.max(1, +val('slqblock') || 4)
  i.e. read the "SLQ mode" dropdown; if unset, default to 4.


CHANGE 5 — pass the block size into the four density plots  (modified)
------------------------------------------------------------------------------------------------
  index_prediction.html            ~L4971–4974
  index_prediction_multiclass.html ~L4970–4973

  The four eigenspectrum plots are H, G, S, and the full loss Hessian ∇²L. Each calls slqDensity
  with a different matrix-vector product (hvpF/hvpG/hvpS/hvpL). I added `bk` (= ST.slqBlock) as
  the new last argument to all four:

      const bk = S.slqBlock || 1;
      const dH  = slqDensity(v => hvpF(...), p, S.nProbe, S.mSLQ, 80, 0x11, bk);   // function Hessian H
      const dG  = slqDensity(v => hvpG(...), p, S.nProbe, S.mSLQ, 80, 0x22, bk);   // Gauss-Newton G
      const dS  = slqDensity(v => hvpS(...), p, S.nProbe, S.mSLQ, 80, 0x33, bk);   // residual term S
      const dHL = slqDensity(v => hvpL(...), p, S.nProbe, S.mSLQ, 80, 0x44, bk);   // full loss Hessian ∇²L


CHANGE 6 — the "SLQ mode" dropdown now defaults to block SLQ (b=4)  (modified)
------------------------------------------------------------------------------------------------
  index_prediction.html            ~L174
  index_prediction_multiclass.html ~L171

  BEFORE:  <option value="1" selected>standard SLQ</option> ... <option value="4">block SLQ (b=4)</option>
  AFTER:   <option value="1">standard SLQ</option>          ... <option value="4" selected>block SLQ (b=4)</option>
  (The user can still pick standard SLQ or b=2/b=8 from the same dropdown.)


CHANGE 7 — server default block size 1 → 4  (modified, server.py)
------------------------------------------------------------------------------------------------
  server.py  L3989   run_stream:      slqBlock = max(1, int(P.get("slqblock", 4)))    # was 1
  server.py  L6118   _parse_params:   "slqblock": max(1, fi("slqblock", 4))           # was 1

  So GPU runs AND headless captures also default to block-4. (The block SLQ math on the server —
  `_block_lanczos_core` L3071, `slq_density` L3111 — already existed; only the default changed.)


PARITY NOTE — index.html (eos_lab widget)
------------------------------------------------------------------------------------------------
  Same seven-style change applied to index.html so all three widgets match. index.html had no
  "SLQ mode" dropdown, so I also ADDED one (defaulting to b=4) next to the "SLQ probes" control.


HOW IT WAS VALIDATED (no browser/node on the box)
------------------------------------------------------------------------------------------------
  There is no JavaScript runtime installed here, so I could not run the widget JS directly.
  Instead I transcribed the JS block Lanczos into Python EXACTLY (same manual Gram-Schmidt, same
  recurrence, same random-start convention) and checked that transcription against (a) exact
  eigenvalues and (b) the already-trusted server `slq_density(block=...)`. Block SLQ matched and
  beat standard SLQ. `review/block_slq.py` is a cleaned-up, simpler version of that check that
  you can run yourself.

WHY block-4 (the finding behind the default)
------------------------------------------------------------------------------------------------
  A separate standalone experiment (spectrum_evolution/spec_evo.py) measured, over training, how
  close each SLQ variant gets to the exact spectrum of Q_r, JJᵀ, and H. Block SLQ beat standard
  SLQ on all three; block 4–8 was the accuracy-per-cost sweet spot (block 16 gave no extra benefit
  and can over-shoot on small operators). So b=4 became the default.
"""

print(__doc__)
