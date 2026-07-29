import base64,io,json
from PIL import Image
def enc(path,w=1100):
    im=Image.open(path).convert("RGB"); im=im.resize((w,int(im.height*w/im.width)),Image.LANCZOS)
    b=io.BytesIO(); im.save(b,"JPEG",quality=90,optimize=True); return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()
FIG={k:enc(f"psstudy/figs/{v}") for k,v in {"fixed":"g_fixedbasis2.png","lowrank":"g_qlowrank.png","cycle":"g_cycle_ev.png","qmodes":"g_qmodes.png"}.items()}
HTML=r'''<meta charset="utf-8">
<title>Progressive Sharpening, Refined</title>
<style>
:root{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#181a20; --muted:#4a4e5a; --faint:#878c99;
  --line:#e5e7ec; --accent:#5b3ec7; --accent-soft:#efeaff; --good:#0f8a5f; --warn:#b7791f;
  --serif:"Charter","Iowan Old Style","Georgia",serif;
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Code",monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0f1014; --panel:#171922; --ink:#eceef4; --muted:#a6abba; --faint:#6b7180;
  --line:#262a36; --accent:#a893ff; --accent-soft:#211d3a; --good:#3fca8f; --warn:#e0b25a;}}
:root[data-theme="dark"]{
  --bg:#0f1014; --panel:#171922; --ink:#eceef4; --muted:#a6abba; --faint:#6b7180;
  --line:#262a36; --accent:#a893ff; --accent-soft:#211d3a; --good:#3fca8f; --warn:#e0b25a;}
:root[data-theme="light"]{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#181a20; --muted:#4a4e5a; --faint:#878c99;
  --line:#e5e7ec; --accent:#5b3ec7; --accent-soft:#efeaff; --good:#0f8a5f; --warn:#b7791f;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6;
  font-size:16.5px;-webkit-font-smoothing:antialiased}
.wrap{max-width:720px;margin:0 auto;padding:60px 24px 100px}
.eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 14px}
h1{font-family:var(--serif);font-weight:600;font-size:40px;line-height:1.12;letter-spacing:-.01em;margin:0 0 20px;text-wrap:balance}
.thesis{font-size:20px;line-height:1.5;color:var(--ink);margin:0 0 10px;font-family:var(--serif)}
.thesis b{color:var(--accent);font-weight:600}
.lede{color:var(--muted);font-size:16px;margin:18px 0 0}
.lede b{color:var(--ink);font-weight:600}
.rule{height:1px;background:var(--line);margin:46px 0}
.fact{margin:0 0 8px}
.fact .n{font-family:var(--mono);font-size:12px;color:var(--accent);letter-spacing:.05em}
.tag{font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:.04em;padding:2px 8px;border-radius:20px;vertical-align:middle;margin-left:9px}
.t-clear{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}
.t-tent{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
h2{font-family:var(--serif);font-weight:600;font-size:25px;line-height:1.2;margin:8px 0 12px;letter-spacing:-.01em;text-wrap:balance}
p{margin:0 0 15px}
p b{color:var(--ink);font-weight:600}
.ev{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:0 10px 10px 0;
  padding:14px 18px;margin:16px 0;font-size:15px;color:var(--muted)}
.ev .h{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:6px}
.ev b{color:var(--ink);font-weight:600}
figure{margin:20px 0 6px}
figure img{width:100%;display:block;border:1px solid var(--line);border-radius:10px;background:#fff}
figcaption{font-size:13px;color:var(--faint);margin-top:8px;font-family:var(--mono);line-height:1.5}
.kicker{font-family:var(--serif);font-size:19px;color:var(--ink);margin:0 0 8px}
.so{background:var(--accent-soft);border:1px solid color-mix(in srgb,var(--accent) 30%,var(--line));border-radius:14px;padding:22px 24px;margin-top:14px}
.so h2{margin-top:0}
mono,.m{font-family:var(--mono);font-size:.9em}
.foot{margin-top:40px;font-size:12.5px;color:var(--faint);font-family:var(--mono);line-height:1.6}
</style>

<div class="wrap">
  <p class="eyebrow">Progressive sharpening · what is actually going on</p>
  <h1>The network sharpens the directions it is learning</h1>
  <p class="thesis">Progressive sharpening is not a quirk of the loss curve. It is <b>feature learning seen through
    curvature</b>: the error signal repeatedly finds a curvature direction it can descend, amplifies it until the
    step size hits the <b>2/η</b> stability wall, then hands off to the next.</p>
  <p class="lede">Under full-batch gradient descent the sharpness (top Hessian eigenvalue) climbs and then parks at
    2/η — the "edge of stability." The object doing the work underneath is the <b>residual-weighted function
    Hessian</b> <span class="m">M_r = Σ_k r_k Q_k</span>, whose top eigenvector is the direction being sharpened
    right now. Four facts pin down what that direction is — and each is settled by one decisive experiment, not by
    eyeballing a plot.</p>

  <div class="rule"></div>

  <p class="fact"><span class="n">FACT 1</span><span class="tag t-clear">clear</span></p>
  <h2>The direction is error-aligned — not just a big-curvature direction</h2>
  <p>The theory rides the top eigenvector of <span class="m">M_r</span>. Is that eigenvector special because it
    carries a lot of curvature, or because it points the <b>right way</b>? Swap the true function-Hessian Q for a
    <b>random</b> one of the same overall size: its top direction becomes unrelated to the gradient. Then make the
    random Q <b>also low-rank</b> — same few directions, same eigenvalues, only the directions randomized. It gets
    the big eigenvalue back, yet its alignment with the gradient <b>stays at chance</b>.</p>
  <div class="ev"><div class="h">decisive test</div>Alignment with the gradient: true Q <b>0.09–0.42</b>,
    frozen-at-init Q <b>0.07–0.36</b>, random Q <b>~0.03</b>, random <em>low-rank</em> Q <b>~0.04</b> (chance).
    So low-rank concentration only explains the eigenvalue's <b>size</b>; what makes the direction work is that it
    <b>lines up with where the error pushes</b> — a real feature of the trained network no random operator has.</div>
  <figure><img src="__LOWRANK__" alt="random low-rank Q recovers the eigenvalue but not the alignment">
    <figcaption>Left: the top eigenvalue — random low-rank (orange) matches the true one. Right: alignment with the gradient — both randoms sit at the dotted chance line; only the true and frozen-at-init Q rise above it.</figcaption></figure>

  <div class="rule"></div>

  <p class="fact"><span class="n">FACT 2</span><span class="tag t-clear">clear</span></p>
  <h2>Half of the sharpening directions are inherited from init — half are newly grown</h2>
  <p>Freeze Q at initialization and the sharpening direction barely moves: it stays <b>96–99% correlated</b> with the
    true, evolving one for the entire run. So much of the structure is present before a single step is taken. But
    that is only half the story. <b>Fix the eigenbasis at initialization</b> and watch each fixed direction: the
    network's curvature energy <b>drains out of the init-dominant directions</b>, and by the end only about
    <b>half</b> of the current sharpening direction still lives in the init top-16 basis. The rest sits in directions
    that were <b>negligible at init</b> — new features the network built during training.</p>
  <div class="ev"><div class="h">decisive test · this is the new panel you asked for</div>Fraction of the current
    top sharpening direction still inside the frozen init top-16 basis: <b>1.00 → 0.45</b> (sparse-parity),
    <b>1.00 → 0.60</b> (saddle), <b>1.00 → 0.50</b> (chebyshev). The gap below 1 is exactly the part of the
    sharpening direction the network <b>grew from scratch</b>. Because the ranking never re-sorts, each row is one
    direction fixed at init — so you can literally watch which ones grow and which flatten.</div>
  <figure><img src="__FIXED__" alt="fixed-basis: energy drains from init-directions into new ones">
    <figcaption>Top: NTK energy per fixed init-direction over training (rank never re-sorts). Bottom: the red curve is how much of the current top direction still lives in the init basis — the shaded gap is the newly-grown part. This is the §6 fixed-basis panel now built into the prediction & prediction_multiclass widgets.</figcaption></figure>
  <figure><img src="__QMODES__" alt="evolving vs init-fixed vs random Q, per dataset">
    <figcaption>The "inherited" half, directly: replacing the true evolving Q with the one frozen at init (green vs pink) leaves both the top curvature (top row) and the residual alignment (bottom row) almost unchanged — correlation 0.96–0.99. A random Q (purple) collapses both. So the sharpening direction is largely fixed at initialization; only its scale needs the live Q.</figcaption></figure>

  <div class="rule"></div>

  <p class="fact"><span class="n">FACT 3</span><span class="tag t-clear">clear</span></p>
  <h2>It repeats — a cycle of hand-offs, capped by the edge</h2>
  <p>The top curvature direction holds steady most of the time, then <b>jumps to a genuinely new direction</b> —
    more than 60° away from every direction seen before. Tracking the eigenvector itself (no threshold counter), it
    jumps <b>4–6 times per run</b>, and the count is the <b>same across random seeds</b>. Each jump is one turn of
    the align → sharpen → destabilize → hand-off cycle, gated by the 2/η stability limit.</p>
  <figure><img src="__CYCLE__" alt="the top curvature direction jumps to new directions repeatedly">
    <figcaption>Purple = cosine between the top direction now and one step ago; it sits at 1 (stable) then crashes to ~0 at each hand-off. Pink lines mark the new directions. Seeds 0 and 1 give the same count.</figcaption></figure>
  <div class="ev"><div class="h">how a "switch" is counted</div>There is no canonical definition, so this is an
    explicit rule: track the top eigenvector <span class="m">u₁</span> of <span class="m">M_r</span> each step; keep
    a running list of anchor directions (seeded at t=0); a step counts as a <b>new direction</b> when its
    <span class="m">|cos|</span> with <b>every</b> anchor so far is below a cutoff <span class="m">τ</span> — i.e. it
    is more than <span class="m">arccos τ</span> away from <em>all</em> directions seen before (so returning to an old
    direction does not re-count). The count is threshold-dependent: <b>4–6 at τ=0.5 (&gt;60°)</b>, up to ~11 at
    τ=0.85 (&gt;32°). The <em>phenomenon</em> — repeated far-away rotations — is robust; the exact number is a
    choice of <span class="m">τ</span>.</div>

  <div class="rule"></div>

  <p class="fact"><span class="n">FACT 4</span><span class="tag t-clear">clear</span></p>
  <h2>The theory works on two independent axes</h2>
  <p>Whether the frozen-<span class="m">M_r</span> theory predicts the sharpening splits into two separate questions.
    It gets the <b>direction</b> right when the curvature is <b>low-rank</b>: exact on sparse-parity, good on
    chebyshev and saddle, lost on CIFAR. It gets the <b>size</b> right when the curvature <b>grows slowly</b>: exact
    on sparse-parity and saddle, blowing up on chebyshev and CIFAR. Chebyshev is the clean separation — <b>perfect
    direction, wrong size</b>, because its top curvature is precisely the thing growing fast. The one quantity
    predictable everywhere is the <b>total</b> curvature (the trace).</p>

  <div class="rule"></div>

  <div class="so">
    <h2>So — the one picture</h2>
    <p style="margin-bottom:0">The error signal repeatedly selects an <b>error-aligned, low-rank</b> curvature
      direction, amplifies it to the <b>2/η edge</b>, and <b>hands off</b> to the next. The directions it uses are
      <b>partly inherited from initialization and partly grown during training</b>. So <span class="m">M_r</span>'s
      top eigenvector is the right thing to watch — but trust it only as a <em>direction</em>, and only for a few
      steps at a time, because the network keeps growing new ones. Progressive sharpening is the curvature-space
      shadow of feature learning.</p>
  </div>

  <p class="foot">Full-batch GD · MSE · tanh MLPs (+ VGG-11) · datasets: sparse-parity, saddle, chebyshev, MNIST,
    CIFAR. Every claim tagged <span style="color:var(--good)">clear</span> is checked across runs and, where noted,
    across seeds. The fixed-basis diagnostic (Fact 2) is now an optional panel — "§6 fixed-basis (init ranking)" —
    in both the prediction and prediction_multiclass widgets.</p>
</div>'''
for k,u in {"__LOWRANK__":FIG["lowrank"],"__FIXED__":FIG["fixed"],"__CYCLE__":FIG["cycle"],"__QMODES__":FIG["qmodes"]}.items():
    HTML=HTML.replace(k,u)
open("results_webpage/ps_refined.html","w").write(HTML)
print("wrote ps_refined.html %.1f MB"%(len(HTML)/1e6))
