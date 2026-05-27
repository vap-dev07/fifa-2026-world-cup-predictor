"""
FastAPI server for the FIFA 2026 World Cup Predictor.

Startup: loads the trained ensemble model and team feature database into memory.
Endpoints:
    GET  /            - health check
    POST /api/predict - head-to-head match probability prediction
"""

from contextlib import asynccontextmanager
import pathlib

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = pathlib.Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "world_cup_ensemble.pkl"
FEATURES_PATH = ROOT / "data" / "processed" / "team_features_db.csv"

# Approximate World Cup draw rate used as the upper bound for draw probability.
# When two teams are perfectly matched (balance=1), draw probability peaks here.
DRAW_BASE = 0.25

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = joblib.load(MODEL_PATH)
    _state["db"] = pd.read_csv(FEATURES_PATH)
    print(f"Model loaded from   : {MODEL_PATH}")
    print(f"Feature DB loaded   : {FEATURES_PATH}  ({len(_state['db'])} rows)")
    yield
    _state.clear()


app = FastAPI(
    title="FIFA 2026 World Cup Prediction Engine",
    description=(
        "A machine-learning ensemble (XGBoost + Random Forest + Logistic Regression) "
        "that predicts head-to-head match outcomes for the FIFA 2026 World Cup. "
        "Submit any two national teams and receive win/draw probabilities derived from "
        "historical performance features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class MatchRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"team_a": "Argentina", "team_b": "Germany"}
            ]
        }
    }

    team_a: str
    team_b: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lookup(team_name: str) -> pd.Series:
    """
    Return a single consolidated feature vector for a team.
    Matches case-insensitively; averages rows when a team appears more than once.
    Raises HTTP 400 if the team is not in the database.
    """
    db: pd.DataFrame = _state["db"]
    mask = db["team_name"].str.lower() == team_name.strip().lower()
    rows = db.loc[mask]
    if rows.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Team '{team_name}' was not found in the database. "
                "Please check the spelling or try a different name."
            ),
        )
    return rows.drop(columns=["team_name"]).mean()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


_LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FIFA 2026 World Cup Predictor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {
      background: #0d1117;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .card {
      background: #161b22;
      border: 1px solid #30363d;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }

    /* ── Select fields ── */
    .select-field {
      width: 100%;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      color: #e6edf3;
      padding: 10px 36px 10px 12px;
      font-size: 0.875rem;
      font-weight: 500;
      appearance: none;
      -webkit-appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%238b949e' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 10px center;
      cursor: pointer;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .select-field:focus {
      outline: none;
      border-color: #58a6ff;
      box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.12);
    }
    .select-field option { background: #161b22; color: #e6edf3; }

    /* ── Predict button ── */
    .btn-predict {
      width: 100%;
      padding: 12px;
      border-radius: 10px;
      border: none;
      background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
      color: #fff;
      font-size: 0.9rem;
      font-weight: 600;
      letter-spacing: 0.03em;
      cursor: pointer;
      transition: background 0.2s, transform 0.15s, box-shadow 0.2s;
    }
    .btn-predict:hover:not(:disabled) {
      background: linear-gradient(135deg, #388bfd 0%, #58a6ff 100%);
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(88, 166, 255, 0.3);
    }
    .btn-predict:active:not(:disabled) { transform: translateY(0); }
    .btn-predict:disabled { opacity: 0.55; cursor: not-allowed; }

    /* ── Progress bars ── */
    .progress-track {
      background: #21262d;
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      width: 0;
      border-radius: inherit;
      transition: width 0.85s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Animations ── */
    .fade-in-up { animation: fadeInUp 0.4s ease forwards; }
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(14px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Live dot ── */
    .live-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #3fb950; display: inline-block;
      animation: blink 2s ease-in-out infinite;
    }
    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.3; }
    }
  </style>
</head>
<body class="min-h-screen flex flex-col items-center px-4 py-14">

  <!-- ── Header ──────────────────────────────────────────────────── -->
  <header class="text-center mb-10 w-full max-w-lg">
    <div style="font-size: 3.5rem; line-height: 1; margin-bottom: 1rem;">&#127942;</div>
    <h1 style="font-size: clamp(1.6rem, 5vw, 2.4rem); font-weight: 800; color: #f0f6fc; letter-spacing: -0.03em; line-height: 1.2;">
      FIFA 2026 World Cup Predictor
    </h1>
    <p class="mt-3 text-sm leading-relaxed mx-auto" style="color: #8b949e; max-width: 400px;">
      ML ensemble of XGBoost, Random Forest &amp; Logistic Regression trained on
      international match history &mdash; forecast head-to-head probabilities for any two nations.
    </p>
    <div class="inline-flex items-center mt-4 px-3 py-1 rounded-full text-xs font-semibold"
         style="background: #0d2b1a; border: 1px solid #238636; color: #3fb950; gap: 6px;">
      <span class="live-dot"></span>Model Active &mdash; v1.0.0
    </div>
  </header>

  <!-- ── Prediction Form Card ────────────────────────────────────── -->
  <div class="card rounded-2xl p-6 w-full max-w-2xl mb-4">
    <p class="text-xs font-bold uppercase tracking-widest mb-6" style="color: #8b949e; letter-spacing: 0.1em;">
      &#9881; Configure Match
    </p>

    <div class="flex flex-col sm:flex-row sm:items-end gap-3">
      <!-- Team A -->
      <div class="flex-1">
        <label class="block text-xs font-bold uppercase tracking-wider mb-2" style="color: #58a6ff; letter-spacing: 0.08em;">
          Team A
        </label>
        <select id="teamA" class="select-field">
          <option value="" disabled selected>Loading&hellip;</option>
        </select>
      </div>

      <!-- VS badge -->
      <div class="flex justify-center flex-shrink-0" style="padding-bottom: 2px;">
        <span class="flex items-center justify-center rounded-lg text-xs font-black"
              style="width:40px;height:38px;background:#21262d;border:1px solid #30363d;color:#8b949e;letter-spacing:0.1em;">
          VS
        </span>
      </div>

      <!-- Team B -->
      <div class="flex-1">
        <label class="block text-xs font-bold uppercase tracking-wider mb-2" style="color: #3fb950; letter-spacing: 0.08em;">
          Team B
        </label>
        <select id="teamB" class="select-field">
          <option value="" disabled selected>Loading&hellip;</option>
        </select>
      </div>
    </div>

    <button id="predictBtn" class="btn-predict mt-6" onclick="analyzeMatch()">
      &#9889;&ensp;Analyze Match Outcome
    </button>
  </div>

  <!-- ── Error Toast ─────────────────────────────────────────────── -->
  <div id="errorToast" class="hidden w-full max-w-2xl mb-4">
    <div class="flex items-center gap-3 px-4 py-3 rounded-xl fade-in-up"
         style="background:#2d1515;border:1px solid #da3633;color:#ff7b72;">
      <span style="font-size:1.1rem;">&#9888;</span>
      <span id="errorMsg" class="text-sm font-medium"></span>
    </div>
  </div>

  <!-- ── Results Panel ───────────────────────────────────────────── -->
  <div id="resultsPanel" class="hidden card rounded-2xl p-6 w-full max-w-2xl">
    <p class="text-xs font-bold uppercase tracking-widest mb-5" style="color: #8b949e; letter-spacing: 0.1em;">
      &#128200; Match Analysis
    </p>
    <h3 id="matchTitle" class="text-lg font-bold mb-7" style="color: #f0f6fc;"></h3>

    <div class="flex flex-col gap-5">

      <!-- Team A Win -->
      <div>
        <div class="flex justify-between items-center mb-2">
          <span id="labelA" class="text-sm font-semibold" style="color: #58a6ff;"></span>
          <span id="pctA"   class="text-sm font-bold" style="color: #58a6ff; font-family: monospace;"></span>
        </div>
        <div class="progress-track">
          <div id="barA" class="progress-fill" style="background: linear-gradient(90deg, #1f6feb, #79c0ff);"></div>
        </div>
      </div>

      <!-- Draw -->
      <div>
        <div class="flex justify-between items-center mb-2">
          <span class="text-sm font-semibold" style="color: #d29922;">Draw</span>
          <span id="pctDraw" class="text-sm font-bold" style="color: #d29922; font-family: monospace;"></span>
        </div>
        <div class="progress-track">
          <div id="barDraw" class="progress-fill" style="background: linear-gradient(90deg, #9e6a03, #e3b341);"></div>
        </div>
      </div>

      <!-- Team B Win -->
      <div>
        <div class="flex justify-between items-center mb-2">
          <span id="labelB" class="text-sm font-semibold" style="color: #3fb950;"></span>
          <span id="pctB"   class="text-sm font-bold" style="color: #3fb950; font-family: monospace;"></span>
        </div>
        <div class="progress-track">
          <div id="barB" class="progress-fill" style="background: linear-gradient(90deg, #238636, #56d364);"></div>
        </div>
      </div>
    </div>

    <!-- Outcome chip -->
    <div class="mt-6 text-center">
      <span id="outcomeChip" class="inline-block px-4 rounded-full text-xs font-bold"
            style="padding-top:6px;padding-bottom:6px;border:1px solid #30363d;background:#21262d;color:#e6edf3;"></span>
    </div>
  </div>

  <!-- ── Footer links ────────────────────────────────────────────── -->
  <div class="flex gap-3 mt-8">
    <a href="/docs" class="px-4 py-2 rounded-lg text-xs font-medium"
       style="background:#21262d;border:1px solid #30363d;color:#c9d1d9;text-decoration:none;transition:border-color 0.2s;"
       onmouseover="this.style.borderColor='#58a6ff'" onmouseout="this.style.borderColor='#30363d'">
      &#128196; Swagger UI
    </a>
    <a href="/redoc" class="px-4 py-2 rounded-lg text-xs font-medium"
       style="background:#21262d;border:1px solid #30363d;color:#c9d1d9;text-decoration:none;transition:border-color 0.2s;"
       onmouseover="this.style.borderColor='#58a6ff'" onmouseout="this.style.borderColor='#30363d'">
      &#128209; ReDoc
    </a>
  </div>

  <footer class="mt-6 text-center" style="color:#484f58;font-size:0.72rem;">
    Ensemble: XGBoost + Random Forest + Logistic Regression &bull; FIFA 2026 &bull; Built with FastAPI
  </footer>

  <!-- ── JavaScript ──────────────────────────────────────────────── -->
  <script>
    // Populate both dropdowns from /api/teams
    async function loadTeams() {
      try {
        const res = await fetch('/api/teams');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const { teams } = await res.json();
        ['teamA', 'teamB'].forEach((id, i) => {
          const sel = document.getElementById(id);
          sel.innerHTML = '<option value="" disabled selected>' +
            (i === 0 ? 'Select Team A…' : 'Select Team B…') + '</option>';
          teams.forEach(name => {
            const o = document.createElement('option');
            o.value = o.textContent = name;
            sel.appendChild(o);
          });
        });
      } catch (e) {
        ['teamA', 'teamB'].forEach(id =>
          (document.getElementById(id).innerHTML =
            '<option value="" disabled selected>Failed to load teams</option>'));
      }
    }

    function showError(msg) {
      const wrap = document.getElementById('errorToast');
      document.getElementById('errorMsg').textContent = msg;
      wrap.classList.remove('hidden');
      clearTimeout(wrap._t);
      wrap._t = setTimeout(() => wrap.classList.add('hidden'), 4500);
    }

    function animateBar(id, pct) {
      const el = document.getElementById(id);
      el.style.transition = 'none';
      el.style.width = '0%';
      // Double rAF forces a reflow so the transition fires
      requestAnimationFrame(() => requestAnimationFrame(() => {
        el.style.transition = 'width 0.85s cubic-bezier(0.4,0,0.2,1)';
        el.style.width = Math.max(pct, 0).toFixed(2) + '%';
      }));
    }

    async function analyzeMatch() {
      const teamA = document.getElementById('teamA').value;
      const teamB = document.getElementById('teamB').value;

      document.getElementById('errorToast').classList.add('hidden');

      if (!teamA || !teamB) {
        showError('Please select both teams before analyzing.');
        return;
      }
      if (teamA === teamB) {
        showError('A team cannot play against itself — please choose two different nations.');
        return;
      }

      // Hide stale results
      const panel = document.getElementById('resultsPanel');
      panel.classList.add('hidden');
      panel.classList.remove('fade-in-up');

      // Loading state
      const btn = document.getElementById('predictBtn');
      btn.disabled = true;
      btn.innerHTML =
        '<span style="display:inline-block;width:13px;height:13px;border:2px solid rgba(255,255,255,0.25);' +
        'border-top-color:#fff;border-radius:50%;animation:spin 0.7s linear infinite;' +
        'vertical-align:-2px;margin-right:8px;"></span>Analyzing…';

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ team_a: teamA, team_b: teamB }),
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: 'Prediction failed.' }));
          showError(err.detail || 'Prediction failed. Please try again.');
          return;
        }

        const d = await res.json();
        const pA    = d.team_a.win_probability * 100;
        const pDraw = d.draw_probability       * 100;
        const pB    = d.team_b.win_probability * 100;

        document.getElementById('matchTitle').textContent = d.match;
        document.getElementById('labelA').textContent     = d.team_a.name + ' Win';
        document.getElementById('labelB').textContent     = d.team_b.name + ' Win';
        document.getElementById('pctA').textContent       = pA.toFixed(1)    + '%';
        document.getElementById('pctDraw').textContent    = pDraw.toFixed(1) + '%';
        document.getElementById('pctB').textContent       = pB.toFixed(1)    + '%';

        // Outcome chip — style changes with likely winner
        const chip = document.getElementById('outcomeChip');
        if (pA >= pB && pA >= pDraw) {
          chip.textContent = '▶ ' + d.team_a.name + ' most likely to win';
          chip.style.cssText = 'display:inline-block;padding:6px 18px;border-radius:999px;font-size:0.78rem;font-weight:700;border:1px solid #1f6feb;background:#0d1f3c;color:#58a6ff;';
        } else if (pB >= pA && pB >= pDraw) {
          chip.textContent = '▶ ' + d.team_b.name + ' most likely to win';
          chip.style.cssText = 'display:inline-block;padding:6px 18px;border-radius:999px;font-size:0.78rem;font-weight:700;border:1px solid #238636;background:#0d2b1a;color:#3fb950;';
        } else {
          chip.textContent = '▶ Match likely to end in a draw';
          chip.style.cssText = 'display:inline-block;padding:6px 18px;border-radius:999px;font-size:0.78rem;font-weight:700;border:1px solid #9e6a03;background:#2b1d0e;color:#e3b341;';
        }

        // Reveal panel then animate bars
        void panel.offsetWidth; // force reflow before re-adding animation class
        panel.classList.remove('hidden');
        panel.classList.add('fade-in-up');

        setTimeout(() => {
          animateBar('barA',    pA);
          animateBar('barDraw', pDraw);
          animateBar('barB',    pB);
        }, 60);

      } catch (e) {
        showError('Network error — is the server running?');
      } finally {
        btn.disabled = false;
        btn.innerHTML = '&#9889;&ensp;Analyze Match Outcome';
      }
    }

    loadTeams();
  </script>

</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=_LANDING_HTML)


@app.get("/api/teams")
def get_teams():
    db: pd.DataFrame = _state["db"]
    return {"teams": sorted(db["team_name"].dropna().unique().tolist())}


@app.post("/api/predict")
def predict(req: MatchRequest):
    vec_a = _lookup(req.team_a)
    vec_b = _lookup(req.team_b)

    X = pd.DataFrame([vec_a, vec_b])
    proba = _state["model"].predict_proba(X)

    # Raw model scores: P(each team is a tournament winner)
    p_a = float(proba[0, 1])
    p_b = float(proba[1, 1])

    # Bradley-Terry normalisation -> two-way win split
    total = p_a + p_b
    raw_a = p_a / total
    raw_b = p_b / total

    # Draw scales with how evenly matched the teams are:
    # balance=1 when perfectly equal, 0 when completely one-sided.
    balance = 1.0 - abs(raw_a - raw_b)
    draw_prob = DRAW_BASE * balance

    # Re-normalise: win_a + win_b + draw_prob == 1 exactly
    win_a = raw_a * (1.0 - draw_prob)
    win_b = raw_b * (1.0 - draw_prob)

    return {
        "match": f"{req.team_a} vs {req.team_b}",
        "team_a": {
            "name": req.team_a,
            "win_probability": round(win_a, 4),
        },
        "draw_probability": round(draw_prob, 4),
        "team_b": {
            "name": req.team_b,
            "win_probability": round(win_b, 4),
        },
    }
