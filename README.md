# Deep Acoustic Intelligence — UWSN CNN-LSTM Localization (Django Major Project)

A Django + Python implementation of the framework from the paper **"Deep
Acoustic Intelligence Using a Hybrid CNN-LSTM Framework for Real-Time
Localization in Mobile Underwater Sensor Networks"**.

It simulates a 3D underwater wireless sensor network (UWSN), generates
noisy acoustic ranging data (TOA / RSSI / Doppler with multipath), and
compares the paper's proposed **hybrid CNN-LSTM AI prediction engine**
against the legacy baselines described in the paper:

- DV-Hop
- TOA (time-of-arrival trilateration)
- RSSI (received-signal-strength trilateration)
- Anchor-based (weighted centroid)
- Random Forest / Gradient Boosting (the paper's own lightweight baselines)
- **Proposed CNN-LSTM** (with a constrained-optimization refinement stage)

Results are shown in a web dashboard with a Table I/II-style comparison
table and interactive charts (mean error, energy, PDR, network lifetime).

---

## 1. Project structure

```
uwsn_project/
├── manage.py
├── requirements.txt
├── uwsn_project/                  # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py / asgi.py
└── localization/                  # Django app
    ├── models.py                  # SimulationRun (stores params + JSON results)
    ├── views.py                   # index / run_simulation / results / history / about
    ├── urls.py
    ├── admin.py
    ├── management/commands/
    │   └── run_demo_simulation.py # CLI: run a simulation without the web UI
    ├── simulation/                # <-- the actual AI / algorithm code
    │   ├── network.py             # 3D deployment + ocean-current mobility model
    │   ├── channel.py             # acoustic propagation: TOA / RSSI / Doppler + noise
    │   ├── baselines.py           # DV-Hop, TOA, RSSI, Anchor-based
    │   ├── ai_models.py           # CNN-LSTM (Keras) + Random Forest / Gradient Boosting
    │   ├── optimizer.py           # localization optimization / error-correction stage
    │   ├── metrics.py             # mean error, energy, PDR, lifetime, delay, throughput
    │   └── pipeline.py            # orchestrates the full 8-stage architecture (Fig. 1)
    ├── templates/localization/    # Bootstrap-free custom dashboard UI
    └── static/localization/css/   # dashboard styling
```

## 2. Setup

Requires **Python 3.10+**.

```bash
cd uwsn_project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note on TensorFlow:** `tensorflow` is a fairly large install. If you
> only want to demo the pipeline quickly (baselines + Random
> Forest/Gradient Boosting + a deterministic fallback predictor standing
> in for the CNN-LSTM), you can skip it:
> ```bash
> pip install Django numpy scipy scikit-learn
> ```
> The project detects whether TensorFlow is available (`TF_AVAILABLE` in
> `ai_models.py`) and automatically falls back to a lightweight predictor
> if it isn't installed, so the whole app still runs end-to-end. Install
> `tensorflow` to train and run the **real** CNN-LSTM model.

```bash
python manage.py migrate
python manage.py createsuperuser     # optional, for /admin/
python manage.py runserver
```

Then open **http://127.0.0.1:8000/** in your browser.

## 3. Using the app

1. On the home page, set the simulation parameters (number of nodes,
   anchors, mobility timesteps, channel noise level, drift speed range,
   CNN-LSTM training epochs) and click **Run Simulation**.
2. The results page shows:
   - A Table I/II-style comparison across all six methods
   - Bar charts for mean localization error, energy consumption, PDR,
     and network lifetime
3. Past runs are saved to the database and browsable under **History**.
4. The **About the Paper** page summarizes the source paper's problem
   statement, proposed architecture, and reported figures.

### Command-line demo (no web server needed)

```bash
python manage.py run_demo_simulation --nodes 60 --anchors 15 --timesteps 40 --epochs 10
```

## 4. How each pipeline stage maps to the paper (Fig. 1 / Fig. 2)

| Paper stage | Code |
|---|---|
| Mobile Underwater Node | `simulation/network.py` (`UnderwaterNetwork`, `Node`) |
| Acoustic Signal Collection | `simulation/channel.py` (`generate_ranging_snapshot`) |
| Noise Reduction | `simulation/pipeline.py` (`_bandpass_denoise`) |
| Feature Extraction | `simulation/pipeline.py` (`_extract_feature_vector`) |
| AI Prediction Engine (CNN-LSTM) | `simulation/ai_models.py` (`build_cnn_lstm_model`) |
| Localization Optimization | `simulation/optimizer.py` (`refine_estimate` / `batch_refine`) |
| Position Estimation / Final Coordinates | `simulation/pipeline.py` (`run_full_simulation`) |

## 5. Extending the project (ideas for your report / viva)

These map directly to the paper's **Future Work** (Section VII), so they
also work well as "future scope" talking points:

- Swap the random-drift mobility model for a Deep Reinforcement Learning
  trajectory planner.
- Add a Federated Learning mode so multiple simulated deployments train a
  shared CNN-LSTM without sharing raw ranging data.
- Add a "Digital Twin" replay view comparing predicted vs. true trajectories
  frame-by-frame.
- Port the CNN-LSTM inference path to run without TensorFlow (e.g. a
  distilled/quantized model) to demonstrate Edge-AI feasibility on
  resource-constrained nodes.
- Add AUV-assisted adaptive anchor placement as an alternative network
  topology option in `network.py`.

## 6. Important note on results

The absolute numbers you'll see (mean error in meters, energy in mJ, etc.)
come from **this project's own simulator**, which reimplements the
paper's described approach but is not the original authors' codebase (not
published). The paper itself notes its reported figures are from
simulation, not a physical sea trial, and are "an optimistic scenario."
Treat the comparison as illustrative of the *relative* behavior described
in the paper — the CNN-LSTM/optimization pipeline outperforming
distance-, connectivity-, and even tree-ensemble baselines, especially on
energy consumption and network lifetime — rather than an exact
reproduction of Table I/II. Simulation parameters (noise level, anchor
density, drift speed) are all exposed on the home page so you can tune
them and discuss sensitivity in your report.

## 7. Citation

Based on: Shanthini, K.S., Kiruba Sri, S., Dhakshinamoorthy, P.A., Rahul,
V., Arunprasaath, V.E., *"Deep Acoustic Intelligence Using a Hybrid
CNN-LSTM Framework for Real-Time Localization in Mobile Underwater Sensor
Networks"*, KIT-Kalaignarkarunanidhi Institute of Technology, Coimbatore.

## Modification Log (Language/Methodology honesty fix)

`localization/simulation/metrics.py` originally contained hardcoded energy/
delay constants copied from the paper's Table I/II with random jitter added
to disguise them as computed values. This has been replaced with:

- **Energy**: standard first-order WSN radio energy model (Heinzelman et
  al., 2000), computed from the real number of anchor messages used and
  their real distances.
- **Delay**: real acoustic propagation time (distance / 1500 m/s) plus a
  fixed processing delay.
- **PDR**: real ratio of anchors used vs. anchors visible in range.

`pipeline.py` now tracks which real anchor distances each method actually
used, instead of passing nothing into the metrics functions.

### Known open issues (need your decision before Monday)
1. **Anchor-based method's ~160-210m error is real, not a bug.** A naive
   weighted-centroid estimator is known in the literature to perform
   poorly in large, anchor-sparse 3D deployments -- this matches that
   expectation. You can (a) report this honestly as a finding, or (b)
   upgrade Anchor-based to a proper multilateration method for a fairer
   comparison. Talk to your guide about which is expected for your paper.
2. **PDR for the AI-based methods reads oddly low (~9-10%)** because they
   intentionally use only 1 fresh anchor message per update (by design,
   for energy savings) rather than failing to deliver 90% of packets. The
   current PDR formula conflates "used fewer messages on purpose" with
   "packet loss." This metric needs a clearer definition before it goes in
   the paper -- flag this to your guide.
3. **CNN-LSTM is NOT yet trained for real** -- TensorFlow isn't installed
   in this environment. Until you install it and run this on your own
   machine, `is_placeholder_untrained_model: True` will appear in its
   result, and those numbers must not be used.
