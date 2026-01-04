# 🏸 ShuttleSense — Real-Time Badminton AI Coach

> A real-time, low-latency AI badminton coach that watches your training and gives **instant corrective feedback** on footwork and technique — like a coach standing beside you.

---

## 🚀 Why ShuttleSense Exists

Badminton players don’t fail because they lack effort.  
They fail because they **repeat mistakes without feedback**.

- Coaches aren’t always available
- Video review is slow and confusing
- YouTube advice is generic
- Small timing & posture errors compound into bad habits

**ShuttleSense fixes this by correcting you _while you practice_, not after.**

---

## 🎯 Product Philosophy

ShuttleSense is built on 5 non-negotiable principles:

1. **Low latency beats intelligence**
2. **One correction is better than ten insights**
3. **Rules before machine learning**
4. **Real improvement over flashy demos**
5. **Depth over feature count**

This is a **training partner**, not a highlight generator.

---

## 🧠 What ShuttleSense Is (and Is Not)

### ✅ What It Is
- A real-time training coach
- A technique correction system
- A personal learning engine that improves over time

### ❌ What It Is Not
- Match analytics tool
- Shuttle tracking system (yet)
- Social or multiplayer app
- Generic “AI tips” chatbot

---

## 📌 V1 Scope (Current Focus)

### ✅ Included
- Real-time coaching for **shadow footwork drills**
- On-device pose estimation (no cloud latency)
- Audio + visual corrective cues
- Continuous technique score
- Local session history

### ❌ Not Included (by design)
- Live rally analysis
- Shuttle tracking
- Opponent detection
- Stroke contact analysis
- Social / leaderboard features

---

## 🏃 Supported Drills (V1)

Users must select a drill before starting (important for accuracy):

- 6-corner shadow footwork
- Side-to-side defensive movement
- Front-back movement

---

## 📷 Camera & Setup Requirements

- **Camera angle:** Behind view (preferred) or 45° back-side
- **Visibility:** Full body (head → feet)
- **Device:** Phone on tripod (static)
- **Environment:** Indoor court or home space
- **Frame rate:** ≥ 30 FPS preferred

---

## ⚙️ Architecture Overview (Hybrid)

### On-Device (Real-Time)
- MediaPipe Pose / MoveNet
- 20–30 FPS pose inference
- Rule-based event detection
- Audio feedback engine
- Target latency: **<150 ms**

### Cloud (Google Cloud – V2+)
- Pose/video upload
- Cloud Run / Vertex AI
- Deep post-session analysis
- Long-term learning & personalization

---

## 🧩 Real-Time Event Detection (V1)

Detected continuously during sessions:

- Split step / hop
- Push-off initiation
- Direction change
- Lunge phase
- Recovery to base

---

## 🧠 Coaching Rules — “Coach Brain” (V1)

> Only **ONE correction at a time**  
> Cue cooldown: ~1.5 seconds

### Split Step
- Missing → “Split step”
- Late → “Split earlier”

### Stance & Balance
- Stance too narrow → “Wider base”
- Upright posture → “Lower stance”

### Movement
- Delayed first step → “Explode sooner”
- Knee collapsing inward → “Knee out”

### Recovery
- No base recovery → “Back to base”
- Slow recovery → “Recover faster”
- Balance instability → “Control landing”

---

## 🔊 Feedback Modalities

### Audio (Primary)
- Short spoken cues: “Split”, “Wider”, “Recover”
- Beep patterns for timing errors

### Visual (Secondary)
- Skeleton overlay
- Highlighted joints on error
- Base zone marker
- Green / red timing indicators

---

## 📊 Scoring System

**Live Technique Score (0–100)**  
Computed from:
- Split step timing consistency
- Stance width stability
- Recovery speed
- Balance control

Displayed smoothly (no jitter).

---

## 🧪 Session Flow

### Start
1. Select drill
2. Camera visibility check
3. Countdown (3-2-1)

### During
- Live skeleton overlay
- Audio corrections
- Real-time score

### End
- Final score
- Top 3 recurring mistakes
- Suggested drills

---

## 💾 Data Storage (V1)

Stored locally on device:
- Session metadata
- Aggregate scores
- Error frequency
- Drill type

> No cloud sync required for V1.

---

## ☁️ Google Cloud Usage (V2+)

Best use of Google Cloud credits:

- Multi-rep temporal analysis
- Consistency & fatigue detection
- Comparison vs reference movement patterns
- Longitudinal player learning

**Important:**  
Store **pose data**, not raw video, for efficiency and privacy.

---

## 🧪 Open-Source Research Review

### CoachAI-Projects (GitHub)
- Focus: rally analytics, stroke forecasting, simulation
- Research-grade, not real-time coaching
- Useful later for:
  - movement pattern learning
  - rally prediction research

**Conclusion:**  
Not usable directly for V1 real-time coaching.  
Used only for inspiration and datasets.

---

## 🛣️ Roadmap

### Phase 1 — Real-Time Coach (Weeks 1–2)
- On-device footwork coach
- Pose + rule engine
- Local storage
- Session summaries

### Phase 2 — Cloud Intelligence (Weeks 3–4)
- Post-session reports
- Consistency metrics
- Fatigue detection

### Phase 3 — Personal Badminton Model (Weeks 5–8)
- Reference comparison
- Personal weakness detection
- Injury risk signals
- Coach review mode

---

## ✅ Success Criteria

V1 is successful if:
- Players correct mistakes **within the same session**
- Feedback feels accurate >70% of the time
- Players want to use it **every practice**
- Improvement is visible in weeks, not months

---

## 🧭 Final Truth

You are not building:

> “AI that tells you what you did wrong.”

You are building:

> **A system that understands how YOU play badminton and helps you improve faster than training alone.**

That is rare.  
That is worth building.

---

## 📌 Next Steps
- [ ] Threshold values for coaching rules  
- [ ] Google Cloud architecture + cost  
- [ ] Opus build task breakdown  
- [ ] UI wireframes  

---

**Built with focus. Built for improvement.**
