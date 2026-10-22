# Project page storyboard

The page tells the paper's introduction as a story on one task (retail/103 from Tau2-Bench), fifty attempts at a time.
Approved 2026-10-21. The acts follow the intro paragraph by paragraph.

## Thesis

> We set out to learn a representation that captures the critical behavioral patterns of agents on long-horizon, tool-calling tasks.

## The spine

| Intro paragraph | Act | Headline | Visual | Number | Hinge to the next act |
|---|---|---|---|---|---|
| 1. Trajectories are long; decisive tokens are few | **0 · Thesis** | The thesis sentence. | Hero loop: one attempt with a rising token counter | 100K tokens | "Start with one attempt." |
| 1 | **1 · One attempt** | Twenty steps, a hundred thousand tokens, and one decision that sealed it. | One failing rollout; rewind, the shared-prefix passing rollout branches off at the critical branch point | the branch step | "Could we have seen it coming?" |
| 3. Pairs recur as states | **2 · The map** | Rewinding every step is exponential. But actions and observations repeat, so fifty attempts draw one map. | The schematic re-sampling tree (3^8); fifty independent lanes; tuples snap into states; the DAG accrues one rollout at a time, edges widen and tint, branch points surface | 6,561 rollouts | "If the map keeps the outcome, a simple classifier should find it." |
| 4. Outcome is recoverable | **3 · The finding** | It does. Successes and failures separate along the path. | Routes propagating through the map; Table 1 heatmap; look inside: the Fig. 1 explorer | 0.95 AUROC | "Every task, though, needs its own classifier." |
| 5. Per-task fitting transfers nothing | **4 · The obstacle** | Thirty labeled rollouts per task, and nothing carries over. | Baseline AUROC vs labeled rollouts per class (1 to 33), lock on task change | 33 rollouts | "So learn the representation instead." |
| 6a. Precog-RM | **5 · The idea** | One encoder, meta-learned across tasks. A rollout becomes a point; a batch of labeled rollouts becomes two prototypes. | Architecture pass with zoom-ins (embed, project, encode, pool, z); sixteen labeled rollouts fed through; prototypes; look inside: the latent map | 128-d | "How few rollouts? How far does it carry?" |
| 6b. Few-shot, transfer, prefix | **6 · The proof** | Two per class. Unseen policies. Mid-rollout. | Figure 3; Tables 2 and 13; a new rollout scored step by step with the Figure 4 frontier; look inside: the prefix slider | k = 2 | "What is it good for?" |
| 6c. GRPO | **7 · The payoff** | Stop the rollouts that are already decided. 47% fewer training FLOPs. | One GRPO group of sixteen with the controller; Table 3 frontier; look inside: Figure 5 | 47.3% | "Back to the thesis." |
| Contributions | **8 · Return** | A representation that keeps the outcome, and carries it. | The task picker: the whole story replays on a new task; links; citation | — | — |

## How it tells rather than explains

1. **One cast, one stage.** The same task, agent sprite, fifty rollouts and sixteen-rollout batch appear in every act. The map built in act 2 is the stage for act 3; Precog-RM's latent panel in act 5 is "the map, learned", drawn where the map stood.
2. **Hinge lines.** Each act ends on a short italic question; the next headline answers it.
3. **Captions that land.** Every loop ends in a two-second hold, and the act's conclusion fades in under the visual at that moment.
4. **One number per act, big.** Nothing else in the number strip.
5. **The camera moves in one direction.** Act 1 zooms into a single step; act 2 pulls out to fifty rollouts and the map; act 5 zooms into the model and back; act 6 pulls out to nine cells, three benchmarks, three policies; act 7 to a GRPO run. Zoom level equals scope.
6. **Progress is a trajectory.** A thin lane across the top of the page; an agent sprite slides along it as you scroll and lands on a green check at the end.
7. **Constant colour grammar.** Blue action, orange observation, green pass, red fail, grey unknown, blue ring for a branch point. Legend shown once, in act 1.
8. **Sidebars, not sections.** "Look inside" panels stay collapsed; the story never depends on opening one.
9. **The ending replays.** Act 8's task picker reloads the whole story on a task the reader chose.

## Text budget

Acts 0, 2, 4 and 5 may carry a second paragraph of three or four sentences, in the intro's own words: the medium, why states, why per-task fitting stalls, what "no fitting at inference" buys. Acts 1, 3, 6 and 7 stay at one line plus the number.

## Decisions taken

- Hero visual: the single attempt with a rising token counter. The fifty lanes belong to act 2.
- The exponential re-sampling tree lives inside act 2 as the reason for states, not as its own interlude.
- The branch in act 1 uses two real rollouts that share their first states (the pair with the longest shared state prefix that still falls inside the drawn steps), captioned as such; the tree is the only schematic visual.
- The DAG in act 2 uses all fifty rollouts; the model acts use the sixteen-rollout batch.

## Where things live

- `docs/opening.js` — acts 1–3 (scenes a1, a2, a3), computed from the loaded task bundle; act 3's closing chart reads `docs/results.json`.
- `docs/pipeline.js` — the model scene; `story.js` renders slices of its timeline as loops.
- `docs/story.js` — loop manager, GRPO loop, panel redraws, rail.
- `docs/charts.js` + `docs/results.json` — result figures from the shipped results (`scripts/build_web_results.py`).
- `docs/app.js` — the collapsed interactives (Fig. 1 explorer, latent map, prefix slider).
- `scripts/build_web_bundles.py` — per-task bundles, served from the dataset repo's `web/` folder.
