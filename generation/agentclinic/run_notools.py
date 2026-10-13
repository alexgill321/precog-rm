#!/usr/bin/env python3
"""No-tools ablation: MedQA_Ext with the REQUEST TEST capability removed from the
doctor prompt. Mirrors notebooks/agentclinic_notools_ablation.ipynb sections 3+5
exactly (audited against agentclinic._run_single; only deliberate deviations are
the prompt and the test-refusal branch). Resumable; same file naming as harness.

Usage:  .venv/bin/python run_notools.py --doctor deepseek --n-rollouts 10
"""
import argparse, json, os, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agentclinic as ac

DOCTORS = {
    "gpt5mini": "gpt-5-mini",
    "deepseek": "vertex-deepseek-v3.2",
    "qwen": "vertex-qwen3-80b",
    "terra": "gpt-5.6-terra",
    "deepseek_or": "openrouter-deepseek",   # OpenRouter fallback (Vertex MaaS congested)
    "qwen_or": "openrouter-qwen",
}
PATIENT_LLM = MEASUREMENT_LLM = MODERATOR_LLM = "gpt-4o-mini"
DATASET, TOTAL_INFERENCES = "MedQA_Ext", 20

NO_TOOLS_PROMPT = (
    'You are a doctor named Dr. Agent who only responds in the form of dialogue. '
    'You are inspecting a patient who you will ask questions in order to understand '
    'their disease. You are only allowed to ask {max_infs} questions total before you '
    'must make a decision. You have asked {infs} questions so far. You cannot order '
    'any tests, imaging, or physical examinations -- you may only talk to the patient. '
    'Your dialogue will only be 1-3 sentences in length. Once you have decided to '
    'make a diagnosis please type "DIAGNOSIS READY: [diagnosis here]"'
    '\n\nBelow is all of the information you have. {presentation}. \n\n Remember, you '
    'must discover their disease by talking to the patient only. No tests or exams '
    'are available.'
)
REFUSAL_MSG = ("RESULTS: No tests or examinations are available in this setting. "
               "Please continue the interview with the patient or provide your diagnosis.")


class PromptedDoctor(ac.DoctorAgent):
    def __init__(self, template, **kw):
        self.template = template
        super().__init__(**kw)

    def system_prompt(self):
        return self.template.format(max_infs=self.MAX_INFS, infs=self.infs,
                                    presentation=self.presentation)


def run_rollout(loader, scenario_id, rollout_id, doctor_llm, out_dir):
    out_path = Path(out_dir) / f"scenario_{scenario_id:04d}_rollout_{rollout_id:02d}.json"
    if out_path.exists():
        return None
    scenario = loader.get_scenario(scenario_id)
    doctor = PromptedDoctor(NO_TOOLS_PROMPT, scenario=scenario, backend_str=doctor_llm,
                            max_infs=TOTAL_INFERENCES)
    patient = ac.PatientAgent(scenario=scenario, backend_str=PATIENT_LLM)
    ac._pop_thread_tokens()

    trajectory, pi_dialogue, correctness, tool_attempts = [], "", False, 0
    for inf_id in range(TOTAL_INFERENCES):
        if inf_id == TOTAL_INFERENCES - 1:
            pi_dialogue += "This is the final question. Please provide a diagnosis.\n"
        ac._pop_last_reasoning()
        dd = doctor.inference_doctor(pi_dialogue)
        doctor_turn = {"role": "doctor", "content": dd}
        reasoning = ac._pop_last_reasoning()
        if reasoning:
            doctor_turn["reasoning"] = reasoning
        trajectory.append(doctor_turn)

        if "diagnosis ready" in dd.lower():
            correctness = ac.compare_results(dd, scenario.diagnosis_information(),
                                             MODERATOR_LLM, None) == "yes"
            break

        if "request test" in dd.lower():
            tool_attempts += 1
            pi_dialogue = REFUSAL_MSG
            trajectory.append({"role": "measurement", "content": pi_dialogue})
            patient.add_hist(pi_dialogue)
            time.sleep(0.5)
            continue

        pi_dialogue = patient.inference_patient(dd)
        trajectory.append({"role": "patient", "content": pi_dialogue})
        time.sleep(0.5)

    tokens = ac._pop_thread_tokens()
    record = {
        "scenario_id": scenario_id, "rollout_id": rollout_id, "dataset": DATASET,
        "doctor_llm": doctor_llm, "correct": correctness,
        "correct_diagnosis": str(scenario.diagnosis_information()),
        "tokens": tokens, "cost_usd": round(ac._compute_cost(tokens), 6),
        "history": trajectory,
        "prompt_variant": "no_tools", "allow_tests": False,
        "on_test_request": "refuse", "tool_attempts": tool_attempts,
    }
    with open(out_path, "w") as f:
        json.dump(record, f)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--doctor", choices=list(DOCTORS), required=True)
    p.add_argument("--n-rollouts", type=int, default=10)
    p.add_argument("--tasks", type=int, nargs=2, default=[0, 20], metavar=("LO", "HI"))
    p.add_argument("--task-ids", type=str, default=None,
                   help="comma-separated scenario ids; overrides --tasks range")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--out-dir", type=str, default=None,
                   help="override output dir (default: medqa-ext-<doctor>-notools-<n>rollouts)")
    args = p.parse_args()

    doctor_llm = DOCTORS[args.doctor]
    out_dir = (Path(args.out_dir) if args.out_dir else
               Path("outputs") / f"medqa-ext-{args.doctor}-notools-{args.n_rollouts}rollouts")
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = ac.ScenarioLoaderMedQAExtended()

    task_iter = ([int(x) for x in args.task_ids.split(",") if x.strip()]
                 if args.task_ids else list(range(*args.tasks)))
    jobs = [(s, r) for s in task_iter for r in range(args.n_rollouts)]
    done = sum((out_dir / f"scenario_{s:04d}_rollout_{r:02d}.json").exists() for s, r in jobs)
    print(f"no-tools | doctor={doctor_llm} -> {out_dir}", flush=True)
    print(f"{len(jobs)} rollouts, {done} already done, concurrency={args.concurrency}", flush=True)

    correct, cost, fails = 0, 0.0, 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_rollout, loader, s, r, doctor_llm, out_dir): (s, r)
                for s, r in jobs}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                fails += 1
                print(f"\n{futs[fut]} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
                continue
            if rec is None:
                continue
            done += 1
            correct += rec["correct"]
            cost += rec["cost_usd"]
            print(f"  {done}/{len(jobs)} | {correct} correct of new | ${cost:.3f}", flush=True)
    print(f"\ndone: {done}/{len(jobs)} complete, {fails} failed, new-rollout cost ${cost:.4f}", flush=True)


if __name__ == "__main__":
    main()
