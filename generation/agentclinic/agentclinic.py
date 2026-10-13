import argparse
import anthropic
import openai, re, random, time, json, os
import threading
import requests
from datetime import timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Vertex MaaS auth (mirrors src/bfcl_wrap/vertex_bfcl_handler.py)
# ---------------------------------------------------------------------------

_vertex_credentials_cache = None
_vertex_token_lock = threading.Lock()
_vertex_token_cache: dict = {"token": None, "expiry": 0.0}

_VERTEX_MODELS = {
    "vertex-qwen3-80b": {
        "model_id": "qwen/qwen3-next-80b-a3b-thinking-maas",
        "location": "global",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 32768,
        "strip_harmony": False,
    },
    "vertex-deepseek-v3.2": {
        "model_id": "deepseek-ai/deepseek-v3.2-maas",
        "location": "global",
        "temperature": 0.8,
        "top_p": None,
        "max_tokens": None,
        "strip_harmony": True,
    },
}

_HARMONY_TAG_RE = re.compile(
    r"<\|(?:start|end|channel|message|call|return|constrain)\|>"
)

# OpenRouter-served doctor models (alternative to Vertex when its quota is
# exhausted). Model ids are env-overridable so exact OpenRouter slugs can be
# set without editing code. Routed through the shared OpenRouter client.
_OPENROUTER_DOCTORS = {
    # ids/prices verified against openrouter.ai/api/v1/models (2026-07). deepseek
    # non-exp v3.2 chosen to match the Vertex deepseek-v3.2-maas baseline.
    "openrouter-deepseek": {"env": "OR_DEEPSEEK_MODEL",
                            "default": "deepseek/deepseek-v3.2",
                            "temperature": 0.8, "top_p": None},
    "openrouter-qwen": {"env": "OR_QWEN_MODEL",
                        "default": "qwen/qwen3-next-80b-a3b-thinking",
                        "temperature": 0.6, "top_p": 0.95},
}


def _strip_harmony_tags(text: str) -> str:
    m = _HARMONY_TAG_RE.search(text)
    return text[: m.start()].strip() if m else text


def _get_vertex_access_token() -> str:
    import google.auth
    import google.auth.transport.requests

    global _vertex_credentials_cache

    with _vertex_token_lock:
        if _vertex_token_cache["token"] and time.time() < _vertex_token_cache["expiry"]:
            return _vertex_token_cache["token"]

        if _vertex_credentials_cache is None:
            _vertex_credentials_cache, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        request = google.auth.transport.requests.Request()
        _vertex_credentials_cache.refresh(request)

        # credentials.expiry is a naive datetime in UTC; timestamp() would
        # misread it as local time, so attach UTC explicitly.
        expiry = getattr(_vertex_credentials_cache, "expiry", None)
        if expiry is not None:
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            _vertex_token_cache["expiry"] = expiry.timestamp() - 60
        else:
            _vertex_token_cache["expiry"] = time.time() + 3000
        _vertex_token_cache["token"] = _vertex_credentials_cache.token
        return _vertex_token_cache["token"]


def _get_vertex_project_id() -> str:
    project_id = os.getenv("GCP_PROJECT_ID")
    if project_id:
        return project_id
    try:
        import google.auth
        _, project = google.auth.default()
        if project:
            return project
    except Exception:
        pass
    raise ValueError("Set GCP_PROJECT_ID env var or configure gcloud default project")


def _query_vertex(model_str: str, prompt: str, system_prompt: str, timeout: int = 600) -> str:
    cfg = _VERTEX_MODELS[model_str]
    project_id = _get_vertex_project_id()
    location = cfg["location"]
    if location == "global":
        url = (
            f"https://aiplatform.googleapis.com/v1/"
            f"projects/{project_id}/locations/global/endpoints/openapi/chat/completions"
        )
    else:
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/"
            f"projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    body = {
        "model": cfg["model_id"],
        "messages": messages,
        "temperature": cfg["temperature"],
    }
    if cfg["max_tokens"] is not None:
        body["max_tokens"] = cfg["max_tokens"]
    if cfg["top_p"] is not None:
        body["top_p"] = cfg["top_p"]

    for attempt in range(3):
        try:
            token = _get_vertex_access_token()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }
            resp = requests.post(url, json=body, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            if "choices" not in data:
                raise Exception(f"Unexpected response (no choices): {json.dumps(data)[:500]}")
            usage = data.get("usage", {})
            _accumulate_tokens(model_str, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            # Stash reasoning (thinking models return it in a separate field).
            # It is NOT part of the returned content — so it never enters any
            # agent's dialogue history — but _run_single pops it and saves it
            # on the doctor turn in the rollout file.
            _thread_local.last_reasoning = data["choices"][0]["message"].get("reasoning_content")
            content = (data["choices"][0]["message"].get("content") or "").strip()
            if cfg["strip_harmony"]:
                content = _strip_harmony_tags(content)
            return content
        except Exception as e:
            is_throttle = "throttled" in str(e).lower() or "429" in str(e)
            wait = (2 ** attempt) * (10 if is_throttle else 2)
            print(f"Vertex warning (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(wait)

    raise Exception(f"_query_vertex: all 3 attempts failed for {model_str}")


# ---------------------------------------------------------------------------
# Token tracking (thread-local, per-model)
# ---------------------------------------------------------------------------

# Pricing per million tokens: {model_str: (input_$/M, output_$/M)}
_MODEL_PRICING = {
    "vertex-deepseek-v3.2": (0.56, 1.68),
    "vertex-qwen3-80b":     (0.15, 1.20),
    "gpt-5-mini":           (0.25, 2.00),
    "gpt-5.6-terra":        (2.50, 15.00),
    "openrouter-deepseek":  (0.269, 0.400),   # deepseek/deepseek-v3.2 (verified)
    "openrouter-qwen":      (0.098, 0.780),   # qwen/qwen3-next-80b-a3b-thinking (verified)
    "gpt-4o-mini":          (0.15, 0.60),
    "gpt4o":                (2.50, 10.00),
}

_thread_local = threading.local()


def _accumulate_tokens(model_str: str, prompt_tokens: int, completion_tokens: int):
    if not hasattr(_thread_local, "tokens"):
        _thread_local.tokens = {}
    entry = _thread_local.tokens.setdefault(model_str, {"prompt": 0, "completion": 0})
    entry["prompt"] += prompt_tokens
    entry["completion"] += completion_tokens


def _pop_thread_tokens() -> dict:
    tokens = getattr(_thread_local, "tokens", {})
    _thread_local.tokens = {}
    return tokens


def _pop_last_reasoning():
    reasoning = getattr(_thread_local, "last_reasoning", None)
    _thread_local.last_reasoning = None
    return reasoning


def _compute_cost(tokens_by_model: dict) -> float:
    total = 0.0
    for model, counts in tokens_by_model.items():
        inp, out = _MODEL_PRICING.get(model, (0.0, 0.0))
        total += counts["prompt"] / 1e6 * inp
        total += counts["completion"] / 1e6 * out
    return total


# ---------------------------------------------------------------------------
# OpenAI client (lazy init, new v1 API)
# ---------------------------------------------------------------------------

_openai_client = None
_openai_direct_client = None
_openai_client_lock = threading.Lock()


def _get_openai_client():
    """Shared client for the gpt-4o-mini support roles (patient/measurement/
    moderator). Prefers OpenRouter whenever OPENROUTER_API_KEY is set — the
    model ids used with it are OpenRouter-style (e.g. ``openai/gpt-4o-mini``) —
    even if OPENAI_API_KEY is also set (needed when the doctor is a gpt-5
    model going direct to api.openai.com)."""
    global _openai_client
    with _openai_client_lock:
        if _openai_client is None:
            if os.environ.get("OPENROUTER_API_KEY"):
                api_key = os.environ["OPENROUTER_API_KEY"]
                base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
            else:
                api_key = os.environ.get("OPENAI_API_KEY", "")
                base_url = os.environ.get("OPENAI_BASE_URL")
            _openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return _openai_client


def _get_openai_direct_client():
    """Client pinned to api.openai.com (for gpt-5 doctor models), regardless
    of any OpenRouter env configuration."""
    global _openai_direct_client
    with _openai_client_lock:
        if _openai_direct_client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY must be set for gpt-5 models")
            _openai_direct_client = openai.OpenAI(api_key=api_key)
    return _openai_direct_client


# ---------------------------------------------------------------------------
# Model inference dispatcher
# ---------------------------------------------------------------------------

_ALL_MODELS = [
    "gpt4", "gpt3.5", "gpt4o", "llama-2-70b-chat", "mixtral-8x7b",
    "gpt-4o-mini", "llama-3-70b-instruct", "gpt4v", "claude3.5sonnet", "o1-preview",
    "gpt-5-mini", "gpt-5.6-terra",
] + list(_VERTEX_MODELS.keys()) + list(_OPENROUTER_DOCTORS.keys())


def query_model(model_str, prompt, system_prompt, tries=30, timeout=20.0,
                image_requested=False, scene=None, max_prompt_len=2**14, clip_prompt=False):
    if model_str not in _ALL_MODELS and "_HF" not in model_str:
        raise Exception("No model by the name {}".format(model_str))
    for _ in range(tries):
        if clip_prompt:
            prompt = prompt[:max_prompt_len]
        try:
            if model_str in _VERTEX_MODELS:
                return _query_vertex(model_str, prompt, system_prompt, timeout=600)

            client = _get_openai_client()

            if image_requested and scene is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "{}".format(scene.image_url)}},
                    ]},
                ]
                if model_str in ("gpt4v", "gpt4", "gpt4o", "gpt-4o-mini"):
                    model_map = {
                        "gpt4v": "gpt-4-vision-preview",
                        "gpt4": "gpt-4-turbo",
                        "gpt4o": "openai/gpt-4o",
                        "gpt-4o-mini": "openai/gpt-4o-mini",
                    }
                    response = client.chat.completions.create(
                        model=model_map[model_str],
                        messages=messages,
                        temperature=0.05,
                        max_tokens=200,
                    )
                    if response.usage:
                        _accumulate_tokens(model_str, response.usage.prompt_tokens, response.usage.completion_tokens)
                    return response.choices[0].message.content

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            if model_str == "gpt4":
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
            elif model_str == "gpt4v":
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
            elif model_str == "gpt-4o-mini":
                # OpenRouter needs the "openai/" prefix; the OpenAI API wants
                # the bare id. Route to whichever endpoint is configured.
                if os.environ.get("OPENROUTER_API_KEY"):
                    mm_client, mm_model = client, "openai/gpt-4o-mini"
                else:
                    mm_client, mm_model = _get_openai_direct_client(), "gpt-4o-mini"
                response = mm_client.chat.completions.create(
                    model=mm_model,
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
            elif model_str == "gpt-5-mini":
                # gpt-5 reasoning model via api.openai.com directly: minimal
                # reasoning effort; rejects custom temperature; no max token
                # cap (reasoning tokens would count against it).
                response = _get_openai_direct_client().chat.completions.create(
                    model="gpt-5-mini-2025-08-07",
                    messages=messages,
                    reasoning_effort="minimal",
                )
            elif model_str == "gpt-5.6-terra":
                # frontier reasoning model via api.openai.com directly; low
                # reasoning effort (lets it actually reason, unlike the
                # gpt-5-mini minimal setting).
                response = _get_openai_direct_client().chat.completions.create(
                    model="gpt-5.6-terra",
                    messages=messages,
                    reasoning_effort="low",
                )
            elif model_str in _OPENROUTER_DOCTORS:
                # DeepSeek / Qwen served via OpenRouter (fallback for Vertex MaaS
                # congestion). Uses the shared OpenRouter client. Reasoning (for
                # the qwen thinking model) is captured off-band, same as Vertex.
                cfg = _OPENROUTER_DOCTORS[model_str]
                mid = os.environ.get(cfg["env"], cfg["default"])
                create_kw = {"model": mid, "messages": messages,
                             "temperature": cfg["temperature"]}
                if cfg["top_p"] is not None:
                    create_kw["top_p"] = cfg["top_p"]
                response = client.chat.completions.create(**create_kw)
                _m = response.choices[0].message
                _r = getattr(_m, "reasoning", None) or getattr(_m, "reasoning_content", None)
                if _r:
                    _thread_local.last_reasoning = _r
            elif model_str == "o1-preview":
                response = client.chat.completions.create(
                    model="o1-preview-2024-09-12",
                    messages=[{"role": "user", "content": system_prompt + prompt}],
                )
            elif model_str == "gpt3.5":
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
            elif model_str == "gpt4o":
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
            elif model_str == "claude3.5sonnet":
                anth_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = anth_client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                return json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "llama-2-70b-chat":
                import replicate
                llama2_url = "meta/llama-2-70b-chat"
                output = replicate.run(llama2_url, input={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "max_new_tokens": 200,
                })
                return re.sub(r"\s+", " ", "".join(output))
            elif model_str == "mixtral-8x7b":
                import replicate
                mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"
                output = replicate.run(mixtral_url, input={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "max_new_tokens": 75,
                })
                return re.sub(r"\s+", " ", "".join(output))
            elif model_str == "llama-3-70b-instruct":
                import replicate
                llama3_url = "meta/meta-llama-3-70b-instruct"
                output = replicate.run(llama3_url, input={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "max_new_tokens": 200,
                })
                return re.sub(r"\s+", " ", "".join(output))
            elif "_HF" in model_str:
                raise Exception("Sorry, fixing TODO :3")

            if response.usage:
                _accumulate_tokens(model_str, response.usage.prompt_tokens, response.usage.completion_tokens)
            answer = response.choices[0].message.content
            answer = re.sub(r"\s+", " ", answer)
            return answer

        except Exception as e:
            print(f"query_model warning ({model_str}): {type(e).__name__}: {str(e)[:200]}")
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


# ---------------------------------------------------------------------------
# Scenario classes (unchanged)
# ---------------------------------------------------------------------------

class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info

    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        with open("agentclinic_medqa.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioMedQAExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info

    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQAExtended:
    def __init__(self) -> None:
        with open("agentclinic_medqa_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQAExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioMIMICIVQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info

    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMIMICIV:
    def __init__(self) -> None:
        with open("agentclinic_mimiciv.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMIMICIVQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJMExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.question = scenario_dict["question"]
        self.image_url = scenario_dict["image_url"]
        self.diagnosis = [_sd["text"]
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        return self.patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"

    def exam_information(self) -> str:
        return self.physical_exams

    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJMExtended:
    def __init__(self) -> None:
        with open("agentclinic_nejm_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJMExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJM:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.question = scenario_dict["question"]
        self.image_url = scenario_dict["image_url"]
        self.diagnosis = [_sd["text"]
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        return self.patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"

    def exam_information(self) -> str:
        return self.physical_exams

    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJM:
    def __init__(self) -> None:
        with open("agentclinic_nejm.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJM(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


# ---------------------------------------------------------------------------
# Agent classes (unchanged logic, same interface)
# ---------------------------------------------------------------------------

class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4o", bias_present=None) -> None:
        self.disease = ""
        self.symptoms = ""
        self.agent_hist = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.reset()
        self.pipe = None
        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_patient(self, question) -> str:
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the doctor response: " + question + "Now please continue your dialogue\nPatient: ", self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(self.symptoms)
        return base + bias_prompt + symptoms

    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"


class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4o", max_infs=20, bias_present=None, img_request=False) -> None:
        self.infs = 0
        self.MAX_INFS = max_infs
        self.agent_hist = ""
        self.presentation = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.reset()
        self.pipe = None
        self.img_request = img_request
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        if self.bias_present == "recency":
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_doctor(self, question, image_requested=False) -> str:
        if self.infs >= self.MAX_INFS:
            return "Maximum inferences reached"
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: ", self.system_prompt(), image_requested=image_requested, scene=self.scenario)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {} questions total before you must make a decision. You have asked {} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"".format(self.MAX_INFS, self.infs) + ("You may also request medical images related to the disease to be returned with \"REQUEST IMAGES\"." if self.img_request else "")
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(self.presentation)
        return base + bias_prompt + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()


class MeasurementAgent:
    def __init__(self, scenario, backend_str="gpt4o") -> None:
        self.agent_hist = ""
        self.presentation = ""
        self.backend = backend_str
        self.scenario = scenario
        self.pipe = None
        self.reset()

    def inference_measurement(self, question) -> str:
        answer = query_model(self.backend, "\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor measurement request: " + question, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(self.information)
        return base + presentation

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
    answer = query_model(
        moderator_llm,
        "\nHere is the correct diagnosis: " + correct_diagnosis + "\n Here was the doctor dialogue: " + diagnosis + "\nAre these the same?",
        "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else."
    )
    return answer.lower()


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def _run_single(scenario_id, rollout_id, scenario, dataset, inf_type, doctor_bias,
                patient_bias, doctor_llm, patient_llm, measurement_llm, moderator_llm,
                total_inferences, img_request, output_dir):
    """Run one rollout of one scenario. Thread-safe."""
    if output_dir:
        out_path = os.path.join(output_dir, f"scenario_{scenario_id:04d}_rollout_{rollout_id:02d}.json")
        if os.path.exists(out_path):
            return None  # already done, skip

    meas_agent = MeasurementAgent(scenario=scenario, backend_str=measurement_llm)
    patient_agent = PatientAgent(scenario=scenario, bias_present=patient_bias, backend_str=patient_llm)
    doctor_agent = DoctorAgent(
        scenario=scenario,
        bias_present=doctor_bias,
        backend_str=doctor_llm,
        max_infs=total_inferences,
        img_request=img_request,
    )

    _pop_thread_tokens()  # clear any leftover tokens from prior runs on this thread

    trajectory = []
    pi_dialogue = ""
    doctor_dialogue = ""
    correctness = False
    correct_diagnosis = scenario.diagnosis_information()

    for _inf_id in range(total_inferences):
        if dataset in ("NEJM", "NEJM_Ext"):
            imgs = "REQUEST IMAGES" in doctor_dialogue if img_request else True
        else:
            imgs = False

        if _inf_id == total_inferences - 1:
            pi_dialogue += "This is the final question. Please provide a diagnosis.\n"

        if inf_type == "human_doctor":
            doctor_dialogue = input("\nQuestion for patient: ")
        else:
            _pop_last_reasoning()  # clear any stale value before the doctor call
            doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue, image_requested=imgs)

        doctor_turn = {"role": "doctor", "content": doctor_dialogue}
        reasoning = _pop_last_reasoning()
        if reasoning:
            doctor_turn["reasoning"] = reasoning
        trajectory.append(doctor_turn)

        # Case-insensitive: some doctor models (e.g. qwen thinking) emit the
        # control tokens in title case ("Diagnosis Ready:"). Verified against
        # existing deepseek data: affects <0.2% of prior rollouts.
        if "diagnosis ready" in doctor_dialogue.lower():
            correctness = compare_results(doctor_dialogue, correct_diagnosis, moderator_llm, None) == "yes"
            break

        if "request test" in doctor_dialogue.lower():
            pi_dialogue = meas_agent.inference_measurement(doctor_dialogue)
            trajectory.append({"role": "measurement", "content": pi_dialogue})
            patient_agent.add_hist(pi_dialogue)
        else:
            if inf_type == "human_patient":
                pi_dialogue = input("\nResponse to doctor: ")
            else:
                pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
            trajectory.append({"role": "patient", "content": pi_dialogue})
            meas_agent.add_hist(pi_dialogue)

        time.sleep(0.5)

    tokens = _pop_thread_tokens()
    cost = _compute_cost(tokens)

    record = {
        "scenario_id": scenario_id,
        "rollout_id": rollout_id,
        "dataset": dataset,
        "doctor_llm": doctor_llm,
        "correct": correctness,
        "correct_diagnosis": str(correct_diagnosis),
        "tokens": tokens,
        "cost_usd": round(cost, 6),
        "history": trajectory,
    }
    if output_dir:
        with open(out_path, "w") as f:
            json.dump(record, f)
    return record


def main(api_key, inf_type, doctor_bias, patient_bias, doctor_llm, patient_llm,
         measurement_llm, moderator_llm, num_scenarios, dataset, img_request,
         total_inferences, num_rollouts=1, concurrency=1, output_dir=None,
         anthropic_api_key=None):

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        scenario_loader = ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM":
        scenario_loader = ScenarioLoaderNEJM()
    elif dataset == "NEJM_Ext":
        scenario_loader = ScenarioLoaderNEJMExtended()
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV()
    else:
        raise Exception("Dataset {} does not exist".format(str(dataset)))

    if num_scenarios is None:
        num_scenarios = scenario_loader.num_scenarios

    jobs = [
        (sid, rid, scenario_loader.get_scenario(id=sid))
        for sid in range(min(num_scenarios, scenario_loader.num_scenarios))
        for rid in range(num_rollouts)
    ]

    total_cost = 0.0
    total_jobs = len(jobs)

    # Count already-completed files for accurate resume progress
    already_done = 0
    if output_dir:
        for sid, rid, _ in jobs:
            p = os.path.join(output_dir, f"scenario_{sid:04d}_rollout_{rid:02d}.json")
            if os.path.exists(p):
                already_done += 1
    total_done = already_done
    total_correct = 0

    print(f"Starting {total_jobs} rollouts ({num_scenarios} scenarios × {num_rollouts} rollouts) with {concurrency} workers")
    if already_done:
        print(f"Resuming: {already_done} already complete, {total_jobs - already_done} remaining")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _run_single, sid, rid, scenario, dataset, inf_type,
                doctor_bias, patient_bias, doctor_llm, patient_llm,
                measurement_llm, moderator_llm, total_inferences,
                img_request, output_dir
            ): (sid, rid)
            for sid, rid, scenario in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                total_done += 1
                total_cost += result.get("cost_usd", 0.0)
                if result["correct"]:
                    total_correct += 1
                pct = int(total_correct / total_done * 100)
                print(f"\r  {total_done}/{total_jobs} rollouts  |  {total_correct} correct ({pct}%)  |  ${total_cost:.3f}", end="", flush=True)

    print(f"\n\nDone: {total_correct}/{total_done} correct ({int(total_correct/total_done*100) if total_done else 0}%)")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation CLI')
    parser.add_argument('--openai_api_key', type=str, required=False, help='OpenAI API Key')
    parser.add_argument('--anthropic_api_key', type=str, default=None, required=False)
    parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
    parser.add_argument('--doctor_bias', type=str, default='None', choices=["None", "recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--patient_bias', type=str, default='None', choices=["None", "recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--doctor_llm', type=str, default='vertex-deepseek-v3.2')
    parser.add_argument('--patient_llm', type=str, default='gpt-4o-mini')
    parser.add_argument('--measurement_llm', type=str, default='gpt-4o-mini')
    parser.add_argument('--moderator_llm', type=str, default='gpt-4o-mini')
    parser.add_argument('--agent_dataset', type=str, default='MedQA')
    parser.add_argument('--doctor_image_request', type=bool, default=False)
    parser.add_argument('--num_scenarios', type=int, default=None, required=False)
    parser.add_argument('--total_inferences', type=int, default=20, required=False)
    parser.add_argument('--num_rollouts', type=int, default=1, required=False, help='Number of rollouts per scenario')
    parser.add_argument('--concurrency', type=int, default=4, required=False, help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default=None, required=False, help='Directory to save per-scenario trajectory JSON files')

    args = parser.parse_args()

    main(
        api_key=args.openai_api_key,
        inf_type=args.inf_type,
        doctor_bias=args.doctor_bias,
        patient_bias=args.patient_bias,
        doctor_llm=args.doctor_llm,
        patient_llm=args.patient_llm,
        measurement_llm=args.measurement_llm,
        moderator_llm=args.moderator_llm,
        num_scenarios=args.num_scenarios,
        dataset=args.agent_dataset,
        img_request=args.doctor_image_request,
        total_inferences=args.total_inferences,
        output_dir=args.output_dir,
        num_rollouts=args.num_rollouts,
        concurrency=args.concurrency,
        anthropic_api_key=args.anthropic_api_key,
    )
