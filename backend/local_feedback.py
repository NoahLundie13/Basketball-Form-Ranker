import json
import os
import re
import threading


DEFAULT_MODEL_ID = "mlx-community/Qwen3-4B-4bit"
MODEL_ID = os.environ.get("SHOT_FEEDBACK_MODEL", DEFAULT_MODEL_ID)
_MODEL = None
_TOKENIZER = None
_LOADED_MODEL_ID = None
_MODEL_LOCK = threading.Lock()
_FORBIDDEN_TERMS = re.compile(
    r"\b(?:zaid|shai|reference|gold standard|standard|benchmark|compare(?:d)?(?:\s+to|\s+with)?)\b",
    re.IGNORECASE,
)
_CONTRADICTORY_ADVICE = re.compile(
    r"\b(?:reduce|lower|decrease|lessen|cut)\s+(?:the\s+)?(?:shooting\s+)?force\b",
    re.IGNORECASE,
)
_ACTION_WORDS = re.compile(
    r"\b(try|adjust|keep|use|let|focus|aim|bring|relax|time|finish|drive|practice|"
    r"bend|straighten|extend|flex|hold|wait|delay|stay|shift|align|allow|cue|avoid|increase|decrease|"
    r"maintain|support|supports|steady|position|place)\b",
    re.IGNORECASE,
)
_BENEFIT_WORDS = re.compile(
    r"\b(help|helps|helping|so|because|allow|allows|make|makes|improve|keep|promote|promotes|support|supports)\b",
    re.IGNORECASE,
)
_OVERCONFIDENT_CLAIMS = re.compile(
    r"(?<!not )(?<!n't )(?<!no )\bguarantee(?:d|s)?\b|"
    r"\b(?:optimal|perfect|definitely|always|never)\b|"
    r"\bwill\s+(?:ensure|fix|perfect)\b|"
    r"(?<!helps to )(?<!help to )(?<!help )(?<!helps )\bensure(?:s|d|ing)?\b",
    re.IGNORECASE,
)
_AWKWARD_TIMING = re.compile(r"\btoo\s+(?:soon|early|late)\s+than\b", re.IGNORECASE)


class LocalFeedbackError(RuntimeError):
    pass


def _load_model(model_id):
    global _MODEL, _TOKENIZER, _LOADED_MODEL_ID
    if _MODEL is None or _LOADED_MODEL_ID != model_id:
        try:
            from mlx_lm import load
        except ImportError as error:
            raise LocalFeedbackError(
                "Local feedback needs mlx-lm on an Apple Silicon Mac. Install requirements.txt."
            ) from error
        _MODEL, _TOKENIZER = load(model_id)
        _LOADED_MODEL_ID = model_id
    return _MODEL, _TOKENIZER


def _make_prompt(findings):
    system_message = (
        "You are a practical basketball shooting coach. Turn the supplied measured shot facts "
        "into concise, specific coaching feedback. The desired values describe how the shot "
        "should move; they are not proof that one technique is universally correct. Use only "
        "the supplied facts and metric meaning. For each finding, say what is off and where in "
        "the shot, phrase the absolute difference as higher or lower than it should be, give one "
        "concrete adjustment, and explain intuitively how that adjustment may help. Use the "
        "'difference' value for the amount of deviation; do not mistake 'observed_value' for the "
        "difference. For example, an observed angle of 180 with a difference of 83.9 means the "
        "angle is about 84 degrees higher than it should be, not 180 degrees higher. Follow the "
        "metric meaning and coaching context. Joint angle describes bending/straightening at the "
        "joint, not the joint's position in space. Never advise reducing shooting force or "
        "dropping the shooting arm. Keep advice cautious: say it can help, not that it guarantees "
        "a result. Do not name people, data sources, baselines, standards, comparisons, or "
        "references. Do not invent measurements, causes, or diagnoses. For a large angle "
        "difference near 90 degrees, natural wording such as 'nearly 90 degrees higher' is "
        "appropriate. Keep each finding to 2 or 3 short sentences. "
        "Return only valid JSON in this shape: {\"feedback\":[{\"metric\":\"metric_key\","
        "\"message\":\"coaching text\"}]}"
    )
    user_message = (
        "Write exactly one coaching message for each supplied finding. Each message must include "
        "the absolute difference (or a natural rounding such as 'nearly 90 degrees'), the phrase "
        "'than it should', a specific physical adjustment, and how the adjustment may help. "
        "State where in the shot it happens. For guide-elbow timing, if opening_timing.direction "
        "is 'earlier', say it opens too soon and advise keeping the guide hand supported or "
        "waiting until the shooting arm begins extending. If direction is 'later', advise "
        "starting the guide-hand movement sooner. When opening_timing is present, state its "
        "difference_percentage_points as 'percentage points' and use its earlier/later direction; "
        "do not substitute the elbow-angle difference. Say 'starts opening X percentage points "
        "earlier/later than it should'; do not combine 'too soon' with 'than'. Never reverse "
        "this timing cue. Do not mention "
        "the input fields or output "
        "anything outside the required JSON. Findings: "
        + json.dumps(findings, separators=(",", ":"), allow_nan=False)
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _parse_response(response, findings):
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    start = response.find("{")
    if start < 0:
        raise LocalFeedbackError("The local model did not return JSON feedback.")
    try:
        payload, _ = json.JSONDecoder().raw_decode(response[start:])
    except json.JSONDecodeError as error:
        raise LocalFeedbackError("The local model returned invalid JSON feedback.") from error

    entries = payload.get("feedback") if isinstance(payload, dict) else None
    expected = {finding["metric"] for finding in findings}
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise LocalFeedbackError("The local model returned the wrong number of feedback items.")

    messages = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise LocalFeedbackError("A local feedback item was not a JSON object.")
        metric = entry.get("metric")
        message = entry.get("message")
        if metric not in expected or metric in messages or not isinstance(message, str):
            raise LocalFeedbackError("The local model returned an unknown or duplicate metric.")
        message = " ".join(message.split())
        lowered = message.lower()
        if _FORBIDDEN_TERMS.search(message):
            raise LocalFeedbackError("Feedback must not name or mention a comparison source.")
        if _CONTRADICTORY_ADVICE.search(message):
            raise LocalFeedbackError("Feedback must not recommend reducing shooting force.")
        if _OVERCONFIDENT_CLAIMS.search(message):
            raise LocalFeedbackError("Feedback must not promise or guarantee a result.")
        if _AWKWARD_TIMING.search(message):
            raise LocalFeedbackError(
                "State timing naturally, such as 'starts opening 37 percentage points earlier than it should'."
            )
        if "than it should" not in lowered or not re.search(r"\d", message):
            raise LocalFeedbackError("Feedback must state a measured amount as different than it should be.")
        timing = next(
            (finding.get("opening_timing") for finding in findings if finding["metric"] == metric),
            None,
        )
        if timing:
            expected_timing = abs(float(timing["difference_percentage_points"]))
            timing_numbers = _number_variants(expected_timing)
            if (not any(_contains_number(message, value) for value in timing_numbers)
                    or "percentage point" not in lowered
                    or timing["direction"] not in lowered):
                raise LocalFeedbackError(
                    "Guide-elbow feedback must state the measured timing difference and direction."
                )
        elif "difference" in next(finding for finding in findings if finding["metric"] == metric):
            finding = next(finding for finding in findings if finding["metric"] == metric)
            expected_difference = abs(float(finding["difference"]))
            expected_numbers = _number_variants(expected_difference)
            near_ninety = (
                finding.get("unit") == "degrees"
                and 80 <= expected_difference < 100
                and "nearly 90 degrees" in lowered
            )
            if (not near_ninety
                    and (not any(_contains_number(message, value) for value in expected_numbers)
                         or finding.get("direction", "") not in lowered)):
                raise LocalFeedbackError("Feedback must use the measured difference and direction.")
        if not _ACTION_WORDS.search(message):
            raise LocalFeedbackError("Feedback must include a concrete adjustment.")
        if not _BENEFIT_WORDS.search(message):
            raise LocalFeedbackError("Feedback must explain how the adjustment may help.")
        if timing and timing["direction"] == "earlier":
            opens_sooner = re.search(
                r"\b(?:open|start|begin|extend)\s+(?:it|the guide elbow|the guide hand|the arm)?\s*"
                r"(?:earlier|sooner)\b",
                lowered,
            )
            waits_longer = re.search(r"\b(wait|delay|hold|keep|stay|until)\b", lowered)
            if opens_sooner or not waits_longer:
                raise LocalFeedbackError(
                    "The guide elbow already opens too early; advise delaying it or supporting the ball longer."
                )
        elif timing and timing["direction"] == "later":
            starts_sooner = re.search(
                r"\b(?:open|start|begin|extend)\s+(?:it|the guide elbow|the guide hand|the arm)?\s*"
                r"(?:earlier|sooner)\b",
                lowered,
            )
            if not starts_sooner:
                raise LocalFeedbackError(
                    "The guide elbow opens too late; advise starting its movement sooner."
                )
        if len(message) > 500:
            raise LocalFeedbackError("Feedback must be concise.")
        messages[metric] = message

    return [{"metric": finding["metric"], "message": messages[finding["metric"]]}
            for finding in findings]


def _contains_number(text, number):
    pattern = rf"(?<![\d.]){re.escape(number)}(?![\d.])"
    return re.search(pattern, text) is not None


def _number_variants(value):
    variants = {str(round(value)), f"{value:.1f}"}
    variants.add(f"{value:.2f}".rstrip("0").rstrip("."))
    return variants


def _generate_single_finding(model, tokenizer, generate, make_sampler, finding):
    messages = _make_prompt([finding])
    previous_response = ""
    correction = ""
    last_error = None
    for attempt in range(3):
        prompt_messages = messages
        if correction:
            prompt_messages = messages + [
                {"role": "assistant", "content": previous_response},
                {"role": "user", "content": correction},
            ]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        previous_response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=220,
            sampler=make_sampler(temp=0.0),
            verbose=False,
        )
        try:
            return _parse_response(previous_response, [finding])[0]
        except LocalFeedbackError as error:
            last_error = error
            correction = (
                f"Rewrite the feedback as valid JSON and correct this issue: {error} "
                "Keep the same metric and measurements. Describe likely benefits naturally, "
                "but do not promise or guarantee a result."
            )

    raise LocalFeedbackError(
        f"The local model could not produce valid feedback for {finding['metric']}: {last_error}"
    )


def generate_feedback(findings, model_id=MODEL_ID):
    if not findings:
        return []

    try:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as error:
        raise LocalFeedbackError(
            "Local feedback needs mlx-lm on an Apple Silicon Mac. Install requirements.txt."
        ) from error

    with _MODEL_LOCK:
        model, tokenizer = _load_model(model_id)
        return [
            _generate_single_finding(model, tokenizer, generate, make_sampler, finding)
            for finding in findings
        ]