"""Tests for smart router and A/B test framework."""
from internal.router.ab_test import ABTestConfig, ABTestRouter, Variant, analyze_ab_test
from internal.router.router import QueryComplexity, SmartRouter, classify_query


def test_classify_simple_question():
    assert classify_query("What is the capital of France?") == QueryComplexity.SIMPLE


def test_classify_complex_reasoning():
    prompt = "Analyze step by step why quantization reduces model quality."
    assert classify_query(prompt) == QueryComplexity.COMPLEX


def test_classify_code_generation():
    prompt = "Write code to implement a binary search in Python"
    assert classify_query(prompt) == QueryComplexity.MEDIUM


def test_classify_very_long_prompt():
    prompt = "word " * 600
    assert classify_query(prompt) == QueryComplexity.COMPLEX


def test_router_returns_valid_model():
    router = SmartRouter.default()
    decision = router.route("What is 2+2?")
    assert decision.model in router.routing_table.values()
    assert decision.complexity == QueryComplexity.SIMPLE


def test_router_override():
    router = SmartRouter.default()
    router.override(QueryComplexity.SIMPLE, "my-custom-model:1b")
    decision = router.route("What is 2+2?")
    assert decision.model == "my-custom-model:1b"


def test_ab_test_consistent_routing():
    """Same user should always get the same variant."""
    cfg = ABTestConfig(
        experiment_name="test", control_model="ctl", treatment_model="treat",
        traffic_split=0.5,
    )
    ab = ABTestRouter(cfg)
    user_id = "user-42"
    variants = {ab.route(user_id)[0] for _ in range(10)}
    assert len(variants) == 1  # Same variant every call


def test_ab_test_traffic_distribution():
    """Over many users, roughly split traffic by traffic_split."""
    cfg = ABTestConfig(
        experiment_name="test", control_model="ctl", treatment_model="treat",
        traffic_split=0.5,
    )
    ab = ABTestRouter(cfg)
    variants = [ab.route(f"user-{i}")[0] for i in range(1000)]
    treatment_pct = sum(1 for v in variants if v == Variant.TREATMENT) / len(variants)
    assert 0.4 < treatment_pct < 0.6  # Within 10% of 50%


def test_ab_test_disabled_returns_control():
    cfg = ABTestConfig(
        experiment_name="test", control_model="ctl", treatment_model="treat",
        traffic_split=1.0, enabled=False,
    )
    ab = ABTestRouter(cfg)
    variant, model = ab.route("any-user")
    assert variant == Variant.CONTROL
    assert model == "ctl"


def test_analyze_ab_test_lift():
    result = analyze_ab_test(
        control_values=[10.0, 10.0, 10.0],
        treatment_values=[15.0, 15.0, 15.0],
        metric_name="tokens_per_sec",
    )
    assert result.lift_pct == 50.0
    assert result.treatment_wins
    assert result.sample_size_control == 3
    assert result.sample_size_treatment == 3
