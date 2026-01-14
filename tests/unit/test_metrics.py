import pytest

from hypertic.models.metrics import Metrics


@pytest.mark.unit
class TestMetrics:
    def test_metrics_creation_default(self):
        """Test metrics creation with default values."""
        metrics = Metrics()
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.total_tokens == 0
        assert metrics.provider_metrics is None
        assert metrics.additional_metrics is None

    @pytest.mark.parametrize(
        "input_tokens,output_tokens,total_tokens",
        [
            (100, 200, 300),
            (50, 75, 125),
            (0, 0, 0),
        ],
    )
    def test_metrics_creation_with_values(self, input_tokens, output_tokens, total_tokens):
        """Test metrics creation with various token values."""
        metrics = Metrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        assert metrics.input_tokens == input_tokens
        assert metrics.output_tokens == output_tokens
        assert metrics.total_tokens == total_tokens

    def test_metrics_addition(self):
        """Test metrics addition operator."""
        metrics1 = Metrics(input_tokens=10, output_tokens=20)
        metrics2 = Metrics(input_tokens=15, output_tokens=25)
        result = metrics1 + metrics2
        assert result.input_tokens == 25
        assert result.output_tokens == 45

    def test_metrics_inplace_addition(self):
        """Test metrics in-place addition operator."""
        metrics1 = Metrics(input_tokens=10, output_tokens=20)
        metrics2 = Metrics(input_tokens=15, output_tokens=25)
        metrics1 += metrics2
        assert metrics1.input_tokens == 25
        assert metrics1.output_tokens == 45

    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        result = metrics.to_dict()
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert result["input_tokens"] == 10

    def test_metrics_to_dict_filters_zeros(self):
        """Test that zero values are filtered from dict."""
        metrics = Metrics()
        result = metrics.to_dict()
        assert len(result) == 0 or all(v != 0 for v in result.values() if isinstance(v, (int, float)))

    @pytest.mark.parametrize(
        "usage,expected_input,expected_output,expected_total",
        [
            ({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, 10, 20, 30),
            ({"input_tokens": 15, "output_tokens": 25}, 15, 25, 40),
            ({"billed_units": {"input_tokens": 20, "output_tokens": 30}}, 20, 30, 50),
        ],
    )
    def test_metrics_from_api_response_formats(self, usage, expected_input, expected_output, expected_total):
        """Test metrics creation from different API response formats."""
        metrics = Metrics.from_api_response(usage)
        assert metrics.input_tokens == expected_input
        assert metrics.output_tokens == expected_output
        assert metrics.total_tokens == expected_total

    def test_metrics_provider_metrics(self):
        """Test that provider metrics are preserved."""
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        metrics = Metrics.from_api_response(usage)
        assert metrics.provider_metrics == usage

    def test_metrics_with_additional_metrics(self):
        """Test metrics with additional custom metrics."""
        metrics = Metrics(
            input_tokens=10,
            output_tokens=20,
            additional_metrics={"cost": 0.01},
        )
        assert metrics.additional_metrics == {"cost": 0.01}

    def test_metrics_addition_with_provider_metrics(self):
        """Test metrics addition with provider metrics."""
        metrics1 = Metrics(
            input_tokens=10,
            output_tokens=20,
            provider_metrics={"prompt_tokens": 10, "completion_tokens": 20},
        )
        metrics2 = Metrics(
            input_tokens=5,
            output_tokens=15,
            provider_metrics={"prompt_tokens": 5, "completion_tokens": 15},
        )
        result = metrics1 + metrics2
        assert result.input_tokens == 15
        assert result.output_tokens == 35
        assert result.provider_metrics is not None
        assert "prompt_tokens" in result.provider_metrics
        assert "completion_tokens" in result.provider_metrics

    def test_metrics_addition_with_additional_metrics(self):
        """Test metrics addition with additional metrics."""
        metrics1 = Metrics(
            input_tokens=10,
            output_tokens=20,
            additional_metrics={"cost": 0.01},
        )
        metrics2 = Metrics(
            input_tokens=5,
            output_tokens=15,
            additional_metrics={"cost": 0.02},
        )
        result = metrics1 + metrics2
        assert result.additional_metrics is not None
        assert "cost" in result.additional_metrics

    def test_metrics_radd_with_zero(self):
        """Test right addition with zero."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        result = 0 + metrics
        assert result.input_tokens == 10
        assert result.output_tokens == 20

    def test_metrics_radd_with_metrics(self):
        """Test right addition with Metrics."""
        metrics1 = Metrics(input_tokens=10, output_tokens=20)
        metrics2 = Metrics(input_tokens=5, output_tokens=15)
        result = metrics2.__radd__(metrics1)
        assert result.input_tokens == 15
        assert result.output_tokens == 35

    def test_metrics_radd_with_invalid_type(self):
        """Test right addition with invalid type."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        result = metrics.__radd__("invalid")
        assert result == metrics

    def test_metrics_iadd_with_provider_metrics(self):
        """Test in-place addition with provider metrics."""
        metrics1 = Metrics(
            input_tokens=10,
            output_tokens=20,
            provider_metrics={"prompt_tokens": 10},
        )
        metrics2 = Metrics(
            input_tokens=5,
            output_tokens=15,
            provider_metrics={"completion_tokens": 15},
        )
        metrics1 += metrics2
        assert metrics1.input_tokens == 15
        assert metrics1.output_tokens == 35
        assert metrics1.provider_metrics is not None
        assert "prompt_tokens" in metrics1.provider_metrics
        assert "completion_tokens" in metrics1.provider_metrics

    def test_metrics_iadd_with_invalid_type(self):
        """Test in-place addition with invalid type."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        original = metrics
        metrics += "invalid"
        assert metrics == original

    def test_metrics_from_api_response_input_tokens_variants(self):
        """Test from_api_response with various input token field names."""
        test_cases = [
            ({"input_tokens": 10}, 10),
            ({"prompt_tokens": 20}, 20),
            ({"inputTokens": 30}, 30),
            ({"promptTokenCount": 40}, 40),
        ]
        for usage, expected in test_cases:
            metrics = Metrics.from_api_response(usage)
            assert metrics.input_tokens == expected

    def test_metrics_from_api_response_output_tokens_variants(self):
        """Test from_api_response with various output token field names."""
        test_cases = [
            ({"output_tokens": 10}, 10),
            ({"completion_tokens": 20}, 20),
            ({"outputTokens": 30}, 30),
            ({"candidatesTokenCount": 40}, 40),
        ]
        for usage, expected in test_cases:
            metrics = Metrics.from_api_response(usage)
            assert metrics.output_tokens == expected

    def test_metrics_from_api_response_billed_units(self):
        """Test from_api_response with billed_units format."""
        usage = {
            "billed_units": {
                "input_tokens": 25,
                "output_tokens": 35,
            }
        }
        metrics = Metrics.from_api_response(usage)
        assert metrics.input_tokens == 25
        assert metrics.output_tokens == 35
        assert metrics.total_tokens == 60

    def test_metrics_from_api_response_tokens_nested(self):
        """Test from_api_response with nested tokens format."""
        usage = {
            "tokens": {
                "input_tokens": 15,
                "output_tokens": 25,
            }
        }
        metrics = Metrics.from_api_response(usage)
        assert metrics.input_tokens == 15
        assert metrics.output_tokens == 25
        assert metrics.total_tokens == 40

    def test_metrics_from_api_response_empty(self):
        """Test from_api_response with empty usage."""
        metrics = Metrics.from_api_response({})
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.total_tokens == 0

    def test_metrics_to_dict_with_provider_metrics(self):
        """Test to_dict includes provider_metrics when present."""
        metrics = Metrics(
            input_tokens=10,
            output_tokens=20,
            provider_metrics={"custom": "value"},
        )
        result = metrics.to_dict()
        assert "provider_metrics" in result
        assert result["provider_metrics"]["custom"] == "value"

    def test_metrics_to_dict_with_additional_metrics(self):
        """Test to_dict includes additional_metrics when present."""
        metrics = Metrics(
            input_tokens=10,
            output_tokens=20,
            additional_metrics={"cost": 0.01},
        )
        result = metrics.to_dict()
        assert "additional_metrics" in result
        assert result["additional_metrics"]["cost"] == 0.01

    def test_metrics_to_dict_excludes_none_values(self):
        """Test to_dict excludes None values."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        result = metrics.to_dict()
        assert "provider_metrics" not in result or result["provider_metrics"] is not None
        assert "additional_metrics" not in result or result["additional_metrics"] is not None

    def test_metrics_to_dict_excludes_empty_dicts(self):
        """Test to_dict excludes empty dicts."""
        metrics = Metrics(
            input_tokens=10,
            output_tokens=20,
            provider_metrics={},
            additional_metrics={},
        )
        result = metrics.to_dict()
        # Empty dicts should be filtered out
        assert "provider_metrics" not in result or len(result.get("provider_metrics", {})) > 0
        assert "additional_metrics" not in result or len(result.get("additional_metrics", {})) > 0
