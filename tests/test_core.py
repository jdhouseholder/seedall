import os
import random
import warnings

import numpy as np
import pytest

import seedall
from seedall.core import seed, available, get_states, set_states, temp_seed, SeedContext


class TestPublicAPI:
    """Verify that __init__.py re-exports the full public API."""

    def test_seed_accessible(self):
        assert seedall.seed is seed

    def test_available_accessible(self):
        assert seedall.available is available

    def test_get_states_accessible(self):
        assert seedall.get_states is get_states

    def test_set_states_accessible(self):
        assert seedall.set_states is set_states

    def test_temp_seed_accessible(self):
        assert seedall.temp_seed is temp_seed

    def test_seed_context_accessible(self):
        assert seedall.SeedContext is SeedContext


class TestSeedValidation:
    def test_rejects_float(self):
        with pytest.raises(TypeError, match="int"):
            seed(3.14)

    def test_rejects_string(self):
        with pytest.raises(TypeError, match="int"):
            seed("42")

    def test_rejects_none(self):
        with pytest.raises(TypeError, match="int"):
            seed(None)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            seed(-1)

    def test_accepts_zero(self):
        result = seed(0)
        assert all(v is True for v in result.values())


class TestSeed:
    def test_returns_success_for_available_backends(self):
        result = seed(42)
        assert all(v is True for v in result.values())

    def test_seeds_stdlib_random(self):
        seed(42)
        a = random.random()
        seed(42)
        b = random.random()
        assert a == b

    def test_seeds_numpy(self):
        seed(42)
        a = np.random.rand()
        seed(42)
        b = np.random.rand()
        assert a == b

    def test_seeds_hashseed(self):
        seed(99)
        assert os.environ.get("PYTHONHASHSEED") == "99"

    def test_backends_filter(self):
        result = seed(42, backends=["random"])
        assert list(result.keys()) == ["random"]

    def test_missing_backend_returns_false(self):
        result = seed(42, backends=["nonexistent"])
        assert result == {"nonexistent": False}

    def test_warn_missing(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            seed(42, backends=["nonexistent"], warn_missing=True)
            assert len(w) == 1
            assert "nonexistent" in str(w[0].message)

    def test_no_warn_by_default(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            seed(42, backends=["nonexistent"], warn_missing=False)
            assert len(w) == 0


class TestAvailable:
    def test_contains_builtin_backends(self):
        names = available()
        assert "random" in names
        assert "hashseed" in names
        assert "numpy" in names

    def test_returns_list(self):
        assert isinstance(available(), list)


class TestGetSetStates:
    def test_round_trip_stdlib(self):
        seed(42)
        states = get_states(backends=["random"])
        val_a = random.random()
        set_states(states)
        val_b = random.random()
        assert val_a == val_b

    def test_round_trip_numpy(self):
        seed(42)
        states = get_states(backends=["numpy"])
        val_a = np.random.rand()
        set_states(states)
        val_b = np.random.rand()
        assert val_a == val_b

    def test_get_states_subset(self):
        states = get_states(backends=["random"])
        assert "random" in states
        assert "numpy" not in states

    def test_get_states_all(self):
        states = get_states()
        assert "random" in states
        assert "numpy" in states

    def test_set_states_ignores_unknown(self):
        set_states({"nonexistent": None})  # should not raise


class TestHashseedState:
    def test_restores_none_state(self):
        """set_state(None) should remove PYTHONHASHSEED from env."""
        os.environ.pop("PYTHONHASHSEED", None)
        original = get_states(backends=["hashseed"])
        assert original["hashseed"] is None

        seed(42, backends=["hashseed"])
        assert os.environ.get("PYTHONHASHSEED") == "42"

        set_states(original)
        assert os.environ.get("PYTHONHASHSEED") is None

    def test_restores_previous_value(self):
        seed(11, backends=["hashseed"])
        states = get_states(backends=["hashseed"])
        seed(99, backends=["hashseed"])
        assert os.environ.get("PYTHONHASHSEED") == "99"

        set_states(states)
        assert os.environ.get("PYTHONHASHSEED") == "11"


class TestTempSeed:
    def test_restores_stdlib_state(self):
        seed(0)
        before = random.random()

        seed(0)
        random.random()  # advance state
        with temp_seed(99):
            inside = random.random()
        after = random.random()

        # after exiting temp_seed, should resume from same point
        seed(0)
        random.random()  # advance past first call
        expected_after = random.random()
        assert after == expected_after

    def test_reproducible_inside(self):
        with temp_seed(42):
            a = random.random()
        with temp_seed(42):
            b = random.random()
        assert a == b

    def test_reproducible_numpy_inside(self):
        with temp_seed(42):
            a = np.random.rand()
        with temp_seed(42):
            b = np.random.rand()
        assert a == b


class TestSeedContext:
    def test_enter_exit(self):
        ctx = SeedContext(42)
        ctx.enter()
        a = random.random()
        ctx.exit()

        ctx.enter()
        b = random.random()
        ctx.exit()
        assert a == b

    def test_restores_state_on_exit(self):
        seed(0)
        state_before = get_states(backends=["random"])

        ctx = SeedContext(99)
        ctx.enter()
        random.random()
        ctx.exit()

        state_after = get_states(backends=["random"])
        assert state_before["random"] == state_after["random"]

    def test_with_statement(self):
        with SeedContext(42) as ctx:
            a = random.random()
        with SeedContext(42):
            b = random.random()
        assert a == b

    def test_double_enter_raises(self):
        ctx = SeedContext(42)
        ctx.enter()
        with pytest.raises(RuntimeError, match="already active"):
            ctx.enter()
        ctx.exit()  # cleanup

    def test_exit_without_enter_raises(self):
        ctx = SeedContext(42)
        with pytest.raises(RuntimeError, match="without a matching enter"):
            ctx.exit()

    def test_reusable_after_exit(self):
        ctx = SeedContext(42)
        ctx.enter()
        a = random.random()
        ctx.exit()

        # can re-enter after exiting
        ctx.enter()
        b = random.random()
        ctx.exit()
        assert a == b
