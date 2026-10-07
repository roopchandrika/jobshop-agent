import pytest

from jobshop.tools.registry import ToolRegistry
from tests.helpers import FakeClock, build_ctx


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def ctx(clock):
    return build_ctx(clock=clock)


@pytest.fixture
def registry(ctx):
    return ToolRegistry(ctx)
