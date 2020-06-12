import pytest


def fail_if(boolean):
    if boolean:
        pytest.fail()
