# SPDX-License-Identifier: MPL-2.0
import bgbot_cpp
import pytest

from neongammon import flip_owner
from neongammon.analyzer import resolve_owner


@pytest.mark.parametrize(
    ("name", "owner", "opposite"),
    [
        ("CeNtErEd", bgbot_cpp.CubeOwner.CENTERED, bgbot_cpp.CubeOwner.CENTERED),
        ("PLAYER", bgbot_cpp.CubeOwner.PLAYER, bgbot_cpp.CubeOwner.OPPONENT),
        ("opponent", bgbot_cpp.CubeOwner.OPPONENT, bgbot_cpp.CubeOwner.PLAYER),
    ],
)
def test_resolve_and_flip_accept_names_and_native_owners(name, owner, opposite):
    for value in (name, owner):
        assert resolve_owner(value) == owner
        assert flip_owner(value) == opposite


@pytest.mark.parametrize("convert", [resolve_owner, flip_owner])
@pytest.mark.parametrize("value", ["white", "", 0, None, []])
def test_cube_owner_helpers_raise_value_error_for_invalid_values(convert, value):
    with pytest.raises(ValueError, match="Invalid cube owner.*centered, player, or opponent"):
        convert(value)
