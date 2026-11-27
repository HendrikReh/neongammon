# SPDX-License-Identifier: MPL-2.0
import struct

import bgbot_cpp
import pytest

from neongammon.sl_train import read_weights, write_weights


@pytest.mark.parametrize(("inputs", "encoder"), [(196, 1), (244, 2), (214, 3)])
def test_native_and_python_write_the_same_versioned_format(tmp_path, inputs, encoder):
    network = bgbot_cpp.NeuralNetwork(7, inputs)
    legacy = tmp_path / "legacy.weights"
    native = tmp_path / "native.weights"
    python = tmp_path / "python.weights"

    assert network.save_weights(str(legacy))
    assert network.save_weights(str(native), versioned=True)
    write_weights(str(python), *read_weights(str(legacy)), versioned=True)

    header = struct.pack("<4s5I", b"NGWT", 1, encoder, 7, inputs, 5)
    assert native.read_bytes() == python.read_bytes() == header + legacy.read_bytes()[12:]
    restored = bgbot_cpp.NeuralNetwork(7, inputs, seed=19)
    assert restored.load_weights(str(python))
    assert restored.save_weights(str(python))
    assert python.read_bytes() == legacy.read_bytes()


@pytest.mark.parametrize("damage", ["version", "encoder", "topology", "truncation", "trailing"])
def test_bad_versioned_weights_leave_the_network_intact(tmp_path, damage):
    network = bgbot_cpp.NeuralNetwork(7, 196)
    good = tmp_path / "good.weights"
    bad = tmp_path / "bad.weights"
    assert network.save_weights(str(good), versioned=True)
    original = good.read_bytes()
    substitutions = {"version": (4, 2), "encoder": (8, 2), "topology": (20, 4)}
    if damage in substitutions:
        payload = bytearray(original)
        struct.pack_into("<I", payload, *substitutions[damage])
    elif damage == "truncation":
        payload = original[:-1]
    else:
        payload = original + b"\0"
    bad.write_bytes(payload)

    assert not network.load_weights(str(bad))
    with pytest.raises(ValueError):
        read_weights(str(bad))
    assert network.save_weights(str(bad), versioned=True)
    assert bad.read_bytes() == original
