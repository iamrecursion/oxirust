# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0
"""Tests for the engine-backed ``oxiphysics.rigid.PyWorld``.

Run after ``maturin develop --release`` via::

    python -m pytest python/tests/test_rigid_world.py -v
"""

import math

import pytest

try:
    import oxiphysics as ox

    HAS_OXIPHYSICS = True
except ImportError:
    HAS_OXIPHYSICS = False

pytestmark = pytest.mark.skipif(not HAS_OXIPHYSICS, reason="oxiphysics not built")

DT = 1.0 / 60.0


def _sphere(mass, position, radius):
    body = ox.rigid.PyRigidBody(0, mass, position)
    body.add_collider(ox.rigid.PyCollider(radius, 0.5, 0.0))
    return body


def test_overlapping_spheres_produce_contacts():
    rigid = ox.rigid
    world = rigid.PyWorld([0.0, 0.0, 0.0])
    a = world.add_body(_sphere(1.0, [0.0, 0.0, 0.0], 0.5))
    incoming = _sphere(1.0, [0.8, 0.0, 0.0], 0.5)
    incoming.velocity = [-2.0, 0.0, 0.0]  # approaching: the solver must push
    b = world.add_body(incoming)
    rigid.PyPhysicsPipeline().step(world, DT, 1)
    assert world.body_count() == 2
    contacts = world.get_contacts()
    assert contacts, "overlapping spheres must touch"
    assert {contacts[0].body_a, contacts[0].body_b} == {a, b}
    assert contacts[0].impulse > 0.0


def test_bodies_attribute_is_a_snapshot():
    world = ox.rigid.PyWorld()
    world.add_body(_sphere(1.0, [0.0, 1.0, 0.0], 0.5))
    snapshot = world.bodies
    snapshot.add(ox.rigid.PyRigidBody())
    assert world.body_count() == 1


def test_ball_socket_pendulum_keeps_anchor():
    rigid = ox.rigid
    world = rigid.PyWorld()
    pivot = world.add_body(rigid.PyRigidBody.static_body(0, [0.0, 0.0, 0.0]))
    bob = world.add_body(_sphere(1.0, [1.0, 0.0, 0.0], 0.1))
    joint = world.add_joint(rigid.PyJoint(0, pivot, bob, [0.0, 0.0, 0.0]))
    for _ in range(60):
        world.step(DT)
    assert world.joint_anchor_gap(joint) < 1e-3
    x, y, z = world.get_body(bob).position
    assert y < -0.3
    assert abs(math.sqrt(x * x + y * y + z * z) - 1.0) < 0.01


def test_remove_body_removes_its_joints():
    rigid = ox.rigid
    world = rigid.PyWorld()
    a = world.add_body(_sphere(1.0, [0.0, 0.0, 0.0], 0.5))
    b = world.add_body(_sphere(1.0, [2.0, 0.0, 0.0], 0.5))
    world.add_joint(rigid.PyJoint(0, a, b, [1.0, 0.0, 0.0]))
    assert world.remove_body(a)
    assert world.body_count() == 1
    assert world.joint_count() == 0


def test_invalid_input_raises_value_error():
    rigid = ox.rigid
    world = rigid.PyWorld()
    with pytest.raises(ValueError):
        world.add_body(rigid.PyRigidBody(0, 0.0, [0.0, 0.0, 0.0]))
    a = world.add_body(rigid.PyRigidBody(0, 1.0, [0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        world.add_joint(rigid.PyJoint(0, a, 42, [0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        world.step(0.0)
