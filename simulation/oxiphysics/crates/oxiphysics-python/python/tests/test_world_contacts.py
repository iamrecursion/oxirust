# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``oxiphysics.PhysicsWorld`` contacts.

Covers the impulse direction (approaching bodies bounce, separating bodies are
left alone), Coulomb friction, mass-weighted positional correction, non-sphere
colliders (boxes, capsules, planes) and per-collider ray casts.
"""

import math

import pytest

oxiphysics = pytest.importorskip("oxiphysics")


def _zero_g_world():
    cfg = oxiphysics.SimConfig.zero_gravity()
    cfg.sleep_enabled = False
    return oxiphysics.PhysicsWorld.from_config(cfg)


def _sphere(world, mass, x, y, z, radius, restitution=0.3, friction=0.5):
    cfg = oxiphysics.RigidBodyConfig.dynamic(mass, x, y, z)
    cfg.add_sphere(radius)
    cfg.set_restitution(restitution)
    cfg.set_friction(friction)
    return world.add_rigid_body(cfg)


def _ground_box(world, hy=0.5, friction=0.5):
    cfg = oxiphysics.RigidBodyConfig.static_body(0.0, 0.0, 0.0)
    cfg.add_box(20.0, hy, 20.0)
    cfg.set_friction(friction)
    return world.add_rigid_body(cfg)


def test_elastic_head_on_spheres_swap_velocities():
    world = _zero_g_world()
    a = _sphere(world, 1.0, 0.0, 0.0, 0.0, 1.0, restitution=1.0)
    b = _sphere(world, 1.0, 1.9, 0.0, 0.0, 1.0, restitution=1.0)
    world.set_velocity(a, 1.0, 0.0, 0.0)
    world.set_velocity(b, -1.0, 0.0, 0.0)
    world.step(1e-3)
    assert world.get_velocity(a)[0] == pytest.approx(-1.0, abs=1e-9)
    assert world.get_velocity(b)[0] == pytest.approx(1.0, abs=1e-9)
    (contact,) = world.get_contacts()
    assert contact.normal[0] == pytest.approx(1.0)
    assert contact.impulse == pytest.approx(2.0)


def test_separating_pair_is_not_pulled_back():
    world = _zero_g_world()
    a = _sphere(world, 1.0, 0.0, 0.0, 0.0, 1.0, restitution=1.0)
    b = _sphere(world, 1.0, 1.5, 0.0, 0.0, 1.0, restitution=1.0)
    world.set_velocity(a, -1.0, 0.0, 0.0)
    world.set_velocity(b, 1.0, 0.0, 0.0)
    world.step(1e-3)
    assert world.get_velocity(a) == (-1.0, 0.0, 0.0)
    assert world.get_velocity(b) == (1.0, 0.0, 0.0)


def test_light_bodies_are_not_flung_apart():
    world = _zero_g_world()
    a = _sphere(world, 0.01, 0.0, 0.0, 0.0, 1.0)
    b = _sphere(world, 0.01, 1.9, 0.0, 0.0, 1.0)
    world.step(1.0 / 60.0)
    gap = world.get_position(b)[0] - world.get_position(a)[0]
    # beta * (depth - slop) with the defaults beta = 0.2, slop = 0.001.
    assert gap - 1.9 == pytest.approx(0.2 * (0.1 - 0.001), abs=1e-12)


def test_sphere_dropped_on_static_box_rests_on_it():
    world = oxiphysics.PhysicsWorld(gy=-9.81)
    _ground_box(world)
    ball = _sphere(world, 1.0, 0.0, 3.0, 0.0, 0.5)
    for _ in range(480):
        world.step(1.0 / 120.0)
    y = world.get_position(ball)[1]
    assert 0.99 < y <= 1.0 + 1e-6


def test_box_body_lands_on_box_ground():
    world = oxiphysics.PhysicsWorld(gy=-9.81)
    _ground_box(world)
    top = world.add_box_body(1.0, 0.0, 2.0, 0.0, 0.5, 0.5, 0.5)
    for _ in range(480):
        world.step(1.0 / 120.0)
    assert 0.99 < world.get_position(top)[1] <= 1.0 + 1e-6


def test_friction_makes_a_sliding_sphere_roll():
    def slide(mu):
        cfg = oxiphysics.SimConfig.earth_gravity()
        cfg.sleep_enabled = False
        world = oxiphysics.PhysicsWorld.from_config(cfg)
        _ground_box(world, friction=mu)
        ball = _sphere(world, 1.0, 0.0, 0.999, 0.0, 0.5, restitution=0.0, friction=mu)
        world.set_velocity(ball, 2.0, 0.0, 0.0)
        for _ in range(240):
            world.step(1.0 / 240.0)
        return world.get_velocity(ball)[0], world.get_angular_velocity(ball)[2]

    vx, wz = slide(1.0)
    assert vx == pytest.approx(2.0 * 5.0 / 7.0, rel=0.03)
    assert vx == pytest.approx(-wz * 0.5, abs=0.02)
    vx, wz = slide(0.0)
    assert vx == pytest.approx(2.0, abs=1e-9)
    assert wz == pytest.approx(0.0, abs=1e-9)


def test_raycast_hits_box_and_capsule_colliders():
    world = _zero_g_world()
    cfg = oxiphysics.RigidBodyConfig.static_body(0.0, 0.0, 5.0)
    cfg.add_box(1.0, 1.0, 1.0)
    boxed = world.add_rigid_body(cfg)
    handle, dist = world.raycast(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0)
    assert handle == boxed
    assert dist == pytest.approx(4.0)

    capsule = world.add_capsule_body(1.0, 3.0, 0.0, 0.0, 0.5, 1.0)
    handle, dist = world.raycast(3.0, 10.0, 0.0, 0.0, -1.0, 0.0, 100.0)
    assert handle == capsule
    assert dist == pytest.approx(8.5)

    half = math.pi / 8.0
    world.set_orientation(boxed, 0.0, math.sin(half), 0.0, math.cos(half))
    _, dist = world.raycast(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0)
    assert dist == pytest.approx(5.0 - math.sqrt(2.0))
