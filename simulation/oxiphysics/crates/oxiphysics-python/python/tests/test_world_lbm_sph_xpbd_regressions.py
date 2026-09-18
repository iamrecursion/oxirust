# Copyright 2026 COOLJAPAN OU (Team KitaSan)
# SPDX-License-Identifier: Apache-2.0

"""Regression tests of the world LBM grid, the world SPH simulation and the
XPBD particle velocity.

* ``PyLbmConfig.lid_driven_cavity()`` used to be a periodic box without lid or
  walls, so the "cavity" stayed at rest forever.
* A body acceleration was injected into the LBM fluid at half its magnitude.
* ``XpbdSolver.particle_velocity`` returned the velocity divided by the number
  of substeps.
"""

from __future__ import annotations

import pytest

ox = pytest.importorskip("oxiphysics")


def _world():
    world = getattr(ox, "world", None)
    if world is None:
        pytest.skip("oxiphysics.world is not bound")
    return world


# ---------------------------------------------------------------------------
# LBM: walls and lid
# ---------------------------------------------------------------------------


def test_lbm_cavity_preset_is_a_closed_box_with_a_moving_lid():
    cfg = _world().PyLbmConfig.lid_driven_cavity()
    assert (cfg.width, cfg.height) == (64, 64)
    assert cfg.walls is True
    assert list(cfg.lid_velocity) == pytest.approx([0.1, 0.0])
    # Re = U H / nu = 100 for the cavity side H = 64 cells.
    assert cfg.lid_velocity[0] * cfg.height / cfg.viscosity == pytest.approx(100.0)


def test_lbm_plain_config_has_no_walls_and_no_lid():
    cfg = _world().PyLbmConfig(10, 8, 0.1)
    assert cfg.walls is False
    assert list(cfg.lid_velocity) == [0.0, 0.0]


def test_lbm_moving_lid_drives_the_fluid_and_conserves_mass():
    world = _world()
    cfg = world.PyLbmConfig(16, 16, 0.05)
    cfg.walls = True
    cfg.lid_velocity = [0.05, 0.0]
    grid = world.PyLbmGrid(cfg)
    for _ in range(300):
        grid.step()
    # The fluid below the lid moves along with it ...
    assert all(grid.velocity_at(x, 15)[0] > 0.0 for x in range(16))
    # ... and the closed box keeps its mass.
    assert sum(grid.density_field()) == pytest.approx(16 * 16, abs=1e-9)


def test_lbm_cavity_without_walls_is_not_driven_by_the_lid():
    world = _world()
    cfg = world.PyLbmConfig(12, 12, 0.05)
    cfg.lid_velocity = [0.05, 0.0]  # no walls: there is no lid
    grid = world.PyLbmGrid(cfg)
    for _ in range(50):
        grid.step()
    assert max(abs(u) for u in grid.velocity_field()) < 1e-12


# ---------------------------------------------------------------------------
# LBM: Guo body forcing and solid cells
# ---------------------------------------------------------------------------


def test_lbm_body_force_adds_its_full_acceleration_every_step():
    world = _world()
    cfg = world.PyLbmConfig(8, 6, 0.1)
    cfg.body_force = [1e-5, 0.0]
    grid = world.PyLbmGrid(cfg)
    assert grid.velocity_at(0, 0) == pytest.approx([0.0, 0.0], abs=1e-15)
    steps = 100
    for _ in range(steps):
        grid.step()
    # u = N a exactly (the old collision gave N a / 2).
    for y in range(6):
        for x in range(8):
            ux, uy = grid.velocity_at(x, y)
            assert ux == pytest.approx(steps * 1e-5, rel=1e-9)
            assert uy == pytest.approx(0.0, abs=1e-12)


def test_lbm_set_solid_marks_cells_and_rejects_cells_outside_the_grid():
    world = _world()
    grid = world.PyLbmGrid(world.PyLbmConfig(6, 4, 0.1))
    assert grid.is_solid(2, 1) is False
    assert grid.set_solid(2, 1, True) is True
    assert grid.is_solid(2, 1) is True
    assert grid.set_solid(6, 0, True) is False
    assert grid.velocity_at(2, 1) == pytest.approx([0.0, 0.0])
    assert grid.density_at(2, 1) == 0.0


# ---------------------------------------------------------------------------
# SPH
# ---------------------------------------------------------------------------


def test_sph_block_steps_with_finite_state():
    world = _world()
    sim = world.PySphSim(world.PySphConfig.water())
    sim.add_particle_block([0.0, 0.0, 0.0], 8, 8, 8, 0.05)
    assert sim.particle_count() == 512
    for _ in range(3):
        sim.step(1e-4)
    assert all(abs(c) < 1e3 for c in sim.all_positions())
    assert all(abs(c) < 1e3 for c in sim.all_velocities())


# ---------------------------------------------------------------------------
# XPBD
# ---------------------------------------------------------------------------


def test_xpbd_particle_velocity_is_gravity_times_time():
    solver = ox.XpbdSolver()
    solver.set_gravity(0.0, -9.81, 0.0)
    solver.set_substeps(10)
    idx = solver.add_particle([0.0, 100.0, 0.0], 1.0)
    solver.step(0.1)
    # The result used to be 0.0981, a factor `substeps` too small.
    assert solver.particle_velocity(idx, 0.1) == pytest.approx((0.0, -0.981, 0.0), abs=1e-9)
    # `dt` is optional, may be given by keyword, and is ignored.
    assert solver.particle_velocity(idx) == pytest.approx((0.0, -0.981, 0.0), abs=1e-9)
    assert solver.particle_velocity(idx, dt=0.1) == pytest.approx((0.0, -0.981, 0.0), abs=1e-9)
    assert solver.particle_velocity(idx, 12345.0) == pytest.approx((0.0, -0.981, 0.0), abs=1e-9)


@pytest.mark.parametrize("substeps", [1, 4, 10, 40])
def test_xpbd_particle_velocity_does_not_depend_on_the_substep_count(substeps):
    solver = ox.XpbdSolver()
    solver.set_substeps(substeps)
    idx = solver.add_particle([0.0, 100.0, 0.0], 1.0)
    for _ in range(4):
        solver.step(0.25)
    assert solver.particle_velocity(idx)[1] == pytest.approx(-9.81, abs=1e-9)


def test_xpbd_pinned_and_unstepped_particles_have_no_velocity():
    solver = ox.XpbdSolver()
    falling = solver.add_particle([0.0, 100.0, 0.0], 1.0)
    pinned = solver.add_particle([1.0, 100.0, 0.0], 1.0)
    assert list(solver.particle_velocity(falling)) == [0.0, 0.0, 0.0]
    solver.step(0.1)
    solver.pin_particle(pinned)
    assert solver.particle_velocity(falling)[1] < -0.9
    assert list(solver.particle_velocity(pinned)) == [0.0, 0.0, 0.0]
    assert list(solver.particle_velocity(99)) == [0.0, 0.0, 0.0]
