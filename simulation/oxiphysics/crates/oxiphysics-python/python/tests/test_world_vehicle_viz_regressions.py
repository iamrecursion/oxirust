"""Regression tests of the physics behind the Python world, vehicle, visualizer,
SPH and FEM bindings.

Each test pins one audited defect against an independent reference: an
analytic solution, a conservation law or a brute-force sum.

Covers: world constraints (distance, ball-socket, hinge and slider solved by the
engine's joints), the vehicle tyre model and planar dynamics, the principal
stresses of the stress visualizer, the neighbour-search SPH step and the 3-D
truss assembly.

Run after ``maturin develop`` via::

    python -m pytest python/tests/test_world_vehicle_viz_regressions.py -v
"""

import math
import random

import pytest

try:
    import oxiphysics as ox

    HAS_OXIPHYSICS = True
except ImportError:
    HAS_OXIPHYSICS = False

pytestmark = pytest.mark.skipif(not HAS_OXIPHYSICS, reason="oxiphysics not built")

GRAVITY = 9.81
DT = 1.0 / 240.0


def _norm(v):
    return math.sqrt(sum(c * c for c in v))


def _world(gravity=True):
    return ox.PhysicsWorld(0.0, -GRAVITY if gravity else 0.0, 0.0)


def _dynamic(mass, x, y, z):
    """Configuration of a dynamic body that never falls asleep."""
    config = ox.RigidBodyConfig.dynamic(mass, x, y, z)
    config.set_can_sleep(False)
    return config


def _constraint():
    return ox.world.PyConstraint


# ---------------------------------------------------------------------------
# World constraints
# ---------------------------------------------------------------------------


def test_pendulum_keeps_its_length_and_never_exceeds_the_free_fall_speed():
    world = _world()
    pivot = world.add_static_body(0.0, 0.0, 0.0)
    bob = world.add_rigid_body(_dynamic(1.0, 1.0, 0.0, 0.0))
    rod = _constraint().distance(0, pivot, bob, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0)
    handle = world.add_constraint(rod)
    assert handle == 0
    assert world.constraint_count() == 1

    limit = math.sqrt(2.0 * GRAVITY)
    worst_length_error = 0.0
    fastest = 0.0
    lowest = 0.0
    for _ in range(1000):
        world.step(DT)
        position = world.get_position(bob)
        speed = _norm(world.get_velocity(bob))
        worst_length_error = max(worst_length_error, abs(_norm(position) - 1.0))
        fastest = max(fastest, speed)
        lowest = min(lowest, position[1])
    assert worst_length_error < 1e-3
    assert fastest <= limit
    assert fastest > 0.9 * limit
    assert lowest < -0.99


def test_hinge_swings_in_the_plane_perpendicular_to_its_axis():
    world = _world()
    wall = world.add_static_body(0.0, 0.0, 0.0)
    door = world.add_rigid_body(_dynamic(1.0, 1.0, 0.0, 0.0))
    hinge = _constraint().hinge(0, wall, door, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    world.add_constraint(hinge)
    lowest = 0.0
    for _ in range(600):
        world.step(DT)
        x, y, z = world.get_position(door)
        assert abs(z) < 1e-9
        assert abs(math.hypot(x, y) - 1.0) < 1e-3
        spin = world.get_angular_velocity(door)
        assert abs(spin[0]) < 1e-9 and abs(spin[1]) < 1e-9
        lowest = min(lowest, y)
    assert lowest < -0.99


def test_slider_moves_along_its_rail_only():
    world = _world()
    rail = world.add_static_body(0.0, 0.0, 0.0)
    carriage = world.add_rigid_body(_dynamic(2.0, 1.0, 0.0, 0.0))
    slider = _constraint().slider(0, rail, carriage, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    slider.stiffness = 1.0
    world.add_constraint(slider)
    # 3 N s on 2 kg: 1.5 m/s along the rail, gravity is carried by the rail.
    world.apply_impulse(carriage, 3.0, 0.0, 0.0)
    for _ in range(240):
        world.step(DT)
    x, y, z = world.get_position(carriage)
    assert abs(x - 2.5) < 1e-3
    assert abs(y) < 1e-3 and abs(z) < 1e-3
    vx, vy, vz = world.get_velocity(carriage)
    assert abs(vx - 1.5) < 1e-9
    assert abs(vy) < 1e-9 and abs(vz) < 1e-9
    assert _norm(world.get_angular_velocity(carriage)) < 1e-9


def test_point_to_point_conserves_momentum_and_keeps_the_pivot_together():
    world = _world(gravity=False)
    first = world.add_rigid_body(_dynamic(1.0, 0.0, 0.0, 0.0))
    second = world.add_rigid_body(_dynamic(2.0, 2.0, 0.0, 0.0))
    world.set_velocity(first, 0.0, 1.0, 0.0)
    world.set_velocity(second, 0.0, -0.5, 0.0)
    world.add_constraint(_constraint().point_to_point(0, first, second, [1.0, 0.0, 0.0]))
    for _ in range(480):
        world.step(DT)
    v1 = world.get_velocity(first)
    v2 = world.get_velocity(second)
    momentum = [1.0 * v1[i] + 2.0 * v2[i] for i in range(3)]
    assert _norm(momentum) < 1e-9
    # Both bodies hang on the pivot: with equal and opposite lever arms (1, 0, 0)
    # and (-1, 0, 0) their centres stay symmetric about it up to the rotations.
    p1 = world.get_position(first)
    p2 = world.get_position(second)
    assert _norm([p1[i] - p2[i] for i in range(3)]) <= 2.0 + 1e-3


def test_constraint_errors_are_python_exceptions():
    world = _world()
    a = world.add_rigid_body(_dynamic(1.0, 0.0, 0.0, 0.0))
    b = world.add_rigid_body(_dynamic(1.0, 1.0, 0.0, 0.0))
    constraint = _constraint()
    with pytest.raises(IndexError):
        world.add_constraint(constraint.point_to_point(0, a, 99, [0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        world.add_constraint(constraint.point_to_point(0, a, a, [0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        world.add_constraint(constraint.hinge(0, a, b, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        world.add_constraint(constraint.slider(0, a, b, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        world.add_constraint(
            constraint.distance(0, a, b, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], -1.0)
        )
    with pytest.raises(ValueError):
        world.add_constraint(
            constraint.point_to_point(0, a, b, [0.0, float("inf"), 0.0])
        )
    assert world.constraint_count() == 0

    rod = constraint.distance(0, a, b, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0)
    assert rod.stiffness == pytest.approx(0.5)
    rod.stiffness = 0.75
    assert rod.stiffness == pytest.approx(0.75)
    with pytest.raises(ValueError):
        rod.stiffness = 1.5
    with pytest.raises(ValueError):
        rod.stiffness = float("nan")
    assert rod.stiffness == pytest.approx(0.75)
    assert rod.constraint_type == ox.world.ConstraintType.Distance

    handle = world.add_constraint(rod)
    assert world.constraint_count() == 1
    assert world.remove_constraint(handle)
    assert not world.remove_constraint(handle)
    assert world.constraint_count() == 0


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------


def _drive(vehicle, seconds, throttle, brake=0.0, steer=0.0, dt=0.01):
    for _ in range(round(seconds / dt)):
        vehicle.step(dt, throttle, brake, steer)


def test_sedan_throttle_accelerates_the_chassis_through_the_tyres():
    vehicle = ox.vehicle.PyVehicle.sedan()
    _drive(vehicle, 2.0, 1.0)
    assert vehicle.forward_speed() > 1.0
    assert vehicle.position[0] > 1.0
    # The driven (rear) wheels slip forward, the tyre forces are not zero.
    rear = vehicle.wheel(2)
    assert rear.slip_ratio > 0.0
    assert rear.fx > 0.0
    # Without throttle the sedan stays where it is.
    parked = ox.vehicle.PyVehicle.sedan()
    _drive(parked, 2.0, 0.0)
    assert abs(parked.forward_speed()) < 1e-6


def test_sedan_brakes_to_a_stop_without_rolling_back():
    vehicle = ox.vehicle.PyVehicle.sedan()
    _drive(vehicle, 3.0, 1.0)
    assert vehicle.forward_speed() > 1.0
    _drive(vehicle, 20.0, 0.0, brake=1.0)
    assert -1e-3 <= vehicle.forward_speed() < 0.05


def test_sedan_steering_yaws_the_vehicle_toward_the_steered_side():
    left = ox.vehicle.PyVehicle.sedan()
    _drive(left, 2.0, 0.6)
    _drive(left, 1.0, 0.3, steer=0.5)
    assert left.orientation[2] > 0.05
    assert left.angular_velocity[2] > 0.0
    right = ox.vehicle.PyVehicle.sedan()
    _drive(right, 2.0, 0.6)
    _drive(right, 1.0, 0.3, steer=-0.5)
    # Mirror symmetry of the two turns.
    assert right.orientation[2] == pytest.approx(-left.orientation[2], rel=1e-6)


def _argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def test_pacejka_forces_peak_at_realistic_slips():
    slips = [i * 0.002 for i in range(1, 251)]
    lateral = ox.vehicle.PacejkaCoeffs.road_lateral()
    fy = [ox.vehicle.pacejka_fy(a, lateral.b, lateral.c, lateral.d, lateral.e) for a in slips]
    assert 0.08 <= slips[_argmax(fy)] <= 0.2
    longitudinal = ox.vehicle.PacejkaCoeffs.road_longitudinal()
    fx = [
        ox.vehicle.pacejka_fx(k, longitudinal.b, longitudinal.c, longitudinal.d, longitudinal.e)
        for k in slips
    ]
    assert 0.08 <= slips[_argmax(fx)] <= 0.2
    # The forces have the sign of the slips.
    assert ox.vehicle.pacejka_fy(-0.05, 10.0, 1.9, 1.0, 0.97) < 0.0
    assert ox.vehicle.pacejka_fx(-0.05, 11.0, 1.65, 1.0, 0.0) < 0.0


# ---------------------------------------------------------------------------
# Stress visualizer
# ---------------------------------------------------------------------------


def test_stress_visualizer_pure_shear_has_plus_minus_tau_principal_stresses():
    tau = 50.0
    visualizer = ox.viz.PyStressVisualizer([ox.viz.StressTensor(0.0, 0.0, 0.0, tau, 0.0, 0.0)])
    out = visualizer.compute(0)
    assert list(out.principal_stresses) == pytest.approx([tau, 0.0, -tau], abs=1e-9)
    # sigma_1 acts along (1, 1, 0) / sqrt 2, up to sign.
    first = list(out.principal_directions[0:3])
    root = 1.0 / math.sqrt(2.0)
    assert [abs(c) for c in first] == pytest.approx([root, root, 0.0], abs=1e-9)
    assert first[0] * first[1] > 0.0
    # Mohr circle of the extreme principal stresses: centre 0, radius tau.
    assert out.mohr_circle[4] == pytest.approx(0.0, abs=1e-9)
    assert out.mohr_circle[5] == pytest.approx(tau, abs=1e-9)
    assert out.von_mises == pytest.approx(math.sqrt(3.0) * tau)


def test_stress_visualizer_reproduces_a_diagonal_tensor_and_rejects_bad_input():
    visualizer = ox.viz.PyStressVisualizer(
        [
            ox.viz.StressTensor(2.0, -1.0, 3.0, 0.0, 0.0, 0.0),
            ox.viz.StressTensor(float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0),
        ]
    )
    out = visualizer.compute(0)
    assert list(out.principal_stresses) == pytest.approx([3.0, 2.0, -1.0], abs=1e-12)
    # sigma_1 = s33 along z, sigma_2 = s11 along x, sigma_3 = s22 along y.
    expected = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert [abs(c) for c in out.principal_directions] == pytest.approx(expected, abs=1e-12)
    assert visualizer.compute(1) is None
    assert visualizer.compute(7) is None


def test_stress_visualizer_principal_values_of_a_general_tensor():
    rng = random.Random(7)
    for _ in range(20):
        s = [rng.uniform(-100.0, 100.0) for _ in range(6)]
        out = ox.viz.PyStressVisualizer([ox.viz.StressTensor(*s)]).compute(0)
        s1, s2, s3 = out.principal_stresses
        assert s1 >= s2 >= s3
        s11, s22, s33, s12, s13, s23 = s
        # The invariants of the tensor are those of its principal stresses.
        assert s1 + s2 + s3 == pytest.approx(s11 + s22 + s33, abs=1e-9)
        second = s11 * s22 + s22 * s33 + s33 * s11 - s12**2 - s13**2 - s23**2
        assert s1 * s2 + s2 * s3 + s3 * s1 == pytest.approx(second, abs=1e-6)
        det = (
            s11 * (s22 * s33 - s23**2)
            - s12 * (s12 * s33 - s23 * s13)
            + s13 * (s12 * s23 - s22 * s13)
        )
        assert s1 * s2 * s3 == pytest.approx(det, rel=1e-9, abs=1e-6)
        # Each row is a unit eigenvector: T d = sigma d.
        matrix = [[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]]
        for k, sigma in enumerate((s1, s2, s3)):
            d = out.principal_directions[3 * k : 3 * k + 3]
            assert _norm(d) == pytest.approx(1.0, abs=1e-12)
            for row in range(3):
                assert sum(matrix[row][c] * d[c] for c in range(3)) == pytest.approx(
                    sigma * d[row], abs=1e-9
                )


# ---------------------------------------------------------------------------
# SPH
# ---------------------------------------------------------------------------


def test_sph_densities_match_the_all_pairs_sum():
    rng = random.Random(3)
    simulation = ox.sph.PySphSimulation(ox.sph.PySphConfig.water())
    # The values of PySphConfig.water().
    h, mass, stiffness, rest_density = 0.1, 0.02, 200.0, 1000.0
    points = [[rng.uniform(0.0, 0.5) for _ in range(3)] for _ in range(250)]
    for point in points:
        simulation.add_particle(point)
    # A zero-length step evaluates the densities of the current positions.
    simulation.step(0.0)
    coefficient = 315.0 / (64.0 * math.pi * h**9)
    expected = []
    for i in points:
        rho = 0.0
        for j in points:
            r2 = sum((i[k] - j[k]) ** 2 for k in range(3))
            if r2 < h * h:
                rho += mass * coefficient * (h * h - r2) ** 3
        expected.append(max(rho, 1e-3))
    actual = simulation.get_density_field()
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a == pytest.approx(e, rel=1e-12, abs=1e-12)
    # The pressure follows the linear equation of state.
    pressure = simulation.get_pressure_field()
    for rho, p in zip(actual, pressure):
        assert p == pytest.approx(stiffness * (rho / rest_density - 1.0), rel=1e-12)


# ---------------------------------------------------------------------------
# FEM truss
# ---------------------------------------------------------------------------

EA = 1.0e6


def _two_bar_truss():
    truss = ox.world.PyFemAssembly.from_coords([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    truss.add_bar_element(0, 2, EA)
    truss.add_bar_element(1, 2, EA)
    for dof in range(6):
        truss.fix_dof(dof)
    return truss


def test_two_bar_truss_member_forces_follow_statics():
    load = 1000.0
    truss = _two_bar_truss()
    truss.apply_force(3 * 2 + 1, -load)
    assert truss.solve()
    for bar in range(2):
        assert truss.element_force(bar) == pytest.approx(-load / math.sqrt(2.0), rel=1e-9)
    assert truss.displacement(3 * 2 + 1) == pytest.approx(-load * math.sqrt(2.0) / EA, rel=1e-9)
    assert truss.displacement(3 * 2) == pytest.approx(0.0, abs=1e-15)
    # The load is still there: solving does not erase the user's forces.
    assert truss.force(3 * 2 + 1) == pytest.approx(-load)
    assert truss.solve()
    assert truss.element_force(0) == pytest.approx(-load / math.sqrt(2.0), rel=1e-9)


def test_truss_rejects_a_load_no_bar_can_carry():
    truss = _two_bar_truss()
    # The truss lies in the x-y plane: nothing stiffens the z direction.
    truss.apply_force(3 * 2 + 2, 10.0)
    with pytest.raises(ValueError):
        truss.solve()
    assert truss.force(3 * 2 + 2) == pytest.approx(10.0)
    with pytest.raises(IndexError):
        truss.apply_force(100, 1.0)
