from dataclasses import replace
from distill.benchmarking.breakeven import Assumptions, breakeven_volume, student_monthly, teacher_monthly


def base():
    return Assumptions(teacher_usd_per_req=0.001, labeling_usd=50, train_gpu_hours=2,
                       gpu_hourly_usd=1.0, student_rps=20, utilization=0.5)


def test_breakeven_in_expected_range():
    v = breakeven_volume(base())
    assert 5e5 < v < 1e6        # ~ (730 + 52/12) / 0.001


def test_pricier_teacher_lowers_breakeven():
    a = base()
    assert breakeven_volume(replace(a, teacher_usd_per_req=0.002)) < breakeven_volume(a)


def test_free_teacher_never_breaks_even():
    assert breakeven_volume(replace(base(), teacher_usd_per_req=0.0)) is None


def test_student_wins_at_high_volume():
    a = base()
    assert student_monthly(a, 1e8) < teacher_monthly(a, 1e8)