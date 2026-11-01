
import time

from aienergy.meter import PowerMeter


def test_meter_has_compatible_attributes_after_run():
    meter = PowerMeter(interval=0.05)

    with meter:
        time.sleep(0.2)

    assert hasattr(meter, "joules")
    assert hasattr(meter, "avg_w")
    assert hasattr(meter, "method")
    assert hasattr(meter, "seconds")


def test_meter_runs_without_nvidia_gpu():
    meter = PowerMeter(interval=0.05)

    with meter:
        time.sleep(0.2)

    assert meter.seconds > 0
    assert meter.joules >= 0
    assert meter.avg_w >= 0


def test_meter_reports_sampling_coverage():
    meter = PowerMeter(interval=0.05)

    with meter:
        time.sleep(0.2)

    assert meter.seconds > 0
    assert meter._covered > 0
    assert meter._covered <= meter.seconds + 0.05
def test_make_row_subtracts_idle_energy_using_covered_time():
    from aienergy.meter import make_row

    meter = PowerMeter()
    meter.joules = 100.0
    meter._covered = 4.0
    meter.avg_w = 25.0
    meter.method = "cpu_util_model"

    row = make_row(
        backend="test",
        machine="test_machine",
        model="test_model",
        params_b=1,
        prompt_id="test",
        rep=1,
        prompt_tokens=10,
        out_tokens=20,
        wall_s=5.0,
        prefill_tok_s=10.0,
        gen_tok_s=5.0,
        m=meter,
        idle_w=10.0,
    )

    assert row["active_j"] == 60.0