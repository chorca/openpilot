from types import SimpleNamespace

from opendbc.car import gen_empty_fingerprint
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, long_control_state_trans


class TestLongControlStateTransition(OpenpilotTestCase):

  def test_stay_stopped(self):
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off

  def test_engage(self):
    active = True
    current_state = LongCtrlState.off
    next_state = long_control_state_trans(active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid

  def test_should_stop_hysteresis_and_brake_authority(self):
    # Enter stopping only below 0.3 m/s, hold stopping below 0.8 m/s while a_target < 0.1,
    # and release immediately when a_target >= 0.1 (standstill launch).
    assert not should_stop(0.35, -0.15, keep_stopping=False)
    assert should_stop(0.28, -0.15, keep_stopping=False)
    assert should_stop(0.55, -0.15, keep_stopping=True)
    assert not should_stop(0.85, -0.15, keep_stopping=True)
    assert not should_stop(0.0, 0.12, keep_stopping=True)

    # Stopping state must never reduce brake authority below a_target.
    cp = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    loc = LongControl(cp)
    loc.long_control_state = LongCtrlState.stopping
    loc.last_output_accel = -0.25
    cs = SimpleNamespace(vEgo=0.25, aEgo=0.0, brakePressed=False, cruiseState=SimpleNamespace(standstill=False))
    out = loc.update(True, cs, -2.4, True, [-3.5, 2.0])
    assert out <= -2.4
