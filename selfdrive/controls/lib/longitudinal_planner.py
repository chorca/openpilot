#!/usr/bin/env python3
import math
import numpy as np

import cereal.messaging as messaging
from cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource, get_T_FOLLOW
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5
J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MIN = -1.2

# Cruise speed control (planner side, upstream #38367 moved it out of the MPC).
# The bare clip(v_cruise - v_ego, A_CRUISE_MIN, max_accel) that #38367 shipped is a
# unity-gain proportional law whose proportional band is only max_accel wide (~0.7 m/s at
# highway speed), so with the car's actuator lag it saturates, overshoots, brakes and saws
# around the set speed. Damped P + slow integral instead:
CRUISE_KP = 0.4          # speed error -> accel
CRUISE_KD = 0.4          # measured accel feedback, damps the actuator lag
CRUISE_KI = 0.06         # slow integral, removes sag on grades
CRUISE_INT_CLIP = 1.5    # m/s*s, anti-windup clamp on the integral state
CRUISE_KD_FILTER = 0.5   # s, low-pass on aEgo before it is used for damping

# Lead-aware target speed: while a lead is tracked, the goal is no longer the set speed.
LEAD_PROB_GATE = 0.5     # modelProb above which a lead is considered real
LEAD_D_STANDSTILL = 4.0  # m, standstill gap
LEAD_T_GAP = 1.45        # s, fallback time gap; the live value comes from get_T_FOLLOW(personality)
LEAD_TAU = 3.0           # s, time allowed to close the remaining gap
LEAD_DV_ALLOW = 2.0      # m/s, max approach speed over a lead slower than the set speed.
                         #   Too small and the car never closes to its follow distance
                         #   (measured: 0.5 m/s left it 78 m back where the stock planner
                         #   settles ~6 m); the taper below is what keeps the approach gentle.
COAST_MIN_SPEED = 2.5    # m/s, below this the car brakes instead of coasting, otherwise the
                         #   coast band would leave it creeping and unable to come to a stop
LAUNCH_SPEED = 2.5       # m/s (~5.5 mph): starting or creeping, not cruising. The rise caps
                         #   exist for highway kickdown and at a standstill stop the car
                         #   pulling away at all (suite: "resume from a stop").
# Personality: the car is an automatic, so a step change in the accel request causes kickdown
# (a second or two of nothing, then a surge). The request is therefore only allowed to build
# slowly, and overspeed inside the coast band is absorbed by coasting (no throttle, no brake).
# The accel ceiling itself stays the car's own curve: capping below it made a clear-road
# catch-up ask for about half of stock, which the driver feels immediately.
def personality_name(personality) -> str:
  """Personality as a name, tolerating both plain ints and capnp enum objects.

  These tables were keyed by the schema-level enum members and looked up with the enum
  carried on the selfdriveState message. Those do not compare equal, so every lookup missed
  and silently fell back: the per-personality tuning was dead code and the fallback coast
  band (2.0 m/s) stopped the brake finishing a stop. Key by name instead.
  """
  try:
    return {0: "aggressive", 1: "standard", 2: "relaxed"}[int(personality)]
  except (TypeError, ValueError, KeyError):
    s = str(personality)
    return s if s in ("aggressive", "standard", "relaxed") else "standard"


# Unknown/absent personality must not silently remove the tuning
FALLBACK_COAST_BAND = 2.0
PERSONALITY_ACCEL_RISE_SCALE = {  # multiplier on the stock cruise jerk profile
  # The car's own cruise law ramps the request at J_CRUISE_VALS (1.6 m/s^3 at rest tapering
  # to 0.6 at highway speed). Absolute caps (0.3 m/s^3 in relaxed) made a re-acceleration
  # after a slowdown take ~15 s to ask for anything useful - measured on a drive as the
  # driver taking over with the pedal 4 s after the car started asking. Scaling stock keeps
  # some kickdown protection without making the car feel dead.
  "relaxed": 0.8,
  "standard": 1.0,
  "aggressive": 1.2,
}


def accel_rise_limit(v_ego, personality) -> float:
  """Rise limit for the accel request (m/s^3): the car's own cruise jerk profile, scaled by
  personality. Below LAUNCH_SPEED the scale is dropped so a standstill launch can ask to move
  at all."""
  scale = 1.0 if v_ego < LAUNCH_SPEED else PERSONALITY_ACCEL_RISE_SCALE.get(personality_name(personality), 1.0)
  return float(np.interp(v_ego, A_CRUISE_MAX_BP, J_CRUISE_VALS)) * scale
PERSONALITY_COAST_BAND = {        # m/s of overspeed handled by coasting instead of braking
  "relaxed": 2.0,
  "standard": 1.5,
  "aggressive": 1.0,
}
RELEASE_RATE = 0.8                # m/s^3, how fast braking may be released (still a ramp)
LIGHT_BRAKE_KP = 0.45             # per m/s of overspeed beyond the coast band
LIGHT_BRAKE_MAX = 1.2             # m/s^2, this law never brakes harder than this
CRUISE_KP_LEAD = 0.3              # gain for the lead-limited part of the target (slower and
                                  #   steadier than the speed loop: it runs at long range)

LEAD_DREL_FILTER_TAU = 0.3  # s, low-pass on the lead's gap (kept short: a lagged gap inflates
                            #   the closing allowance and turns a gentle approach into throttle)
LEAD_VLEAD_FILTER_TAU = 0.7 # s, low-pass on the lead's speed (vision lead jitters several m/s)
LEAD_RESET_JUMP = 10.0   # m, a jump larger than this is a different object: reset the filters
TARGET_RISE_RATE = 0.5   # m/s^2, how fast the target speed may rise when a lead is lost:
                         #   stops a lead flickering out of view from snapping the target (and
                         #   the command) back to the set speed
TARGET_RISE_RATE_LAUNCH = 3.0  # m/s^2, the same limit while pulling away: 0.5 m/s^2 needs
                               #   ~18 s to ask for 20 mph, which reads as refusing to move.
TARGET_FALL_RATE = 2.0   # m/s^2, how fast it may fall when a lead appears (stay responsive)
LEAD_LOSS_HOLD = 2.0     # s, keep using the last lead this long when it briefly stops being
                         #   reported, extrapolating the gap; vision leads drop out for a
                         #   second at a time and without this the car surges each time

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


def get_lead_target_speed(v_cruise, v_ego, d_rel, v_lead, t_gap=LEAD_T_GAP):
  """Target speed while a lead is tracked: never approach a slower lead faster than
  LEAD_DV_ALLOW, tapering to the lead's own speed as the gap reaches the desired time gap.

  Without this, "cruise" asked for the set speed even with a car 60 m ahead, so the planner
  commanded acceleration at a lead it could clearly see and only the MPC's close-range
  obstacle ever braked for it.

  d_rel/v_lead are the filtered lead measurements (radarState.leadOne is vision-only on
  this platform and jitters by several metres frame to frame); None means no lead.
  """
  if d_rel is None or v_lead is None:
    return v_cruise
  d_rel, v_lead = float(d_rel), float(v_lead)
  if not (math.isfinite(d_rel) and math.isfinite(v_lead)) or d_rel <= 0.:
    return v_cruise
  d_safe = LEAD_D_STANDSTILL + t_gap * v_ego
  allowance = float(np.clip((d_rel - d_safe) / LEAD_TAU, 0.0, LEAD_DV_ALLOW))
  return max(0.0, min(v_cruise, v_lead + allowance))


def get_cruise_accel(e2e, v_target, v_ego, a_ego, a_cruise_prev, angle_steers, CP, dt,
                     accel_coast, allow_throttle, integ, personality=log.LongitudinalPersonality.standard,
                     v_cruise=None):
  """Cruise-speed command, computed outside the MPC (upstream #38367).

  Damped proportional + slow integral on (v_target - v_ego), where v_target is the lead-aware
  target speed. The damping uses the *measured* accel - the actuator overshoots the request
  during transients, which is what turned the plain proportional law into a saw-tooth - and is
  one-sided (only while a_ego > 0): damping a decelerating car would add acceleration and cancel
  the brake request, and a law that will not hold a stop is a safety problem. `integ` is a
  one-element list holding the integral state, clamped for anti-windup.
  """
  _pers = personality_name(personality)
  max_accel = ACCEL_MAX if e2e else get_max_accel(v_ego)
  damping = -CRUISE_KD * max(a_ego, 0.0)   # one-sided: never softens a brake request

  if not e2e:
    a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
    a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
    a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
    max_accel = min(max_accel, a_x_allowed)
    if not allow_throttle:
      clipped_accel_coast = max(accel_coast, ACCEL_MIN)
      coast_limit = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [max_accel, clipped_accel_coast])
      max_accel = min(max_accel, coast_limit)

  # Two different reasons to slow down, handled differently:
  #  * being over the SET SPEED is relaxed - coast, and only brake well over it;
  #  * a lead-limited target is not relaxed - a slower car ahead needs a controlled approach,
  #    so it gets the proportional law, which starts easing off from far away instead of
  #    letting the car run up on it. Taking min() of the two keeps braking when the lead
  #    asks for it and coasting when only the set speed is exceeded.
  err = v_target - v_ego
  err_speed = (v_cruise - v_ego) if v_cruise is not None else err
  coast_band = PERSONALITY_COAST_BAND.get(_pers, FALLBACK_COAST_BAND)
  if v_cruise is not None and v_cruise <= 1e-3:
    # explicit stop request: no coasting. The light brake scales with overspeed-minus-band, so
    # with a band in force it fades out near the band and the car crawls instead of stopping.
    coast_band = 0.0

  if err_speed < 0.:
    overspeed = -err_speed
    if overspeed <= coast_band and v_ego > COAST_MIN_SPEED:
      a_speed = 0.0                     # coast: no throttle, no brake
      coasting = True
    elif overspeed <= coast_band:
      # creeping below the coast floor: plain proportional braking so the car can still stop
      a_speed = min(0.0, CRUISE_KP * err_speed + CRUISE_KI * integ[0] + damping)
      coasting = False
    else:
      a_speed = min(0.0, -min(LIGHT_BRAKE_MAX, LIGHT_BRAKE_KP * (overspeed - coast_band)) + damping)
      coasting = False
  else:
    a_speed = CRUISE_KP * err_speed + CRUISE_KI * integ[0] + damping
    coasting = False

  lead_limited = (v_cruise is not None) and (v_target < v_cruise - 1e-6)
  if lead_limited:
    a_lead = CRUISE_KP_LEAD * err + CRUISE_KI * integ[0] + damping
    target_accel = min(a_speed, a_lead)
  else:
    target_accel = a_speed

  # Integrate only when the integral term is actually driving the command, and freeze it
  # while the command is saturated (otherwise it winds up and biases the next phase).
  if not (coasting and not lead_limited) and not (target_accel >= max_accel or target_accel <= A_CRUISE_MIN):
    integ[0] = float(np.clip(integ[0] + err * dt, -CRUISE_INT_CLIP, CRUISE_INT_CLIP))
  target_accel = float(np.clip(target_accel, A_CRUISE_MIN, max_accel))

  # Ramp: releasing the brake may be brisk, but the accel request itself only ever builds
  # slowly — a step up in the request is what makes the automatic kick down and then surge.
  j_fall = float(np.interp(v_ego, A_CRUISE_MAX_BP, J_CRUISE_VALS))
  j_rise = RELEASE_RATE if a_cruise_prev < 0. else accel_rise_limit(v_ego, _pers)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_fall * dt, a_cruise_prev + j_rise * dt))

  return target_accel


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.a_cruise = init_a
    self.cruise_integ = [0.0]
    self.a_ego_filter = FirstOrderFilter(0.0, CRUISE_KD_FILTER, self.dt)
    self.lead_drel_filter = FirstOrderFilter(0.0, LEAD_DREL_FILTER_TAU, self.dt)
    self.lead_vlead_filter = FirstOrderFilter(0.0, LEAD_VLEAD_FILTER_TAU, self.dt)
    self.lead_seen = False
    self.lead_hold_t = 0.0
    self.v_target_prev = init_v
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_a_target_prev = 0.0
    self.output_should_stop = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  @staticmethod
  def parse_model(model_msg):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
      len(model_msg.velocity.x) == ModelConstants.IDX_N and
      len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm):
    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    personality = sm['selfdriveState'].personality
    _pers = personality_name(personality)

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    reset_state = reset_state or not v_cruise_initialized

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])
      self.a_cruise = self.a_desired
      self.cruise_integ[0] = 0.0
      self.a_ego_filter.x = sm['carState'].aEgo
      self.lead_seen = False
      self.lead_hold_t = 0.0
      # the target speed is NOT reset here: this branch also fires before the cruise speed is
      # initialised, and pinning the target to the current speed leaves the car unable to ask
      # to move off. The limiter below bounds it, and the command is clipped to aEgo anyway.
      self.output_a_target_prev = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    if force_slow_decel:
      v_cruise = 0.0

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(sm['radarState'], personality=sm['selfdriveState'].personality)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                                                        action_t=action_t, vEgoStopping=self.CP.vEgoStopping)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    # Cruise command now lives in the planner (upstream #38367), not in the MPC. The target
    # speed it tracks is lead-aware, so the car stops accelerating at cars it can see. The
    # lead's gap/speed are low-passed first: this platform's lead comes from the model and
    # jitters several metres frame to frame, which would otherwise appear in the command.
    lead = sm['radarState'].leadOne
    lead_tracked = (lead.status and lead.modelProb >= LEAD_PROB_GATE
                    and math.isfinite(lead.dRel) and lead.dRel > 0. and math.isfinite(lead.vLead))
    if lead_tracked:
      if not self.lead_seen or abs(lead.dRel - self.lead_drel_filter.x) > LEAD_RESET_JUMP:
        self.lead_drel_filter.x = lead.dRel
        self.lead_vlead_filter.x = lead.vLead
      lead_d_rel = self.lead_drel_filter.update(lead.dRel)
      lead_v_lead = self.lead_vlead_filter.update(lead.vLead)
      self.lead_seen = True
      self.lead_hold_t = 0.0
    elif self.lead_seen and self.lead_hold_t < LEAD_LOSS_HOLD:
      # brief dropout: keep pacing the last known lead with its gap extrapolated
      self.lead_hold_t += self.dt
      self.lead_drel_filter.x = max(0.5, self.lead_drel_filter.x + (self.lead_vlead_filter.x - v_ego) * self.dt)
      lead_d_rel, lead_v_lead = self.lead_drel_filter.x, self.lead_vlead_filter.x
    else:
      self.lead_seen = False
      lead_d_rel = lead_v_lead = None

    v_target_raw = get_lead_target_speed(v_cruise, v_ego, lead_d_rel, lead_v_lead,
                                         t_gap=get_T_FOLLOW(personality))
    # rate-limit the target speed so a lead flickering out of view cannot snap it back to the
    # set speed. An explicit stop request (set speed zero, e.g. forceDecel) is NOT limited:
    # it has to bite now, not decay over the seconds a 2 m/s^2 fall limit would take.
    if v_cruise <= 1e-3:
      self.v_target_prev = 0.0
    else:
      self.v_target_prev = float(np.clip(v_target_raw, self.v_target_prev - TARGET_FALL_RATE * self.dt,
                                         self.v_target_prev + (TARGET_RISE_RATE if v_ego >= LAUNCH_SPEED
                                                               else TARGET_RISE_RATE_LAUNCH) * self.dt))
    v_target = self.v_target_prev
    a_ego_filt = self.a_ego_filter.update(sm['carState'].aEgo)
    self.a_cruise = get_cruise_accel(sm['selfdriveState'].experimentalMode, v_target, v_ego, a_ego_filt,
                                     self.a_cruise, steer_angle_without_offset, self.CP, self.dt,
                                     accel_coast, self.allow_throttle, self.cruise_integ, personality,
                                     v_cruise)
    cruise_should_stop = v_ego < self.CP.vEgoStopping and self.a_cruise < 0.1

    # Take the lowest accel of {MPC's lead-driven command, cruise, [e2e]}. Being the
    # minimum means a lead-following command can never be overridden by cruise just
    # because the lead is far away, which is what the old cruise-obstacle rule did.
    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)]
    if sm['selfdriveState'].experimentalMode:
      candidates.append((output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e))

    output_a_target, plan_source, _ = min(candidates, key=lambda c: c[0])
    self.mpc.source = plan_source
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates)

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    # Never step the accel request UP: the automatic needs the request to build slowly or it
    # kicks down and then surges. Braking is deliberately not rate-limited here (a delayed
    # brake is a safety problem, and the car's own actuator lag already smooths it).
    j_rise = accel_rise_limit(v_ego, _pers)
    self.output_a_target_prev = min(output_a_target, self.output_a_target_prev + j_rise * self.dt)
    self.output_a_target = np.clip(self.output_a_target_prev, accel_clip[0], accel_clip[1])
    # the state must follow what was published: if a transient clip held the output down,
    # recovering from it must not release the accumulated step
    self.output_a_target_prev = float(self.output_a_target)
    self.prev_accel_clip = accel_clip

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)
