import unittest

from openpilot.cereal import log, messaging
from opendbc.car.structs import car
from openpilot.selfdrive.controls.radard import (
  RADAR_TO_CAMERA,
  RadarD,
  VisionLeadFilter,
  _LEAD_ACCEL_TAU,
  get_RadarState_from_vision,
  get_lead,
)


def _make_lead_msg(x: float, v: float, a: float = 0.0, y: float = 0.0, prob: float = 0.99):
  md = messaging.new_message("modelV2")
  md.modelV2.leadsV3 = [log.ModelDataV2.LeadDataV3.new_message()]
  ld = md.modelV2.leadsV3[0]
  ld.x = [x]
  ld.y = [y]
  ld.v = [v]
  ld.a = [a]
  ld.prob = prob
  return md.as_reader().modelV2.leadsV3[0]


class TestRadardVision(unittest.TestCase):
  def test_schema_compatibility_and_four_arg_signature(self):
    lead_msg = _make_lead_msg(x=31.52, v=20.0, a=0.2, y=-0.3)
    vf = VisionLeadFilter()
    out = get_RadarState_from_vision(lead_msg, 20.0, 20.0, 0.95, vision_filter=vf)
    msg = log.RadarState.LeadData.new_message(**out)
    self.assertTrue(msg.present)
    self.assertFalse(msg.radar)
    self.assertEqual(msg.radarTrackId, -1)
    self.assertAlmostEqual(msg.dRel, 30.0, places=4)
    self.assertAlmostEqual(msg.yRel, 0.3, places=4)

  def test_symmetric_vision_filter_and_jump_reset(self):
    f_pos = VisionLeadFilter(tau=0.45, dt=0.05)
    f_neg = VisionLeadFilter(tau=0.45, dt=0.05)
    self.assertAlmostEqual(f_pos.update(40.0, 0.0), 0.0)
    self.assertAlmostEqual(f_neg.update(40.0, 0.0), 0.0)
    for _ in range(10):
      up = f_pos.update(40.0, 2.0)
      dn = f_neg.update(40.0, -2.0)
      self.assertAlmostEqual(up, -dn, places=9)
    # Jump > 8 m resets immediately to raw vrel
    self.assertAlmostEqual(f_pos.update(52.0, -3.5), -3.5, places=9)

  def test_standstill_launch_no_w_gap_and_fast_tau(self):
    vf = VisionLeadFilter()
    lead_msg = _make_lead_msg(x=4.5 + RADAR_TO_CAMERA, v=1.2, a=0.8)
    out = get_RadarState_from_vision(lead_msg, v_ego=0.0, model_v_ego=0.0, lead_prob=0.98, a_ego=0.0, vision_filter=vf)
    self.assertAlmostEqual(out["vRel"], 1.2, places=6)
    self.assertAlmostEqual(out["vLead"], 1.2, places=6)
    self.assertAlmostEqual(out["aLeadK"], 0.8, places=6)
    self.assertAlmostEqual(out["aLeadTau"], 0.3, places=6)

  def test_highway_symmetric_a_ego_debias_accel_and_mild_brake(self):
    # 1. Highway acceleration surge (+1.0 m/s^2 at 31 m/s): removes +1.10*a_ego and +0.35*a_ego without w_hwy opening suppression
    vf_acc = VisionLeadFilter()
    lead_acc = _make_lead_msg(x=50.0 + RADAR_TO_CAMERA, v=34.10, a=0.75)
    out_acc = get_RadarState_from_vision(lead_acc, v_ego=31.0, model_v_ego=31.0, lead_prob=0.95, a_ego=1.0, vision_filter=vf_acc)
    self.assertAlmostEqual(out_acc["vRel"], 2.0, places=5)
    self.assertAlmostEqual(out_acc["aLeadK"], 0.40, places=5)
    self.assertAlmostEqual(out_acc["aLeadTau"], 0.3, places=5)

    # 2. Highway mild braking (-0.8 m/s^2 at 27 m/s behind non-braking lead a=0.0): removes nose-dive pitch leak symmetrically
    vf_brk = VisionLeadFilter()
    lead_steady = _make_lead_msg(x=35.0 + RADAR_TO_CAMERA, v=26.12, a=0.0)
    out_brk = get_RadarState_from_vision(lead_steady, v_ego=27.0, model_v_ego=27.0, lead_prob=0.98, a_ego=-0.8, vision_filter=vf_brk)
    # vrel_raw = (26.12 - 27.0) - 1.10 * (-0.8) = 0.0
    self.assertAlmostEqual(out_brk["vRel"], 0.0, places=5)
    self.assertAlmostEqual(out_brk["aLeadK"], 0.28, places=5)

    # 3. Exact zero-crossing symmetry at highway speed (no sign-dependent vRel gain kink around vRel = 0)
    vf_p, vf_n = VisionLeadFilter(), VisionLeadFilter()
    out_p = get_RadarState_from_vision(_make_lead_msg(42.0 + RADAR_TO_CAMERA, 29.5, 0.0), 28.0, 28.0, 0.99, a_ego=0.0, vision_filter=vf_p)
    out_n = get_RadarState_from_vision(_make_lead_msg(42.0 + RADAR_TO_CAMERA, 26.5, 0.0), 28.0, 28.0, 0.99, a_ego=0.0, vision_filter=vf_n)
    self.assertAlmostEqual(out_p["vRel"], -out_n["vRel"], places=6)

  def test_closing_no_distance_attenuation_and_hard_stop_gate(self):
    vf = VisionLeadFilter()
    # Far gradual approach at 70 m, 30 m/s: closing speed is NOT attenuated by w_dist
    lead_far = _make_lead_msg(x=70.0 + RADAR_TO_CAMERA, v=27.0, a=-0.10)
    out_far = get_RadarState_from_vision(lead_far, v_ego=30.0, model_v_ego=30.0, lead_prob=0.95, a_ego=0.0, vision_filter=vf)
    self.assertAlmostEqual(out_far["vRel"], -3.0, places=5)
    self.assertAlmostEqual(out_far["aLeadTau"], _LEAD_ACCEL_TAU, places=5)

    # Hard lead braking (a_lead_raw = -1.2 <= -0.80) while ego also brakes (a_ego = -2.0): neg_lim = 0.0 -> zero positive boost
    vf_brk = VisionLeadFilter()
    lead_brk = _make_lead_msg(x=25.0 + RADAR_TO_CAMERA, v=15.0, a=-1.2)
    out_brk = get_RadarState_from_vision(lead_brk, v_ego=20.0, model_v_ego=20.0, lead_prob=0.99, a_ego=-2.0, vision_filter=vf_brk)
    self.assertAlmostEqual(out_brk["vRel"], -5.0, places=5)
    self.assertAlmostEqual(out_brk["aLeadK"], -1.2, places=5)
    self.assertAlmostEqual(out_brk["aLeadTau"], 0.3, places=5)

  def test_radard_dropout_resets_vision_filter(self):
    rd = RadarD(0.0)
    sm = messaging.SubMaster(["modelV2", "carState", "radarTracks"])
    rr = car.RadarData.new_message()

    def _feed(frame: int, prob: float, v_lead: float):
      sm.seen["modelV2"] = True
      sm.recv_frame["carState"] = frame
      cs = messaging.new_message("carState")
      cs.carState.vEgo = 20.0
      cs.carState.aEgo = 0.0
      sm.data["carState"] = cs.as_reader().carState
      md = messaging.new_message("modelV2")
      md.modelV2.velocity.x = [20.0]
      md.modelV2.leadsV3 = [
        log.ModelDataV2.LeadDataV3.new_message(x=[40.0 + RADAR_TO_CAMERA], y=[0.0], v=[v_lead], a=[0.0], prob=prob),
        log.ModelDataV2.LeadDataV3.new_message(x=[80.0 + RADAR_TO_CAMERA], y=[0.0], v=[20.0], a=[0.0], prob=0.0),
      ]
      sm.data["modelV2"] = md.as_reader().modelV2
      rd.update(sm, rr.as_reader())

    _feed(1, 0.95, 22.0)
    self.assertTrue(rd.radar_state.leadOne.present)
    self.assertIsNotNone(rd.vision_filters[0].vrel_filt)

    # Drop probability to 0 until lead_prob_filter decays <= 0.5 -> vision_filter resets
    for f in range(2, 15):
      _feed(f, 0.0, 22.0)
    self.assertFalse(rd.radar_state.leadOne.present)
    self.assertIsNone(rd.vision_filters[0].vrel_filt)

    # Re-acquire closing lead -> starts fresh without stale +2 m/s state
    _feed(16, 0.95, 17.0)
    self.assertTrue(rd.radar_state.leadOne.present)
    self.assertAlmostEqual(rd.radar_state.leadOne.vRel, -3.0, places=4)


if __name__ == "__main__":
  unittest.main()
