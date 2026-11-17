"""Independent invariants for numerical changes, not duplicate formula tests."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import analyse as a
from run_cases import supply,case_spec,enthalpy
import climate
import cfd_gen

class PhysicalInvariants(unittest.TestCase):
    def test_evaporation_conserves_enthalpy_and_obeys_limits(self):
        for T,RH,p in [(42,10,94200),(33,65,101325),(38,99,100800)]:
            for flow in [1.25,2.5,3.75]:
                Ts,Y,info=supply(T,RH,p,flow=flow)
                self.assertLessEqual(Ts,T)
                self.assertGreaterEqual(Ts,info['Twb']-1e-8)
                self.assertLessEqual(Y,cfd_gen.Y_h2o(Ts,100,p/100)+1e-9)
                self.assertLessEqual(info['evaporated_Lph'],100+1e-8)
                self.assertAlmostEqual(info['h_ambient'],info['h_supply'],places=9)

    def test_polygon_quadrature_preserves_area(self):
        for name in ['bayfront','medina']:
            for h in [1.5,4.]:
                g=a.geometry(name,h)
                self.assertAlmostEqual(float(g['area'].sum()),g['poly'].area,places=6)
                self.assertTrue(np.all(g['area']>0))

    def test_off_is_exactly_zero_and_overcooling_is_not_rewarded(self):
        for name in ['bayfront','medina']:
            g=a.geometry(name,4.)
            zero={n:np.zeros(len(g['xy'])) for n in ['dT','dw','dv2']}
            result,fields=a.evaluate(name,g,zero)
            self.assertAlmostEqual(result['normalised_utility_person_hours'],0.,places=9)
            np.testing.assert_allclose(fields['AT_off'],fields['AT_on'])
            bad=dict(zero,dT=np.full(len(g['xy']),80.))
            result,_=a.evaluate(name,g,bad)
            self.assertLess(result['normalised_utility_person_hours'],0.)

    def test_plant_capacity_includes_latent_load(self):
        s,m=case_spec('bf','D',True)
        k={'meta':{'metadata':dict(m,spec=s)}}
        r=a.resource(k,'bayfront',94)
        self.assertFalse(r['capacity_feasible'])
        self.assertGreater(r['thermal_kW'],490)
        self.assertGreater(r['power_kW'],94*.35)
        maxn=int(490//r['thermal_per_unit_kW'])
        self.assertTrue(a.resource(k,'bayfront',maxn)['capacity_feasible'])
        self.assertFalse(a.resource(k,'bayfront',maxn+1)['capacity_feasible'])

    def test_density_clamp(self):
        self.assertEqual(climate.eps_fn(.65,2.),climate.eps_fn(.16,2.))
        self.assertEqual(climate.eps_fn(.001,1.2),climate.eps_fn(.02,1.2))

    def test_scenario_layouts_have_no_duplicate_devices(self):
        positions=a.geometry('bayfront',4.)['positions']
        for n in [1,4,8,14]:
            p=a.select_positions(positions,n)
            self.assertEqual(len(np.unique(p,axis=0)),n)
            np.testing.assert_array_equal(p[:1],a.select_positions(positions,1))

if __name__=='__main__':unittest.main()
