"""Unfiltered synthetic battery-scheduling scenario grid.

This varies target power availability and the declared reserve-trajectory
prior. It does not claim measured-market validation or calibrated devices.
Every scenario, including failures or inconclusive results, is retained.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np

from application_battery import battery_instance,checks,physical_schedule,serializable
from exchange_pilot import run_exchange
from src.multidimensional_certification import Projector

ROOT=Path(__file__).resolve().parent


def run_grid(output,*,time_limit=15.,iterations=10):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(application='synthetic_battery_load_shifting',horizon_slots=8,
        slot_hours=3.,latent_dimension=4,reserve_basis='cosine_modes_1_to_4',
        target_power_kw=[1.,2.,4.],reserve_amplitude_kwh=[4.,8.,12.],
        source_power_kw=.5,source_tariff_usd_per_kwh=1.2,
        initial_queries=16,global_oracle_time_limit_seconds=time_limit,
        maximum_exchange_iterations=iterations,positive_threshold=1e-6,
        repetitions=1,scope='Complete 3x3 synthetic scenario grid; all outcomes retained.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[]
    started=time.perf_counter()
    for power in protocol['target_power_kw']:
        for amplitude in protocol['reserve_amplitude_kwh']:
            label=f'battery_grid_P{power:g}_R{amplitude:g}'
            row=dict(label=label,target_power_kw=power,reserve_amplitude_kwh=amplitude)
            begin=time.perf_counter()
            try:
                instance=battery_instance(4,horizon=8,reserve_basis='cosine',
                    target_power_kw=power,reserve_amplitude_kwh=amplitude)
                verified=checks(instance)
                row.update(instance=serializable(instance),physical_checks=verified)
                # The controlled relative prefix puts all solver logs in output.
                log_label='battery_grid_validation/'+label
                exchange=run_exchange(instance['A'],instance['b'],instance['c'],instance['G'],
                    label=log_label,time_limit=time_limit,iterations=iterations,log_directory=output)
                row.update(passed=True,exchange=exchange)
                final=exchange['history'][-1]
                mixed=physical_schedule(instance,np.asarray(final['anchor']))
                projector=Projector(instance['A'],instance['b'],8)
                extremes=[]
                for sign in [-1.,1.]:
                    parameter=np.full(4,sign)
                    action,_=projector(instance['c']+instance['G']@parameter)
                    schedule=physical_schedule(instance,action)
                    extremes.append(dict(parameter=parameter.tolist(),
                        whitened_action=action.tolist(),**serializable(schedule)))
                mixed_violation=float(np.max(instance['physical_A']@mixed['power_kw']-instance['physical_b'],initial=0.))
                assert mixed_violation<2e-7
                row['mixed_physical_schedule']=serializable(mixed)
                row['two_declared_corner_schedules']=extremes
                row['mixed_schedule_maximum_constraint_violation']=mixed_violation
            except Exception as exc:
                row.update(passed=False,error_type=type(exc).__name__,error=str(exc))
            row['complete_scenario_seconds']=time.perf_counter()-begin
            rows.append(row)
            (output/(label+'.json')).write_text(json.dumps(serializable(row),indent=2),encoding='utf-8')
            with (output/'cases.jsonl').open('a',encoding='utf-8') as file:
                file.write(json.dumps(serializable(row))+'\n')
            print(label,'detected',row.get('exchange',{}).get('detected'),
                'certificate',row.get('exchange',{}).get('certificate'),
                'error',row.get('error'),flush=True)
    summary=dict(passed=all(r['passed'] for r in rows),scenarios=len(rows),
        successful_computations=sum(r['passed'] for r in rows),
        positive_certificates=sum(r.get('exchange',{}).get('detected',False) for r in rows),
        total_seconds=time.perf_counter()-started,
        outcomes=[dict(label=r['label'],power_kw=r['target_power_kw'],amplitude_kwh=r['reserve_amplitude_kwh'],
            passed=r['passed'],certificate=r.get('exchange',{}).get('certificate'),
            iterations=len(r.get('exchange',{}).get('history',[])),
            oracle_statuses=[h['oracle']['status'] for h in r.get('exchange',{}).get('history',[])],
            total_seconds=r.get('exchange',{}).get('total_seconds'),error=r.get('error')) for r in rows])
    (output/'runs.json').write_text(json.dumps(serializable(dict(protocol=protocol,rows=rows)),indent=2),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'battery_grid_validation')
    parser.add_argument('--time-limit',type=float,default=15.)
    parser.add_argument('--iterations',type=int,default=10)
    args=parser.parse_args()
    print(json.dumps(run_grid(args.output,time_limit=args.time_limit,iterations=args.iterations),indent=2))


if __name__=='__main__':
    main()
