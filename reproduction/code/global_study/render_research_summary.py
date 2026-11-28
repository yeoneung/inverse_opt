"""Summarize every completed cohort and export standalone scientific figures."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'rendered')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    checked=json.loads((ROOT/'independent_result_checks.json').read_text())
    assert checked['passed'] and checked['compact_p8_complete']
    groups={
        'capacity_q16_p4':[r for r in checked['rows'] if r['label'].startswith('p4validation_q16_')],
        'capacity_q32_p4':[r for r in checked['rows'] if r['label'].startswith('p4validation_q32_')],
        'capacity_q12_p8':[r for r in checked['rows'] if r['label'].startswith('compact_capacity_extended/')],
        'battery_q8_p4':[r for r in checked['rows'] if r['label'].startswith('battery_grid_')],
    }
    summary={key:dict(cases=len(rows),positive=sum(r['detected'] for r in rows),
        minimum_positive_bound=min(r['validated_certificate'] for r in rows if r['detected']),
        median_total_seconds_including_final_validation=float(np.median([r['pipeline_and_repair_seconds'] for r in rows])),
        maximum_total_seconds_including_final_validation=max(r['pipeline_and_repair_seconds'] for r in rows))
        for key,rows in groups.items()}
    tight=json.loads((ROOT/'battery_tight_reference.json').read_text())['rows'][0]
    summary['primary_battery_reference']={key:tight[key] for key in ['mixed_risk_lower','mixed_risk_upper','proper_risk_lower','proper_risk_upper','actual_penalty_lower','actual_penalty_upper','squared_risk_ratio_interval','captured_fraction_interval']}
    summary['scope']={'global_upper_alone_cannot_rescue_original_fixed_anchors':18,
        'generic_q32_p8_pilot_still_inconclusive':True,
        'battery_data':'Synthetic tariffs and loads, ideal lossless battery; not field validation.',
        'numerics':'Floating-point global solver bounds plus independent QP/contact/separation validation, not interval arithmetic.',
        'manuscript_modified':False}
    (args.output/'validated_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    figures=args.output/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(10,3.1),layout='constrained')
    labels=['q=16,p=4','q=32,p=4','q=12,p=8']
    keys=list(groups)[:3]
    for index,(label,key) in enumerate(zip(labels,keys)):
        s=summary[key]
        axes[0].bar(index,s['positive']/s['cases'],color='#2166ac')
        axes[0].text(index,s['positive']/s['cases']+.025,f"{s['positive']}/{s['cases']}",ha='center')
    axes[0].set_xticks(range(3),labels,rotation=18)
    axes[0].set(ylim=(0,1.18),ylabel='Positive certificate fraction',title='Global exchange, all conditions')
    axes[1].bar(range(3),[summary[k]['median_total_seconds_including_final_validation'] for k in keys],color=['#2166ac','#2166ac','#b35806'])
    axes[1].set_xticks(range(3),labels,rotation=18)
    axes[1].set(yscale='log',ylabel='Complete median wall time (s)',title='Cost includes validation')
    axes[1].set_yticks([1,10,100],['1','10','100'])
    axes[1].yaxis.set_minor_formatter(__import__('matplotlib').ticker.NullFormatter())
    grid=json.loads((ROOT/'battery_grid_validation/runs.json').read_text())['rows']
    check={r['label'].rsplit('/',1)[-1]:r for r in groups['battery_q8_p4']}
    values=np.array([[check[next(r['label'] for r in grid if r['target_power_kw']==power and r['reserve_amplitude_kwh']==amp)]['validated_certificate'] for amp in [4,8,12]] for power in [1,2,4]])
    axes[2].imshow(values,origin='lower',aspect='auto',cmap='Blues',vmin=0)
    for i in range(3):
        for j in range(3):axes[2].text(j,i,f'{values[i,j]:.2g}',ha='center',va='center',color='white' if values[i,j]>.0007 else 'black',fontsize=8)
    axes[2].set_xticks(range(3),['4','8','12']);axes[2].set_yticks(range(3),['1','2','4'])
    axes[2].set(xlabel='Reserve band half-width (kWh)',ylabel='Power limit (kW)',title='Battery: all 9 lower bounds')
    fig.savefig(figures/'global_and_battery_results.pdf');fig.savefig(figures/'global_and_battery_results.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.1),layout='constrained')
    hours=np.arange(9)*3
    for key,label,color,style in [('mixed_physical_schedule','Mixed center','#2166ac','-'),('proper_physical_schedule','One compatible objective','#b35806','--')]:
        schedule=tight[key]
        axes[0].stairs(schedule['power_kw'],hours,label=label,color=color,linestyle=style,linewidth=1.7)
        axes[1].plot(hours,np.r_[16,schedule['soc_kwh']],label=label,color=color,linestyle=style,linewidth=1.7)
    axes[0].set(xlabel='Hour',ylabel='Battery power (kW)',title='24-hour schedule')
    axes[1].axhline(4,color='grey',ls=':',lw=1);axes[1].axhline(32,color='grey',ls=':',lw=1)
    axes[1].set(xlabel='Hour',ylabel='Stored energy (kWh)',ylim=(2,34),title='Coupled state constraints')
    for y,lo,hi,color in [(0,tight['mixed_risk_lower'],tight['mixed_risk_upper'],'#2166ac'),(1,tight['proper_risk_lower'],tight['proper_risk_upper'],'#b35806')]:
        mid=(lo+hi)/2
        axes[2].errorbar(mid,y,xerr=np.array([[mid-lo],[hi-mid]]),fmt='o',capsize=7,color=color)
    axes[2].set_yticks([0,1],['Mixed','Single objective']);axes[2].set(xlabel='Worst squared error in Q metric',ylim=(-.7,1.7),title='Independent global risk intervals')
    axes[2].text(.05,.04,'Risk ratio: 1.01345\nCertified fraction of excess: 43.18%',transform=axes[2].transAxes,fontsize=8)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.1),ncol=2,frameon=False,fontsize=8)
    fig.savefig(figures/'battery_application.pdf',bbox_inches='tight');fig.savefig(figures/'battery_application.png',dpi=220,bbox_inches='tight');plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
