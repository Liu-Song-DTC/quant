"""Plot account recovery on identical sealed data; never write strategy inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/percent_repair_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,default=ROOT/'.review_runs/arm_pct_ms_after_day_gates_20261108')
    parser.add_argument('--label',default='T Selection order')
    parser.add_argument('--status',default='Audited research replay; main merge pending')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/percent_repair_equity_drawdown_20261108')
    args=parser.parse_args()
    specs=[('A Baseline',ROOT/'.review_runs/arm_fundscore_repro_20261106','#525252','--'),('F Units fixed',ROOT/'.review_runs/arm_pct_strict_units_20261107','#c24136',':'),(args.label,args.candidate,'#047857','-')]
    data=[];metadata={'status':args.status,'curves':[]};calendar=None;seals=[]
    for label,root,color,style in specs:
        root=root.resolve();path=root/'corrected/equity_curve.csv';frame=pd.read_csv(path,parse_dates=['date'])
        assert frame.date.is_monotonic_increasing and not frame.date.duplicated().any()
        nav=frame.nav.to_numpy(dtype=float);assert np.isfinite(nav).all() and (nav>0).all() and nav[0]==250000
        if calendar is None:calendar=frame.date
        else:assert frame.date.equals(calendar),'Account calendars differ'
        assert json.loads((root/'corrected_input_stamp.json').read_text())['unchanged'] is True
        seals.append(json.loads((root/'manifest.json').read_text())['data_sha256'])
        ret=(nav/nav[0]-1)*100;dd=(nav/np.maximum.accumulate(nav)-1)*100
        data.append((label,frame.date,ret,dd,color,style))
        metadata['curves'].append({'label':label,'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(frame),'total_return_percent':float(ret[-1]),'max_drawdown_percent':float(dd.min())})
    assert len(set(seals))==1,'Different sealed data inputs'
    metadata['data_sha256']=seals[0]
    with plt.rc_context({'font.size':9,'axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True}):
        fig,axes=plt.subplots(2,1,figsize=(10.6,6.8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        fig.subplots_adjust(left=.08,right=.985,top=.91,bottom=.095,hspace=.15)
        for label,dates,ret,dd,color,style in data:
            axes[0].plot(dates,ret,color=color,linestyle=style,linewidth=1.8,label=f'{label}: {ret[-1]:.2f}% | DD {abs(dd.min()):.2f}%')
            axes[1].plot(dates,dd,color=color,linestyle=style,linewidth=1.2)
            idx=int(np.argmin(dd));axes[1].scatter(dates.iloc[idx],dd[idx],color=color,s=19,zorder=4)
            right=dates.iloc[idx]>dates.iloc[-1]-pd.Timedelta(days=180)
            axes[1].annotate(f'{label.split()[0]} {dd[idx]:.2f}%',(dates.iloc[idx],dd[idx]),xytext=(-6 if right else 5,-12),textcoords='offset points',ha='right' if right else 'left',fontsize=8,color=color)
        for ax in axes:
            ax.axvspan(pd.Timestamp('2023-01-01'),pd.Timestamp('2024-12-31'),color='#64748b',alpha=.06)
            ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        axes[0].set_ylabel('Cumulative return (%)');axes[1].set_ylabel('Drawdown (%)')
        axes[0].legend(loc='upper left',frameon=True,framealpha=.95,fontsize=8.5)
        axes[1].axhline(0,color='#94a3b8',linewidth=.7);axes[1].set_ylim(min(float(x[3].min()) for x in data)*1.23,1)
        axes[1].xaxis.set_major_locator(mdates.YearLocator());axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[1].set_xlim(calendar.iloc[0],calendar.iloc[-1])
        fig.suptitle('Same sealed data: return recovery and drawdown',fontsize=13,fontweight='bold',y=.965)
        footer=f'{calendar.iloc[0]:%Y-%m-%d} to {calendar.iloc[-1]:%Y-%m-%d} | Gray band: 2023-2024 | {args.status}'
        fig.text(.08,.018,footer,fontsize=7.5,color='#475569')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        for suffix in ['.png','.svg']:fig.savefig(args.output.with_suffix(suffix),dpi=180)
        plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(args.output.with_suffix('.png'))

if __name__=='__main__':
    main()
