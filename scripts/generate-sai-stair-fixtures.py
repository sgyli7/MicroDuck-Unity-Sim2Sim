"""Regenerate C# golden fixtures from a specified Sai source checkout."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--sai-source',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.sai_source/'src'))
from sai_agent.control import observation_numpy,targets_stairs_numpy,HeadingHold
rng=np.random.default_rng(98);rows=[]
for i in range(96):
    q=rng.normal(0,.1,23);q[2]=.2192;q[3:7]/=np.linalg.norm(q[3:7])
    v=rng.normal(0,.2,22);command=np.array([[-.16,0,.012,.12,.16][i%5],.3 if i%2 else 0.])
    crouch=float(i%4)/3;previous=rng.uniform(-1,1,16).astype(np.float32);action=rng.normal(0,.8,16).astype(np.float32)
    t=[0,.8,1.6,2.4,3.2,6.4][i%6]+[-1e-7,0,1e-7][i%3]
    heights=np.zeros(24) if i%4==0 else rng.choice([0.,.02,.04,.06],24)
    rows.append(dict(q=q.tolist(),v=v.tolist(),command=command.tolist(),crouch=crouch,previous=previous.tolist(),action=action.tolist(),time=t,heights=heights.tolist(),stairs=True,
        observation=observation_numpy(q,v,command,crouch,previous,t*2*np.pi/3.2,heights).tolist(),
        target=targets_stairs_numpy(action,command,crouch,t/3.2,heights).tolist()))
heading=HeadingHold();heading_rows=[]
for i in range(64):
    command=np.zeros(2) if i%10==0 else np.array([.16,.3 if i%2 else -.3])
    target=rng.normal(0,1,16);yaw=float(rng.uniform(-3.5,3.5));rate=float(rng.normal())
    heading_rows.append(dict(command=command.tolist(),input_target=target.tolist(),yaw=yaw,yaw_rate=rate,target=heading.apply(target,command,yaw,rate).tolist()))
out=Path(__file__).resolve().parents[1]/'tests/sai_contract_console/python-stair-fixtures.json'
out.write_text(json.dumps(dict(reference='Sai frozen stair reference and heading controller',source_sha256=hashlib.sha256((a.sai_source/'src/sai_agent/control.py').read_bytes()).hexdigest(),cases=rows,heading_cases=heading_rows),separators=(',',':'))+'\n')
print(out)
