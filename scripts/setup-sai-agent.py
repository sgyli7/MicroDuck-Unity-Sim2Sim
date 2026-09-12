"""Fetch a pinned Sai package and stage its original model/actor for Unity."""
import argparse
import io
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PIN='8ea8348d95193c2bad91874816a76e5527238e30'
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,help='Optional local Sai_Agent_001 checkout for development')
a=p.parse_args()
staging=ROOT/'TuanjieProject/Assets/StreamingAssets/SaiAgent001'
policy=ROOT/'TuanjieProject/Assets/SaiAgent001/Generated/flat-v1.onnx'
prefixes=('models/full/','licenses/')
names=('policies/flat-v1.onnx','policies/flat-v1.json','THIRD_PARTY_NOTICES.md','LICENSE')
def put(relative,data):
    if relative=='policies/flat-v1.onnx':destination=policy
    else:destination=staging/relative
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
if a.source:
    for f in a.source.rglob('*'):
        if not f.is_file():continue
        relative=f.relative_to(a.source).as_posix()
        if relative.startswith(prefixes) or relative in names:put(relative,f.read_bytes())
else:
    url=f'https://codeload.github.com/sgyli7/Sai_Agent_001/zip/{PIN}'
    with urllib.request.urlopen(url,timeout=120) as response:archive=response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for member in z.infolist():
            if member.is_dir():continue
            relative=member.filename.partition('/')[2]
            if '..' in Path(relative).parts or Path(relative).is_absolute():raise ValueError('Invalid archive path')
            if relative.startswith(prefixes) or relative in names:put(relative,z.read(member))
# No reliance on a previous local test having produced the flat full model.
import xml.etree.ElementTree as ET
source=staging/'models/full/visual.xml';tree=ET.parse(source);world=tree.getroot().find('worldbody')
for body in list(world.findall('body')):
    if body.get('name')=='item':world.remove(body)
for geom in list(world.findall('geom')):
    if geom.get('name','').startswith('course_'):world.remove(geom)
k=tree.getroot().find('keyframe')
if k is not None:tree.getroot().remove(k)
tree.write(staging/'models/full/locomotion-articulated.xml',encoding='unicode')
metadata=json.loads((staging/'policies/flat-v1.json').read_text())
import hashlib
assert hashlib.sha256(policy.read_bytes()).hexdigest()==metadata['onnx_sha256']
(staging/'package-pin.json').write_text(json.dumps({'commit':PIN,'local_override':a.source is not None,'actor_sha256':metadata['onnx_sha256']},indent=2)+'\n')
print('Sai model and policy staged. Open TuanjieProject, choose SaiAgent001/Build Demo, then Play.')
