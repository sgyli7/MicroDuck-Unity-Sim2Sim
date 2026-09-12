"""Fetch a pinned Sai package and stage its original model/actor for Unity."""
import argparse
import io
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
import copy
import hashlib

ROOT=Path(__file__).resolve().parents[1]
PIN='6eb18b2abf97ce6de11a18daba93729848736adf'
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,help='Optional local Sai_Agent_001 checkout for development')
a=p.parse_args()
staging=ROOT/'TuanjieProject/Assets/StreamingAssets/SaiAgent001'
policy=ROOT/'TuanjieProject/Assets/SaiAgent001/Generated/flat-v1.onnx'
prefixes=('models/full/','licenses/')
names=('policies/flat-v1.onnx','policies/flat-v1.json','policies/stairs-dev40.onnx','policies/stairs-dev40.json','THIRD_PARTY_NOTICES.md','LICENSE')
def put(relative,data):
    if relative.startswith('policies/') and relative.endswith('.onnx'):destination=policy.parent/Path(relative).name
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
# Reserve one ray group for terrain. This changes sensing/render groups only,
# never collision masks, inertia, mass or joint frames.
for geom in tree.findall('.//geom'):geom.set('group','0')
world.find("geom[@name='ground']").set('group','5')
tree.write(staging/'models/full/locomotion-articulated.xml',encoding='unicode')
courses=[]
for riser in [.02,.04]:
    for descending in [False,True]:
        course=copy.deepcopy(tree);cw=course.getroot().find('worldbody')
        cw.find("geom[@name='ground']").set('pos','0 0 -.002')
        boundaries=[-1.,.45,.63,.81,.99,3.]
        for i,(left,right) in enumerate(zip(boundaries[:-1],boundaries[1:])):
            top=riser*(4-i if descending else i)
            # Same five segment geometry as Sai's accepted four-riser course.
            body=ET.SubElement(cw,'body',name=f'stair_segment_{i}',mocap='true',pos=f'{(left+right)/2} 0 {top-.5}')
            ET.SubElement(body,'geom',name=f'stair_surface_{i}',type='box',size=f'{(right-left)/2} .5 .5',rgba='.42 .46 .48 1',contype='2',conaffinity='5',group='5')
        if descending:
            base=cw.find("body[@name='chassis']");xyz=list(map(float,base.get('pos').split()));xyz[2]+=4*riser;base.set('pos',' '.join(map(str,xyz)))
        name=f'stairs-{round(riser*1000)}-{"down" if descending else "up"}.xml'
        course.write(staging/'models/full'/name,encoding='unicode')
        courses.append(dict(file=name,riser=riser,descending=descending,tread=.18,count=4))
hashes={}
for name in ['flat-v1','stairs-dev40']:
    metadata=json.loads((staging/'policies'/f'{name}.json').read_text())
    digest=hashlib.sha256((policy.parent/f'{name}.onnx').read_bytes()).hexdigest()
    expected=metadata.get('onnx_sha256',metadata.get('sha256'))
    if digest!=expected:raise ValueError(f'{name} actor integrity check failed')
    hashes[name]=digest
(staging/'package-pin.json').write_text(json.dumps({'commit':PIN,'local_override':a.source is not None,'actor_sha256':hashes['flat-v1'],'actors':hashes,'courses':courses},indent=2)+'\n')
print('Sai model and policy staged. Open TuanjieProject, choose SaiAgent001/Build Demo, then Play.')
