"""Fetch the exact Unity binding and locate the matching isolated native library.

Run with Python containing mujoco==3.12.0. Does not install an editor, alter the
Unity package pin or copy third-party binaries into the repository.
"""
from pathlib import Path
import hashlib
import json
import urllib.request
import sys
import mujoco

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'artifacts/sai-native-test';out.mkdir(parents=True,exist_ok=True)
commit='13827e9ee56f097f57acf69ae52b078f9839682d'
url=f'https://raw.githubusercontent.com/google-deepmind/mujoco/{commit}/unity/Runtime/Bindings/MjBindings.cs'
digest='4ce5930762167feba1ca6bef0c426af3011f46cafc87d5e7dae33c1359e2060d'
binding=out/'MjBindings.cs'
if not binding.exists():
    with urllib.request.urlopen(url,timeout=30) as response:binding.write_bytes(response.read())
if hashlib.sha256(binding.read_bytes()).hexdigest()!=digest:raise ValueError('Unity binding hash mismatch')
if mujoco.__version__!='3.12.0':raise ValueError('Native acceptance requires mujoco==3.12.0, matching the Unity binding')
directory=Path(mujoco.__file__).parent
candidates=[p for p in directory.iterdir() if p.is_file() and (p.name.startswith('libmujoco.') or p.name=='mujoco.dll')]
if len(candidates)!=1:raise ValueError(f'Expected one native library; found {candidates}')
native=candidates[0]
bundled=False
key={'win32':'mujocoWindowsX64','darwin':'mujocoMacOSUniversal2'}.get(sys.platform)
if key:
    lock=json.loads((ROOT/'upstream.lock.json').read_text())['nativeBinaries'][key]
    native=ROOT/lock['projectPath']
    if hashlib.sha256(native.read_bytes()).hexdigest()!=lock['sha256']:
        raise ValueError('The project MuJoCo binary does not match its supply-chain lock')
    bundled=True
manifest={'binding_source':url,'binding_sha256':digest,'native_version':mujoco.__version__,
          'native_library':str(native),'native_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),
          'project_bundled_binary':bundled}
(out/'runtime.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out/'runtime.json')
