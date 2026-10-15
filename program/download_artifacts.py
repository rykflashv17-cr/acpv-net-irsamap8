"""Download pinned artifacts, verify SHA256, and safely restore archive files."""
import argparse,hashlib,json,os,shutil,tarfile,zipfile
from pathlib import Path
import requests
from huggingface_hub import hf_hub_download
from project_paths import ROOT,DATA,REPOSITORY

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def restore_tar(path,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(path,'r:*') as t:
        for item in t:
            target=(destination/item.name).resolve()
            if not target.is_relative_to(destination) or not item.isfile():raise ValueError('Archive must contain regular files below the dataset root')
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.extracting')
            with t.extractfile(item) as source,temporary.open('wb') as out:shutil.copyfileobj(source,out,8*1024*1024)
            if temporary.stat().st_size!=item.size:raise ValueError('Extracted size mismatch')
            temporary.replace(target)

def official_zip(url,target,expected,suffix):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and sha256(target)==expected:return
    archive=ROOT/'tmp'/('official-'+target.parent.name+'.zip')
    with requests.get(url,stream=True,timeout=(30,120)) as response:
        response.raise_for_status()
        with archive.open('wb') as out:
            for b in response.iter_content(8*1024*1024):out.write(b)
    try:
        with zipfile.ZipFile(archive) as z:
            names=[n for n in z.namelist() if n.endswith(suffix)]
            if len(names)!=1:raise ValueError('Expected exactly one official checkpoint member')
            temp=target.with_name(target.name+'.download')
            with z.open(names[0]) as source,temp.open('wb') as out:shutil.copyfileobj(source,out,8*1024*1024)
        if sha256(temp)!=expected:raise ValueError('Official checkpoint checksum mismatch')
        temp.replace(target)
    finally:archive.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['model','resume','vae','official','dataset-core','dataset-latents'])
    args=p.parse_args();manifest=json.loads((REPOSITORY/'manifests/artifacts.json').read_text())
    if args.kind in ['vae','official']:
        spec=manifest['dependencies'][args.kind]
        official_zip(spec['url'],ROOT/spec['target'],spec['sha256'],spec['zip_member_suffix']);return
    if args.kind in ['model','resume']:
        spec=manifest['model'];key='model.pth' if args.kind=='model' else 'training_checkpoint.pth'
        path=Path(hf_hub_download(spec['repo_id'],key,revision=spec['revision'],local_dir=ROOT/'weights'))
        if sha256(path)!=spec['files'][key]['sha256']:raise ValueError('Model checksum mismatch')
        print(path);return
    spec=manifest['dataset']
    # Download the complete, revision-pinned list; each file is deleted after verified extraction.
    index=Path(hf_hub_download(spec['repo_id'],'dataset_manifest.json',repo_type='dataset',revision=spec['revision'],local_dir=ROOT/'tmp/dataset_download'))
    if spec.get('manifest_sha256') and sha256(index)!=spec['manifest_sha256']:
        raise ValueError('Dataset manifest checksum mismatch')
    details=json.loads(index.read_text());part='core' if args.kind=='dataset-core' else 'latents'
    for entry in details['shards']:
        if entry['component']!=part:continue
        archive=Path(hf_hub_download(spec['repo_id'],entry['path'],repo_type='dataset',revision=spec['revision'],local_dir=ROOT/'tmp/dataset_download'))
        if sha256(archive)!=entry['sha256']:raise ValueError('Dataset archive checksum mismatch')
        restore_tar(archive,DATA);archive.unlink()
    print('Restored',part,'to',DATA)
if __name__=='__main__':main()
