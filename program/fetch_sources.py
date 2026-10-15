#!/usr/bin/env python3
"""Bounded official downloads, checksum verification and CRC-checked extraction."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import zipfile

from project_paths import ROOT, SOURCE, DATA, RAW
SOURCE = ROOT / 'sources'
EXPECTED = 11347439496
MD5 = '9245969e02b8e26d96c80b1eab549053'
SHA256 = 'c855ccf5f9f2388c07ac7143e00da7bb03528e51ef6b0ef98980a9496f6d1f08'
URL = 'https://china.scidb.cn/download?fileId=248f1b7980a74e17be054176e8a51faf'


def guard(pending):
    subprocess.run(['python', str(ROOT/'storage_guard.py'), '--pending-bytes', str(pending)], check=True, stdout=subprocess.DEVNULL)


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def hash_file(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            md5.update(b); sha.update(b)
    return md5.hexdigest(), sha.hexdigest()


def download_dataset():
    final = SOURCE/'IRSAMap_V2.zip'
    if final.exists():
        assert final.stat().st_size == EXPECTED
        assert hash_file(final) == (MD5, SHA256)
        return final
    guard(EXPECTED + 256*1024*1024)
    part = SOURCE/'IRSAMap_V2.zip.part'
    progress = SOURCE/'download_ranges.json'
    state = json.loads(progress.read_text()) if progress.exists() and part.exists() else {'done': []}
    done = set(state['done']); lock = threading.Lock()
    with part.open('ab'): pass
    with part.open('r+b') as f: f.truncate(EXPECTED)
    size = 32*1024*1024
    chunks = [(i, lo, min(EXPECTED-1,lo+size-1)) for i,lo in enumerate(range(0,EXPECTED,size))]
    began=time.monotonic()

    def one(chunk):
        i, lo, hi = chunk
        if i in done: return
        tmp = ROOT/'tmp'/f'download_{i:04d}.part'
        headers = tmp.with_suffix('.headers')
        subprocess.run(['curl','-fLsS','--retry','5','--retry-delay','2','--connect-timeout','20','--max-time','600',
                        '--max-filesize',str(hi-lo+1),'-r',f'{lo}-{hi}','-D',str(headers),'-o',str(tmp),URL], check=True)
        assert tmp.stat().st_size == hi-lo+1, (i,tmp.stat().st_size)
        assert f'content-range: bytes {lo}-{hi}/{EXPECTED}' in headers.read_text().lower(), headers.read_text()
        with tmp.open('rb') as src, part.open('r+b') as dst:
            dst.seek(lo)
            for b in iter(lambda:src.read(4*1024*1024), b''): dst.write(b)
        tmp.unlink(); headers.unlink()
        with lock:
            done.add(i); atomic_json(progress,{'url':URL,'size':EXPECTED,'done':sorted(done)})
            if len(done)%8==0 or len(done)==len(chunks):
                print(json.dumps({'stage':'download','chunks_done':len(done),'chunks_total':len(chunks),'seconds':round(time.monotonic()-began,1)}),flush=True)

    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one,chunks))
    print('Verifying full archive MD5 and SHA256',flush=True)
    assert hash_file(part)==(MD5,SHA256)
    part.replace(final)
    return final


def extract(archive):
    ready = SOURCE/'raw_ready.json'
    if ready.exists(): return
    guard(12651998008 + 1024**3)
    count=0; total=0; began=time.monotonic()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            path = SOURCE/item.filename
            assert path.resolve().is_relative_to(SOURCE.resolve())
            if item.is_dir(): path.mkdir(parents=True,exist_ok=True); continue
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_name(path.name+'.extracting')
            with z.open(item) as src,temp.open('wb') as dst:
                for b in iter(lambda:src.read(1024*1024),b''):dst.write(b)
            assert temp.stat().st_size == item.file_size
            temp.replace(path); count+=1; total+=item.file_size
            if count%2000==0:print(json.dumps({'stage':'extract','files':count,'bytes':total,'seconds':round(time.monotonic()-began,1)}),flush=True)
    atomic_json(ready,{'archive_md5':MD5,'archive_sha256':SHA256,'files_crc_checked':count,'uncompressed_bytes':total,'elapsed_seconds':time.monotonic()-began})
    print(json.dumps({'stage':'raw_ready','files':count,'bytes':total}),flush=True)


if __name__=='__main__':
    SOURCE.mkdir(exist_ok=True)
    extract(download_dataset())
