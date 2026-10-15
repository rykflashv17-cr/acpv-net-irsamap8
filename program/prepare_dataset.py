#!/usr/bin/env python3
"""IRSAMap V2 -> ACPV-Net eight-class targets, with bounded disk usage."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from PIL import Image, ImageDraw
from rasterio.features import rasterize, shapes as raster_shapes
from shapely import make_valid, set_precision
from shapely.geometry import shape, box, mapping, Polygon, Point, GeometryCollection
from shapely.affinity import translate
from shapely.ops import unary_union, linemerge, polygonize
from shapely.strtree import STRtree
from scipy.ndimage import distance_transform_edt

from project_paths import ROOT, SOURCE, DATA, RAW
sys.path.insert(0,str(ROOT/'program'))

PREVIOUS = ROOT/'data/irsamap_acpv8'

QA = ROOT/'qa/label_correction'
VERSION = 'official-rvwsb-semantic-v3'
_storage_snapshot=None
NAMES = ['farmland', 'tree', 'grass', 'water', 'building', 'road_area', 'sport', 'bareland']
COLORS = [(255,253,145),(32,216,109),(1,252,119),(20,197,246),(210,75,97),(255,200,1),(255,156,95),(204,181,206)]
SOURCE_IDS = {0:7, 10:0, 11:1, 12:2, 21:3, 22:3, 23:3, 24:3, 31:4, 32:5, 34:6, 40:7}
PRIORITY = [5,0,2,1,3,6,4]  # Low -> high: road, vegetation, water, sport, building.
AUGS = ['rot0','rot90','rot180','rot270','flip_h','flip_v','flip_diag','flip_anti_diag']
SCALE = 0.086164


def save_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n')
    temp.replace(path)


def guard(pending=0,force=False):
    global _storage_snapshot
    # Each window reserves its full declared upper bound. Reuse a recent scan
    # plus accumulated reservations to avoid repeatedly stat-ing 350k files.
    # Refresh before a phase, after10minutes, or whenever conservative usage is
    # near the unchanged stop threshold. Every window still checks free space.
    plan=json.loads((ROOT/'storage_budget.json').read_text())
    now=time.monotonic();fs=os.statvfs(ROOT);available=fs.f_bavail*fs.f_frsize
    if _storage_snapshot and not force:
        previous,checked,reserved=_storage_snapshot
        projected=previous['usage']['effective_bytes']+reserved+pending+32*1024**2
        if now-checked<600 and projected<plan['stop_new_writes_at_bytes'] and available-pending>=plan['filesystem_minimum_available_bytes']:
            report={**previous,'check_mode':'bounded_reservations_after_full_scan','pending_bytes':pending,
                    'reserved_since_snapshot_bytes':reserved,'projected_bytes':projected,
                    'filesystem_available_bytes':available,'snapshot_age_seconds':round(now-checked,1),
                    'checked_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            _storage_snapshot=(previous,checked,reserved+pending+32*1024**2)
            save_json(QA/'last_storage_guard.json',report);return
    result=subprocess.run([sys.executable,str(ROOT/'storage_guard.py'),'--pending-bytes',str(pending)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(f'Storage guard failed: {result.stdout} {result.stderr}')
    report=json.loads(result.stdout);report['checked_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());report['check_mode']='full_scan'
    _storage_snapshot=(report,time.monotonic(),pending)
    save_json(QA/'last_storage_guard.json',report)


def polygons(g):
    if g.is_empty: return []
    if g.geom_type=='Polygon': return [g]
    if hasattr(g,'geoms'): return [p for child in g.geoms for p in polygons(child)]
    return []


def line_parts(g):
    if g.is_empty: return []
    if g.geom_type in ['LineString','LinearRing']:return [g]
    if hasattr(g,'geoms'): return [p for child in g.geoms for p in line_parts(child)]
    return []


def source_code(value):
    if value is None: return None
    return int(float(value))


def feature_class(layer, properties):
    value=source_code(properties.get('code'))
    if layer=='vege':
        if value not in (10,11,12):raise ValueError(f'Unknown vegetation code: {value}')
        return SOURCE_IDS[value]
    fixed={'building':(4,31),'roadarea':(5,32),'sport':(6,34)}
    if layer in fixed:
        cid, expected=fixed[layer]
        if value not in (None,0,expected):raise ValueError(f'Conflicting {layer} code: {value}')
        return cid
    if layer=='water':
        if value not in (None,0,21,22,23,24):raise ValueError(f'Unknown water code: {value}')
        return 3
    raise ValueError(layer)


def load_source(split, stem):
    stats={'semantic_authority':'official SegLabel_rvwsb','repaired_features':0,'native_building_instances':0}
    with Image.open(RAW/split/'SegLabel_rvwsb'/(stem+'.png')) as original:arr=np.asarray(original)
    if arr.shape!=(1024,1024):raise ValueError(f'Composite shape {arr.shape}')
    unknown=set(map(int,np.unique(arr)))-set(SOURCE_IDS)
    if unknown:raise ValueError(f'Unexpected composite codes: {unknown}')
    reference=np.empty(arr.shape,np.uint8)
    for code,cid in SOURCE_IDS.items():reference[arr==code]=cid
    # Semantic classes come exclusively from the official composite. Original
    # GeoJSON order/code conflicts cannot overwrite the pixel supervision.
    parts=[[] for _ in range(8)]
    for geometry,cid in raster_shapes(reference,connectivity=4):
        parts[int(cid)].extend(polygons(make_valid(shape(geometry))))
    visible=[unary_union(pp) if pp else GeometryCollection() for pp in parts]
    if abs(sum(g.area for g in visible)-1024**2)>.01:raise ValueError('Official raster partition area mismatch')
    # Native building vectors supply instance identities inside the official
    # building class only; they never change the semantic pixel labels.
    source=RAW/split/'label_geojson_building'/(stem+'.geojson')
    instance_shapes=[];tile=box(0,0,1024,1024)
    if source.exists():
        for index,feature in enumerate(json.loads(source.read_text()).get('features',[]),1):
            if not feature.get('geometry'):continue
            if feature_class('building',feature.get('properties') or {})!=4:raise ValueError('Unexpected building class')
            g=shape(feature['geometry'])
            if not g.is_valid:g=make_valid(g);stats['repaired_features']+=1
            g=g.intersection(tile)
            if not g.is_empty:instance_shapes.append((g,index))
    buildings=[]
    if (reference==4).any():
        instances=rasterize(instance_shapes,out_shape=(1024,1024),fill=0,dtype='int32') if instance_shapes else np.zeros((1024,1024),np.int32)
        instances[reference!=4]=0
        unknown_building=(reference==4)&(instances==0)
        stats['native_building_instances']=len(instance_shapes)
        stats['building_pixels_without_native_instance_before_fill']=int(unknown_building.sum())
        if instances.any() and unknown_building.any():
            distance,nearest=distance_transform_edt(instances==0,return_indices=True)
            fill=unknown_building&(distance<=2.0)
            instances[fill]=instances[tuple(nearest[:,fill])]
        stats['building_pixels_without_native_instance_after_fill']=int(((reference==4)&(instances==0)).sum())
        for geometry,instance in raster_shapes(instances,mask=reference==4,connectivity=4):
            buildings.extend(polygons(make_valid(shape(geometry))))
    stats['source_code_counts']={str(int(c)):int(n) for c,n in zip(*np.unique(arr,return_counts=True))}
    return visible, buildings, reference, stats


def label_faces(faces, old_polys, old_classes):
    tree=STRtree(old_polys)
    ranks={7:-1,**{cid:i for i,cid in enumerate(PRIORITY)}}
    labeled=[]
    for face in faces:
        if face.is_empty or face.area<1e-8:continue
        point=face.representative_point()
        ids=tree.query(point,predicate='within')
        if len(ids)!=1:
            ids=tree.query(point,predicate='intersects')
        if not len(ids):
            nearest=int(tree.nearest(point))
            if old_polys[nearest].distance(point)>1e-7:raise ValueError('Face center outside original semantic partition')
            ids=[nearest]
        # Tiny coincident/sliver faces can have a representative point on more
        # than one original polygon. Apply the same documented semantic priority.
        winner=max(ids,key=lambda i:ranks[old_classes[int(i)]])
        labeled.append((face,old_classes[int(winner)]))
    return labeled


def class_mask(labeled,return_resolved=False):
    result=rasterize([(g,int(c)) for g,c in labeled],out_shape=(512,512),fill=255,dtype='uint8')
    missing=np.argwhere(result==255)
    if len(missing):
        if len(missing)>128:raise ValueError(f'Raster partition has {len(missing)} unassigned pixels')
        geoms=[g for g,c in labeled];tree=STRtree(geoms);ranks={7:-1,**{cid:i for i,cid in enumerate(PRIORITY)}}
        for y,x in missing:
            point=Point(float(x)+.5,float(y)+.5)
            ids=tree.query(point,predicate='intersects')
            if not len(ids):
                nearest=int(tree.nearest(point))
                if geoms[nearest].distance(point)>1e-7:raise ValueError('Actual gap in raster geometry partition')
                ids=[nearest]
            winner=max(ids,key=lambda i:ranks[labeled[int(i)][1]])
            result[y,x]=labeled[int(winner)][1]
    return (result,len(missing)) if return_resolved else result


def geometry_agreement(raster,reference):
    agreement=float((raster==reference).mean());ious={}
    for cid in np.unique(reference):
        a=raster==cid;b=reference==cid
        ious[int(cid)]=float((a&b).sum()/(a|b).sum())
    return agreement,ious


def planar_crop(visible, buildings, x, y, allow_raster_fallback=True):
    window=box(x,y,x+512,y+512)
    old_polys=[]; old_classes=[]
    for cid,g in enumerate(visible):
        for p in polygons(g.intersection(window)):
            old_polys.append(translate(p,xoff=-x,yoff=-y));old_classes.append(cid)
    old_labeled=list(zip(old_polys,old_classes))
    original_mask,original_ties=class_mask(old_labeled,return_resolved=True)
    original_counts=np.bincount(original_mask.ravel(),minlength=8)
    original_areas=[sum(p.area for p,c in old_labeled if c==cid) for cid in range(8)]
    if (original_mask==255).any():raise ValueError('Raster partition contains unassigned pixels')
    boundaries=[p.boundary for p in old_polys]
    # Keep boundaries between adjoining building instances even when their class is identical.
    for p in buildings:
        if p.intersects(window):
            b=p.boundary.intersection(visible[4]).intersection(window)
            if not b.is_empty:boundaries.append(translate(b,xoff=-x,yoff=-y))
    chosen=None; agreement=None; used_tolerance=0.0;used_grid=0.0;last_error=None
    # Simplify each shared chain once, preserving junction endpoints, then re-node the graph.
    for grid in [0.0,0.001]:
        graph=unary_union(boundaries if not grid else [set_precision(b,grid) for b in boundaries])
        parts=line_parts(graph)
        merged=linemerge(parts) if len(parts)>1 else parts[0]
        chains=line_parts(merged)
        for tolerance in [0.75,0.35,0.10,0.0]:
            try:
                simplified=[p.simplify(tolerance,preserve_topology=True) for p in chains]
                noded=unary_union(simplified)
                faces=list(polygonize(noded))
                labeled=label_faces(faces,old_polys,old_classes)
                area_error=abs(sum(p.area for p,c in labeled)-512**2)
                if area_error>0.005:
                    last_error=f'area_error={area_error}';continue
                if not all(p.is_valid for p,c in labeled):last_error='invalid face';continue
                mask,simplified_ties=class_mask(labeled,return_resolved=True)
                agreement=float((mask==original_mask).mean())
                agreement,geom_ious=geometry_agreement(mask,original_mask)
                if agreement<.99 or min(geom_ious.values())<.95:
                    last_error=f'geometry agreement={agreement}, min class IoU={min(geom_ious.values())}';continue
                missing_classes=set(np.unique(original_mask))-set(np.unique(mask))
                if any(original_counts[c]>4 or original_areas[c]>=.01 for c in missing_classes):
                    last_error=f'class disappeared: {list(map(int,missing_classes))}';continue
                chosen=labeled;used_tolerance=tolerance;used_grid=grid;break
            except Exception as error:last_error=repr(error)
        if chosen is not None:break
    if chosen is None:
        # The unsimplified official raster faces are an exact partition. This
        # fallback preserves all class pixels and rasterized building instances.
        exact=[]
        for cid in range(8):
            if cid==4 and buildings and (original_mask==4).any():
                inst=[(translate(g.intersection(window),xoff=-x,yoff=-y),i) for i,g in enumerate(buildings,1) if g.intersects(window)]
                image=rasterize(inst,out_shape=(512,512),fill=0,dtype='int32')
                for geometry,value in raster_shapes(image,mask=original_mask==4,connectivity=4):
                    exact.extend((p,cid) for p in polygons(make_valid(shape(geometry))))
            else:
                for geometry,value in raster_shapes(original_mask,mask=original_mask==cid,connectivity=4):
                    exact.extend((p,cid) for p in polygons(make_valid(shape(geometry))))
        if not np.array_equal(class_mask(exact),original_mask):raise ValueError(f'Exact official raster fallback failed: {last_error}')
        chosen=exact;agreement=1.0;used_tolerance=0.0;used_grid=0.0;simplified_ties=0;missing_classes=set()
        exact_fallback=True
    else:exact_fallback=False
    geom_mask=class_mask(chosen)
    agreement,geom_ious=geometry_agreement(geom_mask,original_mask)
    if agreement<.99 or min(geom_ious.values())<.95:raise ValueError('Geometry exceeds documented approximation limits')
    coords=set();features=[]
    for index,(p,cid) in enumerate(chosen):
        for ring in [p.exterior,*p.interiors]:coords.update((round(float(a),4),round(float(b),4)) for a,b in list(ring.coords)[:-1])
        features.append({'type':'Feature','id':index,'properties':{'class_id':int(cid),'class_name':NAMES[cid]},'geometry':mapping(p)})
    # Semantic training masks stay exact. The vector boundary approximation is
    # checked separately, so raster staircase corners are not all vertex targets.
    return original_mask,features,sorted(coords),{'simplification_tolerance_px':used_tolerance,'simplified_mask_agreement':agreement,
           'geometry_raster_agreement':agreement,'geometry_class_iou':geom_ious,
           'official_raster_exact':True,'exact_raster_geometry_fallback':exact_fallback,'faces':len(features),'vertices':len(coords),
           'raster_boundary_ties_resolved_original':original_ties,'raster_boundary_ties_resolved_simplified':simplified_ties,
           'topology_precision_grid_px':used_grid,'numerical_sliver_classes_removed':[]}

def process_source(task):
    split, filename=task;stem=Path(filename).stem
    start=time.monotonic()
    path=DATA/split/'metadata'/f'{stem}.source.json'
    if path.exists():
        cached=json.loads(path.read_text())
        if cached.get('version')==VERSION and all((DATA/split/'images'/(r['id']+'.png')).exists() and (DATA/split/'masks'/(r['id']+'.png')).exists()
               and (DATA/split/'vertices'/(r['id']+'.json')).exists() and (DATA/split/'annotations'/(r['id']+'.geojson')).exists() for r in cached['crops']):
            return cached
    visible, buildings, reference, stats=load_source(split,stem)
    results=[]
    for y in [0,512]:
        for x in [0,512]:
            sample=f'{stem}_x{x:04d}_y{y:04d}'
            mask,features,vertices,geometry_stats=planar_crop(visible,buildings,x,y)
            ref=reference[y:y+512,x:x+512]
            if not np.array_equal(mask,ref):raise ValueError('Geometry changed official semantic labels')
            base=DATA/split
            for d in ['images','masks','vertices','annotations','metadata']:(base/d).mkdir(parents=True,exist_ok=True)
            image_path=base/'images'/(sample+'.png')
            if not image_path.exists():
                prior_image=PREVIOUS/split/'images'/(sample+'.png')
                if prior_image.exists(): os.link(prior_image,image_path)
                else:
                    with Image.open(RAW/split/'image'/filename) as source_rgb:
                        source_rgb.convert('RGB').crop((x,y,x+512,y+512)).save(image_path,compress_level=3)
            mask_path=base/'masks'/(sample+'.png')
            tmp=mask_path.with_suffix('.png.tmp');Image.fromarray(mask).save(tmp,format='PNG',compress_level=3);tmp.replace(mask_path)
            save_json(base/'vertices'/(sample+'.json'),vertices)
            save_json(base/'annotations'/(sample+'.geojson'),{'type':'FeatureCollection','coordinate_space':'crop pixel edges, x right y down; not geographic CRS','image_size':[512,512],'features':features})
            prior_mask=PREVIOUS/split/'masks'/(sample+'.png')
            previous_mask=np.asarray(Image.open(prior_mask)) if prior_mask.exists() else mask
            confusion=np.bincount((previous_mask.astype(np.int16)*8+mask).ravel(),minlength=64).reshape(8,8)
            prior_vertices=PREVIOUS/split/'vertices'/(sample+'.json')
            previous_vertices=json.loads(prior_vertices.read_text()) if prior_vertices.exists() else None
            same_vertices=previous_vertices==[list(v) for v in vertices]
            counts=np.bincount(mask.ravel(),minlength=8)
            ious={}
            for cid in range(8):
                a=(mask==cid);b=(ref==cid);union=int((a|b).sum())
                ious[str(cid)]=float((a&b).sum()/union) if union else None
            row={'id':sample,'split':split,'source_image':filename,'crop_xyxy':[x,y,x+512,y+512],
                 'class_pixel_counts':counts.tolist(),'official_composite_agreement':float((mask==ref).mean()),
                 'official_composite_class_iou':ious,'version':VERSION,'semantic_authority':'official SegLabel_rvwsb',
                 'changed_pixels_from_previous':int((previous_mask!=mask).sum()) if prior_mask.exists() else None,
                 'previous_vs_corrected_confusion':confusion.tolist() if prior_mask.exists() else None,'vertices_unchanged_from_previous':same_vertices,**geometry_stats}
            save_json(base/'metadata'/(sample+'.json'),row);results.append(row)
    result={'split':split,'source_image':filename,'version':VERSION,'source_stats':stats,'crops':results,'seconds':time.monotonic()-start}
    save_json(path,result)
    return result


def list_tasks(limit=None):
    tasks=[]
    for split in ['train','val','test']:
        names=sorted(p.name for p in (RAW/split/'image').glob('*.png'))
        if limit:
            ids=np.linspace(0,len(names)-1,min(limit,len(names)),dtype=int)
            names=[names[int(i)] for i in ids]
        tasks.extend((split,name) for name in names)
    return tasks


def write_rules():
    DATA.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
    save_json(DATA/'classes.json',[{'train_id':i,'name':n,'color':COLORS[i]} for i,n in enumerate(NAMES)])
    save_json(DATA/'preprocessing_rules.json',{
        'version':VERSION,'source':'official IRSAMap_V2; original train/val/test splits retained',
        'num_classes':8,'source_to_train':SOURCE_IDS,'semantic_authority':'official SegLabel_rvwsb; no vector class priority overrides',
        'mask_requirement':'every crop must equal the mapped official raster pixel for pixel',
        'geometry_requirement':'valid, non-overlapping complete planar partition; >=99% raster agreement and >=95% IoU for every present class; semantic masks remain exactly official',
        'bareland':'code 40 and residual code 0; follows the paper definition of areas outside vegetation/water/artificial cover',
        'water':'codes21/22/23/24 merged; code24 documented in archive README as sea surface',
        'building_instances':'native identities rasterized only inside official building pixels; unassigned border pixels within 2px use nearest known instance',
        'coordinates':'pixel edges, x right y down; vertices at edge512 clipped to511 for heatmaps',
        'geometry_simplification':'shared chains simplified together within0.75px to avoid staircase vertices; check approximation separately; fallback to exact raster polygon partition',
        'heatmap_sigma':3.0,'heatmap_cutoff':1e-8,'augmentations':AUGS,'vae_latent_shape':[3,128,128],
        'vae_scale_factor':SCALE,'latent_sampling':'same deterministic per-item CPU seed as previous version',
        'rgb_storage':'direct crops from official RGB; optional hardlinks when a previous validated crop exists','temporary_heatmap_files':'none',
        'previous_version_required':False,
        'source_paper':'https://arxiv.org/html/2508.16272v1','source_repository':'https://github.com/ucas-dlg/IRSAMap'})


def convert(workers,limit):
    write_rules();tasks=list_tasks(limit);results=[];errors=[];began=time.monotonic()
    if not tasks:raise ValueError('No source images found')
    guard(24_000_000_000,force=True)
    with ProcessPoolExecutor(max_workers=min(workers,len(tasks))) as pool:
        completed=0
        # Bound the submitted write window, including queued tasks. A failed guard
        # stops before the next window, rather than leaving the whole dataset queued.
        for start in range(0,len(tasks),256):
            guard(4_000_000_000)
            futures={pool.submit(process_source,t):t for t in tasks[start:start+256]}
            for future in as_completed(futures):
                task=futures[future];completed+=1
                try:results.append(future.result())
                except Exception as e:
                    errors.append({'task':task,'error':repr(e)})
                    print(json.dumps({'stage':'conversion_error','task':task,'error':repr(e)}),flush=True)
            print(json.dumps({'stage':'convert','sources_done':completed,'sources_total':len(tasks),'errors':len(errors),'seconds':round(time.monotonic()-began,1)}),flush=True)
    summary={'original_images':len(tasks),'successful_originals':len(results),'crops':4*len(results),'errors':errors,'seconds':time.monotonic()-began,'pilot':bool(limit)}
    save_json(QA/('pilot_conversion.json' if limit else 'conversion_summary.json'),summary)
    if errors:raise RuntimeError(f'{len(errors)} source images failed; see conversion summary')
    # Compact manifest, separate from per-sample geometry.
    for split in ['train','val','test']:
        rows=[row for result in results if result['split']==split for row in result['crops']]
        rows.sort(key=lambda r:r['id'])
        save_json(DATA/split/'manifest.json',rows)
    return summary


def legacy_heatmap(vertices):
    sigma=3.0;radius=math.ceil(sigma*math.sqrt(-2*math.log(1e-8)))+1
    result=np.zeros((512,512),np.float32)
    for a,b in vertices:
        x=float(np.clip(a,0,511));y=float(np.clip(b,0,511))
        x0=max(0,int(math.floor(x))-radius);x1=min(512,int(math.ceil(x))+radius+1)
        y0=max(0,int(math.floor(y))-radius);y1=min(512,int(math.ceil(y))+radius+1)
        yy,xx=np.mgrid[y0:y1,x0:x1]
        values=np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma)).astype(np.float32)
        np.maximum(result[y0:y1,x0:x1],values,out=result[y0:y1,x0:x1])
    peak=float(result.max())
    if peak>0:result/=peak
    result[result<1e-8]=0
    return result


def heatmap(vertices):
    coordinates=np.asarray(vertices,dtype=np.float64)
    if len(coordinates) and np.allclose(coordinates,np.round(coordinates),rtol=0,atol=1e-7):
        # For integer raster-boundary vertices, max Gaussian is exactly the
        # Gaussian of the distance to the nearest vertex. Avoid per-vertex grids.
        points=np.clip(np.round(coordinates).astype(np.int32),0,511)
        grid=np.ones((512,512),dtype=bool);grid[points[:,1],points[:,0]]=False
        distance=distance_transform_edt(grid)
        result=np.exp(-(distance**2)/18.0).astype(np.float32)
        result[result<1e-8]=0
        return result
    return legacy_heatmap(vertices)


def augment(array, name):
    if name=='rot0':return array
    if name.startswith('rot'):return np.rot90(array,int(name[3:])//90)
    if name=='flip_h':return np.fliplr(array)
    if name=='flip_v':return np.flipud(array)
    if name=='flip_diag':return array.T
    if name=='flip_anti_diag':return array.T[::-1,::-1]
    raise ValueError(name)


def rows_for_split(split,limit):
    rows=json.loads((DATA/split/'manifest.json').read_text())
    if limit:
        sources=sorted(set(r['source_image'] for r in rows))
        ids=np.linspace(0,len(sources)-1,min(limit,len(sources)),dtype=int)
        chosen={sources[int(i)] for i in ids}
        rows=[r for r in rows if r['source_image'] in chosen]
    return rows


def latent_path(split,sample,aug):
    return DATA/split/'heatmap_augmented_latent_kl-4'/aug/'z'/(sample+'.pt')


def prepare_batch(jobs):
    images=[];seeds=[];cached={}
    for split,sample,aug in jobs:
        key=(split,sample)
        if key not in cached:
            vertices=json.loads((DATA/split/'vertices'/(sample+'.json')).read_text())
            cached[key]=heatmap(vertices)
        a=augment(cached[key],aug).copy()
        images.append(np.repeat((a*2-1)[None,:,:],3,axis=0))
        seeds.append(int.from_bytes(hashlib.sha256(f'IRSAMapV2:{split}:{sample}:{aug}'.encode()).digest()[:8],'little')%(2**63))
    return np.stack(images),seeds


def preview(rows):
    palette=np.array(COLORS,np.uint8)
    canvas=Image.new('RGB',(512*3,560*len(rows)),(255,255,255));draw=ImageDraw.Draw(canvas)
    for index,(split,row) in enumerate(rows):
        sample=row['id'];rgb=Image.open(DATA/split/'images'/(sample+'.png')).convert('RGB')
        mask=np.asarray(Image.open(DATA/split/'masks'/(sample+'.png')))
        color=Image.fromarray(palette[mask]);overlay=Image.blend(rgb,color,.45)
        vertices=json.loads((DATA/split/'vertices'/(sample+'.json')).read_text())
        vd=ImageDraw.Draw(overlay)
        for x,y in vertices:vd.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill=(255,255,255))
        y=index*560
        for col,im in enumerate([rgb,color,overlay]):canvas.paste(im,(512*col,y+35))
        draw.text((5,y+5),f'{split}/{sample} | RGB / 8-class mask / overlay+vertices',fill=(0,0,0))
    path=QA/('sample_preview.png');canvas.save(path)


def encode(batch_size,limit):
    import torch
    from frozen_vae import FrozenVAE,CKPT_SHA
    if len(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(','))!=1:raise RuntimeError('Expose exactly one assigned GPU through CUDA_VISIBLE_DEVICES')
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; CPU latent encoding is not started')
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    # Avoid cuDNN autotuning's large temporary workspaces on the shared machine.
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(0.45)
    jobs=[];existing=0
    for split in ['train','val','test']:
        augs=AUGS if split=='train' else ['rot0']
        for row in rows_for_split(split,limit):
            for aug in augs:
                path=latent_path(split,row['id'],aug)
                if path.exists():existing+=1
                elif row.get('vertices_unchanged_from_previous',False):
                    old=PREVIOUS/split/'heatmap_augmented_latent_kl-4'/aug/'z'/(row['id']+'.pt')
                    if not old.exists():raise ValueError('Missing previous latent for identical vertex targets')
                    path.parent.mkdir(parents=True,exist_ok=True);os.link(old,path);existing+=1
                else:jobs.append((split,row['id'],aug))
        for aug in augs:(DATA/split/'heatmap_augmented_latent_kl-4'/aug/'z').mkdir(parents=True,exist_ok=True)
    guard(len(jobs)*(3*128*128*4+4096)+1024**3,force=True)
    model=FrozenVAE().cuda()
    # Validate equivalent attention before replacing the large explicit attention matrix.
    from ldm.modules.diffusionmodules.model import AttnBlock
    from frozen_vae import sdpa_forward
    attention=next(m for m in model.modules() if isinstance(m,AttnBlock))
    with torch.inference_mode():
        probe=torch.randn(1,attention.in_channels,16,16,device='cuda')
        original=attention(probe);fast=sdpa_forward(attention,probe)
        attention_error=float((original-fast).abs().max())
        if attention_error>1e-3:raise ValueError(f'SDPA attention equivalence check failed: {attention_error}')
    model.use_sdpa()
    began=time.monotonic();done=0;moments=[0,0.0,0.0];peak=0
    batches=[jobs[i:i+batch_size] for i in range(0,len(jobs),batch_size)]
    with ThreadPoolExecutor(max_workers=1) as pool,torch.inference_mode():
        future=pool.submit(prepare_batch,batches[0]) if batches else None
        for index,batch in enumerate(batches):
            array,seeds=future.result()
            future=pool.submit(prepare_batch,batches[index+1]) if index+1<len(batches) else None
            if index%2048==0:guard(2048*batch_size*(3*128*128*4+4096)+256*1024*1024)
            image=torch.from_numpy(array).cuda(non_blocking=False)
            latent=model.sample(image,seeds).cpu().contiguous()
            if latent.shape!=(len(batch),3,128,128) or not torch.isfinite(latent).all():raise ValueError('Invalid latent batch')
            for item,z in zip(batch,latent):
                path=latent_path(*item);tmp=path.with_suffix('.pt.tmp')
                # Clone avoids serializing the whole batch storage into each per-image file.
                torch.save(z.clone(),tmp);tmp.replace(path)
                moments[0]+=z.numel();moments[1]+=float(z.double().sum());moments[2]+=float(z.double().square().sum())
            done+=len(batch);peak=max(peak,torch.cuda.max_memory_allocated())
            if index%128==0 or index==len(batches)-1:
                elapsed=time.monotonic()-began
                status={'stage':'encode','latent_done':done,'latent_total':len(jobs),'existing':existing,'seconds':round(elapsed,1),
                        'latents_per_second':round(done/max(elapsed,.001),3),'gpu_peak_allocated_GB':round(peak/1e9,3)}
                save_json(QA/'encode_progress.json',status);print(json.dumps(status),flush=True)
    summary={'new_latents':done,'existing_latents':existing,'expected_latents':done+existing,'seconds':time.monotonic()-began,
             'vae_sha256':CKPT_SHA,'scale_factor':SCALE,'dtype':'torch.float32','posterior':'sample; deterministic per-item CPU seed',
             'attention':'mathematically equivalent torch SDPA','attention_probe_max_abs_error':attention_error,
             'encoder_precision':'float32; attention matmul TF32 disabled, cuDNN convolutions TF32 enabled','gpu':torch.cuda.get_device_name(0),
             'gpu_peak_allocated_bytes':peak,'augmentation_train':AUGS,'augmentation_val_test':['rot0'],'pilot':bool(limit)}
    summary['gpu_memory_allocator_limit_fraction']=0.45
    if moments[0]:summary.update(latent_mean=moments[1]/moments[0],latent_std=math.sqrt(moments[2]/moments[0]-(moments[1]/moments[0])**2))
    save_json(QA/('pilot_encoding.json' if limit else 'encoding_summary.json'),summary)
    return summary


def check_row(task):
    split,row,with_latents=task;sample=row['id'];base=DATA/split
    with Image.open(base/'images'/(sample+'.png')) as image:
        if image.size!=(512,512) or image.mode!='RGB':raise ValueError(f'Invalid RGB {split}/{sample}')
        image.load()
    with Image.open(base/'masks'/(sample+'.png')) as image:
        mask=np.asarray(image)
        if image.mode!='L' or mask.shape!=(512,512) or mask.dtype!=np.uint8:raise ValueError('Invalid mask shape/type')
        if int(mask.max())>7:raise ValueError('Invalid mask class IDs')
        counts=np.bincount(mask.ravel(),minlength=8)
    with Image.open(RAW/split/'SegLabel_rvwsb'/row['source_image']) as image:original=np.asarray(image)
    mapped=np.empty(original.shape,np.uint8)
    for code,cid in SOURCE_IDS.items():mapped[original==code]=cid
    x,y,x1,y1=row['crop_xyxy']
    if not np.array_equal(mask,mapped[y:y1,x:x1]):raise ValueError('Mask differs from official semantic pixels')
    vertices=np.asarray(json.loads((base/'vertices'/(sample+'.json')).read_text()))
    if vertices.ndim!=2 or vertices.shape[1]!=2 or not np.isfinite(vertices).all() or vertices.min()<0 or vertices.max()>512:raise ValueError('Invalid vertices')
    annotation=json.loads((base/'annotations'/(sample+'.geojson')).read_text())
    labeled=[(shape(f['geometry']),f['properties']['class_id']) for f in annotation['features']]
    if not all(p.is_valid for p,c in labeled):raise ValueError('Invalid polygons')
    if abs(sum(p.area for p,c in labeled)-512**2)>.005:raise ValueError('Invalid planar coverage')
    agreement,geom_ious=geometry_agreement(class_mask(labeled),mask)
    if agreement<.99 or min(geom_ious.values())<.95:raise ValueError('Mask and geometry exceed approximation limits')
    num_latents=0
    if with_latents:
        import torch
        for aug in AUGS if split=='train' else ['rot0']:
            z=torch.load(latent_path(split,sample,aug),map_location='cpu',weights_only=True)
            if not isinstance(z,torch.Tensor) or z.shape!=(3,128,128) or z.dtype!=torch.float32:raise ValueError(f'Invalid latent {split}/{sample}/{aug}')
            if not torch.isfinite(z).all() or float(z.std())<1e-6:raise ValueError('Nonfinite or degenerate latent')
            num_latents+=1
    return counts.tolist(),num_latents


def validate(limit,with_latents=True):
    if with_latents:
        import torch
        torch.set_num_threads(1)
    tasks=[];selected=[];summary={};total_counts=np.zeros(8,np.int64);errors=[];began=time.monotonic();latent_total=0
    split_counts={split:np.zeros(8,np.int64) for split in ['train','val','test']}
    for split in ['train','val','test']:
        rows=rows_for_split(split,limit);summary[split]={'crops':len(rows),'source_images':len(set(r['source_image'] for r in rows))}
        actual={p.stem for p in (DATA/split/'images').glob('*.png')}
        if not limit and actual!={r['id'] for r in rows}:raise ValueError(f'{split} manifest/images mismatch')
        agreement=[r['official_composite_agreement'] for r in rows]
        summary[split]['official_composite_agreement_mean']=float(np.mean(agreement))
        summary[split]['official_composite_agreement_min']=float(np.min(agreement))
        summary[split]['crops_below_90_percent_composite_agreement']=[r['id'] for r in rows if r['official_composite_agreement']<.9]
        tasks.extend((split,row,with_latents) for row in rows)
        selected.extend((split,rows[i]) for i in np.linspace(0,len(rows)-1,min(3,len(rows)),dtype=int))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(check_row,t):t for t in tasks}
        for i,f in enumerate(as_completed(futures),1):
            split,row,_=futures[f]
            try:
                counts,num=f.result();total_counts+=counts;split_counts[split]+=counts;latent_total+=num
            except Exception as e:errors.append({'split':split,'id':row['id'],'error':repr(e)})
            if i%512==0 or i==len(tasks):print(json.dumps({'stage':'validate','samples_checked':i,'total':len(tasks),'errors':len(errors)}),flush=True)
    preview(selected)
    for split,counts in split_counts.items():
        summary[split]['class_pixel_counts']=counts.tolist()
        summary[split]['class_pixel_fractions']=(counts/max(int(counts.sum()),1)).tolist()
    result={'splits':summary,'class_names':NAMES,'class_pixel_counts':total_counts.tolist(),
            'class_pixel_fractions':(total_counts/max(int(total_counts.sum()),1)).tolist(),'latent_files_checked':latent_total,
            'all_geometry_and_raster_consistent':not errors,'errors':errors,'seconds':time.monotonic()-began,
            'pilot':bool(limit),'latents_checked':with_latents,'version':VERSION,
            'all_masks_equal_official_pixels':not errors}
    save_json(QA/('pilot_validation.json' if limit else 'validation_summary.json'),result)
    if errors:raise RuntimeError(f'{len(errors)} samples failed validation')
    return result


def main():
    global RAW
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['convert','encode','validate'])
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--limit-per-split',type=int,default=None)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--without-latents',action='store_true')
    p.add_argument('--raw-root',type=Path,default=RAW)
    args=p.parse_args()
    RAW=args.raw_root
    if not 1<=args.workers<=32:p.error('workers must be 1..32')
    if args.stage=='convert':convert(args.workers,args.limit_per_split)
    elif args.stage=='encode':encode(args.batch_size,args.limit_per_split)
    else:validate(args.limit_per_split,not args.without_latents)


if __name__=='__main__':main()
