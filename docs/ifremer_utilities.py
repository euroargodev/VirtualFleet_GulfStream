"""
Helpers for simulations from Ifremer/Datarmor network or from my laptop
"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import path
import json

try:
    import regionmask
except:
    print("regionmask not found")
import argopy
from argopy import IndexFetcher as ArgoIndexFetcher

# Where do we find the VirtualFleet and VirtualFleet_GulfStream repositories ?
EUROARGODEV = os.path.expanduser('~/git/github/euroargodev')

# Import the VirtualFleet library
sys.path.insert(0, os.path.join(EUROARGODEV, "VirtualFleet"))
try:
    from virtualargofleet import VelocityField, VirtualFleet, FloatConfiguration
except:
    print("virtualargofleet not found")


def load_json(db_file):
    with open(db_file, 'r') as stream:
        db = json.load(stream)
    return db


def save_json(db, db_file):
    with open(db_file, 'w') as stream:
        json.dump(db, stream, indent=2)
    return db


def rect_box(SW, NE):
    # SW: lon_south_west, lat_south_west
    # NE: lon_north_east, lat_north_east
    return [[SW[0],SW[1]],[NE[0],SW[1]],[NE[0],NE[1]],[SW[0],NE[1]]]


def get_regions(dict_regions):
    boxes, numbers, names, abbrevs = [], [], [], []
    for ii, r in enumerate(dict_regions.items()):
        numbers.append(ii)
        names.append(r[1]['name'])
        abbrevs.append(r[0])
        # boxes.append(rect_box([r[1]['box'][0],r[1]['box'][2]],[r[1]['box'][1],r[1]['box'][3]]))
        box = r[1]['box'].replace('[', '').replace(']', '').split(",")
        box = [float(box[0]), float(box[1]), float(box[2]), float(box[3]), box[4].strip(), box[5].strip()]
        boxes.append(rect_box([box[0], box[2]], [box[1], box[3]]))
    regions = regionmask.Regions(boxes, numbers, names, abbrevs, name='BCmask')
    return regions



def get_regions(DB_entry):
    boxes, numbers, names, abbrevs, dict_regions = [], [], [], [], {}
    for ii, r in enumerate(DB_entry.items()):
        numbers.append(ii)
        names.append(r[1]['name'])
        abbrevs.append(r[0])
        # boxes.append(rect_box([r[1]['box'][0],r[1]['box'][2]],[r[1]['box'][1],r[1]['box'][3]]))
        box = r[1]['box'].replace('[', '').replace(']', '').split(",")
        box = [float(box[0]), float(box[1]), float(box[2]), float(box[3]), box[4].strip(), box[5].strip()]
        boxes.append(rect_box([box[0], box[2]], [box[1], box[3]]))
        dict_regions.update({r[0]: {'box': box, 'name': r[1]['name']}})
    regions = regionmask.Regions(boxes, numbers, names, abbrevs, name='BCmask')
    return regions, dict_regions


# Localise the simulations database json file, assuming a standard folder structure (unchanged from github repo):
DBFILE = os.path.sep.join([EUROARGODEV, "VirtualFleet_GulfStream", "data", "simulations_db.json"])
DB = load_json(DBFILE)
# dict_regions = DB['regions']
# regions = get_regions(dict_regions)
regions, dict_regions = get_regions(DB['regions'])


# dict_regions = {
#     'NATL': {'box': [-80,-5.,15,65, '2008-01-01', '2019-01-01'], 'name': 'North Atlantic'},
#     'GSE': {'box': [-75.,-48.,33,45.5, '2008-01-01', '2019-01-01'], 'name': 'Gulf Stream Extension'},
#     'ArgoWBC': {'box': [-75., -30., 36, 51, '2008-01-01', '2019-01-01'], 'name': 'Gulf Stream according to Argo-AST'},
#     'GLORYS-NATL': {'box': [-90., 5., 10., 70., '2008-01-01', '2019-01-01'], 'name': 'GLORYS domain'},
# }

def load_velocity(name='somovar', fpattern='default'):
    """Load a pre-defined VF velocity field instance"""

    if name == 'somovar':
        root = "/home/datawork-lops-oh" if not os.uname()[0] == 'Darwin' else "/Volumes/LOPS_OH"
        src = os.path.join(root, "somovar/WP1/data/GLOBAL-ANALYSIS-FORECAST-PHY-001-024")
        fpattern = "%s/*.nc" if fpattern == 'default' else fpattern
        VELfield = VelocityField(model='GLOBAL_ANALYSIS_FORECAST_PHY_001_024', src=fpattern % src)

    elif name == 'GLORYS':
        root = "/home/ref-ocean-reanalysis" if not os.uname()[0] == 'Darwin' else "/Volumes/REANALYSIS"
        src = os.path.join(root, "global-reanalysis-phy-001-030-daily")
        fpattern = "%s/*/*.nc" if fpattern == 'default' else fpattern
        VELfield = VelocityField(model='GLORYS12V1', src=fpattern % src, isglobal=True)

    elif name == 'GLORYS-NATL':
        root = "/home/datawork-lops-bluecloud" if not os.uname()[0] == 'Darwin' else "/Volumes/BLUECLOUD"
        src = os.path.join(root, "natl-reanalysis-phy-001-030-daily")
        fpattern = "%s/*/*.nc" if fpattern == 'default' else fpattern
        VELfield = VelocityField(model='GLORYS12V1', src=fpattern % src, isglobal=True)  # isglobal True is set on purpose !
        # VELfield = VelocityField(model='GLORYS12V1', src=fpattern % src, isglobal=False)

    elif name == 'ARMORD3D':
        root = "/home/ref-cmems-public" #if not os.uname()[0] == 'Darwin' else "/Volumes/REANALYSIS"
        src = os.path.join(root, "tac/multiobs/MULTIOBS_GLO_PHY_REP_015-002/ARMOR3D/data")
        fpattern = "%s/*/dataset-armor-3d-rep-weekly*.nc" if fpattern == 'default' else fpattern
        VELfield = VelocityField(model='GLORYS12V1', src=fpattern % src, isglobal=True)

    return VELfield


def load_aviso(box=dict_regions['GSE']['box']):
    # MDT and more:
    aviso = xr.open_dataset("~/data/AVISO/data/mdt_cnes_cls2013_global.nc")
    lon_attrs = aviso['lon'].attrs
    aviso = aviso.isel(time=0)
    aviso['speed'] = xr.DataArray(np.sqrt(aviso['u'] ** 2 + aviso['v'] ** 2), dims=['lat', 'lon'])
    aviso['lon'] = (((aviso['lon'] + 180) % 360) - 180)
    aviso['lon'].attrs = lon_attrs
    aviso = aviso.sortby('lon')

    # Squeeze to North-Atlantic:
    large_box = [-90, 5, 0, 70]
    a = aviso.where(aviso['lon'] >= large_box[0], drop=True)
    a = a.where(aviso['lon'] <= large_box[1], drop=True)
    a = a.where(aviso['lat'] >= large_box[2], drop=True)
    aviso = a.where(aviso['lat'] <= large_box[3], drop=True)

    # EKE
    import scipy.io
    mat = scipy.io.loadmat(os.path.expanduser("~/data/AVISO/data/western_north_atlantic/matlab/eke_2000_2010.mat"))
    eke = xr.DataArray(mat['EKE'], coords={'t': mat['years'][0, :], 'lon': mat['x'][:, 0], 'lat': mat['y'][:, 0]},
                       dims=['t', 'lat', 'lon'])
    eke['lon'] = (((eke['lon'] + 180) % 360) - 180)
    aviso['eke'] = eke.interp({'lon': aviso['lon'], 'lat': aviso['lat']})

    # Find the latitude of the max AVISO speed within the GS box:
    a = aviso.where(aviso['lon'] >= box[0], drop=True)
    a = a.where(aviso['lon'] <= box[1], drop=True)
    a = a.where(aviso['lat'] >= box[2], drop=True)
    a = a.where(aviso['lat'] <= box[3], drop=True)
    GSpos_amp = a['lat'][a['speed'].argmax('lat')].rename('pos').to_dataset()
    # a['speed'].plot(vmin=0.1, vmax=0.5)
    # GSpos_amp['pos'].plot()

    # Find the latitude of the max AVISO EKE within the GS box:
    # a['eke'].mean('t').plot(vmin=0, vmax=2000)
    GSpos = a['lat'][a['eke'].mean('t').argmax('lat')]
    GSpos = GSpos.rename('pos').to_dataset()

    Rkm = 75  # Rossby radius in km
    km2deg = 360 / (2 * np.pi * 6371)
    Rdg = Rkm * km2deg
    GSpos['top'] = GSpos['pos'] + 2 * Rdg
    GSpos['btm'] = GSpos['pos'] - 2 * Rdg
    # GSpos['top'].rolling(lon=12, center=True).mean().plot()
    # GSpos['pos'].rolling(lon=12, center=True).mean().plot()
    # GSpos['btm'].rolling(lon=12, center=True).mean().plot()

    aviso['GSpos'] = GSpos_amp['pos']
    aviso['GSpos'].attrs = {'long_name': 'Latitude of max speed'}
    aviso['EKEmax'] = GSpos['pos']
    aviso['EKEmax'].attrs = {'long_name': 'Latitude of max EKE'}
    aviso['EKEmax_top'] = GSpos['top']
    aviso['EKEmax_top'].attrs = {'long_name': 'Latitude of max EKE + 2 Rossby radius'}
    aviso['EKEmax_btm'] = GSpos['btm']
    aviso['EKEmax_btm'].attrs = {'long_name': 'Latitude of max EKE - 2 Rossby radius'}

    return aviso



def get_index_for_regions(dict_regions):
    argopy.clear_cache()

    for r in regions:
        print(r.abbrev)
        index_box = dict_regions[r.abbrev]['box']
        # box = box.replace('[', '').replace(']', '').split(",")
        # index_box = [float(box[0]), float(box[1]), float(box[2]), float(box[3]), box[4].strip(), box[5].strip()]

        # argo = ArgoIndexFetcher(src='gdac', ftp=OPTIONS['local_ftp'], cache=True).region(index_box)
        argo = ArgoIndexFetcher(src='gdac', cache=True).region(index_box)
        dict_regions[r.abbrev]['fetcher'] = argo
        dict_regions[r.abbrev]['index'] = argo.index
        # Add cycle number to each profiles:
        dict_regions[r.abbrev]['index']['cycle_number'] = dict_regions[r.abbrev]['index'].apply(
            lambda x: int("".join([c for c in x['file'].split("/")[-1].split("_")[-1].split(".nc")[0] if c.isdigit()])),
            axis=1)
        # Add profile direction:
        dict_regions[r.abbrev]['index']['direction'] = dict_regions[r.abbrev]['index'].apply(
            lambda x: 'D' if 'D' in x['file'].split("/")[-1].split("_")[-1].split(".nc")[0] else 'A', axis=1)

    return dict_regions


def squeeze_region(regions, domain_name, simu_index):
    p = path.Path([(regions[domain_name].bounds[0], regions[domain_name].bounds[1]),
                   (regions[domain_name].bounds[2], regions[domain_name].bounds[1]),
                   (regions[domain_name].bounds[2], regions[domain_name].bounds[3]),
                   (regions[domain_name].bounds[0], regions[domain_name].bounds[3])])
    return simu_index.iloc[p.contains_points(np.array(simu_index[['longitude', 'latitude']]))[:, np.newaxis]]


def squeeze_natl(regions, simu_index):
    """Make sure profiles are within the analysed domain"""
    simu_index = squeeze_region(regions, 'NATL', simu_index)

    # Try to get rid of the Azores hotspot:
    # p = path.Path([(-37.5, 23.5), (-37.5+3, 23.5), (-37.5+3, 23.5+3), (-37.5, 23.5+3)])
    # p = path.Path([(-40, 23), (-40+5, 23), (-40+5, 23+5), (-40, 23+5)])
    # simu_index = simu_index.iloc[~p.contains_points(np.array(simu_index[['longitude', 'latitude']]))[:, np.newaxis]]

    return simu_index


def squeeze_max_cyc_number(maxNC, simu_index):
    """Possibly ensure a max number of profiles per floats in an index"""
    return simu_index[simu_index['cycle_number'] <= maxNC]


def compute_prof_density(simu_index, r=3, nanmask=True):
    from xhistogram.xarray import histogram
    ds = simu_index.to_xarray()
    xbins = np.arange(-81, 0, r)
    ybins = np.arange(15, 69, r)
    obs = histogram(ds['longitude'], ds['latitude'], bins=[xbins, ybins])
    if nanmask:
        obs = obs.where(obs > 0, other=np.NaN)
    return obs


def remove_weirdo_from_ds(ds, dxmax=3, dymax=3):
    """Drop floats with trajectory points lon/lat changes larger than dxmax/dymax"""
    weirdo = []
    for traj in ds['traj']:
        this = ds.sel(traj=traj)
        if this['lon'].diff('obs').max() > dxmax:
            weirdo.append(np.unique(this['trajectory'])[0])
        if this['lat'].diff('obs').max() > dymax:
            weirdo.append(np.unique(this['trajectory'])[0])
    weirdo = np.unique(weirdo)
    return ds.drop_sel(traj=weirdo)


def id_weirdo_from_ds(ds, dxmax=3, dymax=3):
    """Get the list of traj ID for which lon/lat changes are larger than dxmax/dymax at some point in their trajectory"""
    weirdo = []
    for traj in ds['traj']:
        this = ds.sel(traj=traj)
        if this['lon'].diff('obs').max() > dxmax:
            weirdo.append(np.unique(this['trajectory'])[0])
        if this['lat'].diff('obs').max() > dymax:
            weirdo.append(np.unique(this['trajectory'])[0])
    weirdo = np.unique(weirdo)
    return weirdo


def id_weirdo_from_index(df, dxmax=6, dymax=6):
    """Get the list of traj ID for which lon/lat changes are larger than dxmax/dymax at some point in their trajectory"""
    weirdo = []
    for k, [name, group] in enumerate(df.groupby("traj_id")):
        lon = group['longitude']
        if len(lon)>1 and np.max(np.abs(np.diff(lon))) > dxmax:
            weirdo.append(name)
        lat = group['latitude']
        if len(lat)>1 and np.max(np.abs(np.diff(lat))) > dymax:
            weirdo.append(name)
    weirdo = np.unique(weirdo)
    return weirdo


def clean_this_index(index, weirdWMO, dxmax=3, dymax=3):
    """For each float, we remove data/index after the first float exit out of the domain"""
    from tqdm.notebook import tqdm

    index_list = []
    N = 0
    for wmo in tqdm(weirdWMO):
        this_index = index
        mask = this_index.apply(lambda x: x['traj_id'] == wmo, axis=1)
        this_index = this_index[mask]
        this_index = this_index.reset_index(drop=True)
        for ii, row in this_index.iterrows():
            if ii + 1 < this_index.shape[0]:
                next_row = this_index.loc[ii + 1]
                if np.abs(next_row['latitude'] - row['latitude']) > dymax:
                    # print("traj_id %s going out at %i in latitude" % (wmo, ii))
                    idrop = np.arange(ii + 1, this_index.shape[0])
                    N += len(idrop)
                    this_index = this_index.drop(idrop)
                    pass
                if np.abs(next_row['longitude'] - row['longitude']) > dxmax:
                    # print("traj_id %s going out at %i in longitude" % (wmo, ii))
                    idrop = np.arange(ii + 1, this_index.shape[0])
                    N += len(idrop)
                    this_index = this_index.drop(idrop)
                    pass
        index_list.append(this_index)
    return pd.concat(index_list), N


def remove_weirdo_from_index(data, dxmax=3, dymax=3):
    # Determine which floats are problematic:
    if 'simu' in data:
        weirdWMO = id_weirdo_from_ds(data['simu'], dxmax=dxmax, dymax=dymax)
    else:
        weirdWMO = id_weirdo_from_index(data['index'], dxmax=dxmax, dymax=dymax)
    print("Found %i weird floats, now trimming ..." % len(weirdWMO))

    # Get the index without these problematic WMOs:
    ok_index = data['index']
    mask = ok_index.apply(lambda x: x['traj_id'] not in weirdWMO, axis=1)
    ok_index = ok_index[mask].reset_index(drop=True)

    # Trim the index from profiles of problematic WMOs:
    trimmed_index, Ndrop = clean_this_index(data['index'], weirdWMO, dxmax=dxmax, dymax=dymax)
    print("%i profiles removed" % Ndrop)

    # Check consistency:
    assert (Ndrop + trimmed_index.shape[0] == data['index'].shape[0] - ok_index.shape[0])

    # Final clean-up new index:
    index = pd.concat([ok_index, trimmed_index]).sort_values(by=['traj_id', 'cycle_number']).reset_index(drop=True)

    # Replace with this new index:
    data['index'] = index
    return data


def region2path(OneRegion):
    """Return a matplotlib Path from a regionmask"""
    p = path.Path([(OneRegion.bounds[0], OneRegion.bounds[1]),
                   (OneRegion.bounds[2], OneRegion.bounds[1]),
                   (OneRegion.bounds[2], OneRegion.bounds[3]),
                   (OneRegion.bounds[0], OneRegion.bounds[3])])
    return p


def region2segments(OneRegion):
    """Return dictionary with rectangular faces as matplotlib Path segments"""
    seg = {}
    seg['south'] = path.Path([(OneRegion.bounds[0], OneRegion.bounds[1]),
                              (OneRegion.bounds[2], OneRegion.bounds[1])])
    seg['east'] = path.Path([(OneRegion.bounds[2], OneRegion.bounds[1]),
                             (OneRegion.bounds[2], OneRegion.bounds[3])])
    seg['north'] = path.Path([(OneRegion.bounds[2], OneRegion.bounds[3]),
                              (OneRegion.bounds[0], OneRegion.bounds[3])])
    seg['west'] = path.Path([(OneRegion.bounds[0], OneRegion.bounds[3]),
                             (OneRegion.bounds[0], OneRegion.bounds[1])])
    return seg


def intersectregion(OneRegion, this_df):
    """Given a regionmask, determine which face is intersecting a trajectory segment given by first/last entries of dataframe"""
    segments = region2segments(OneRegion)
    trj = path.Path([(this_df.iloc[0]['longitude'], this_df.iloc[0]['latitude']),
                     (this_df.iloc[-1]['longitude'], this_df.iloc[-1]['latitude'])])
    this_face = None
    for face in segments.keys():
        if segments[face].intersects_path(trj):
            this_face = face
    return this_face


def id_area_float_fluxes(regions, this_index, WMO, domain_name='GSE'):
    from tqdm.notebook import tqdm

    this_index['in_area'] = ''
    this_index['face_in'] = ''
    this_index['face_out'] = ''

    p = region2path(regions[domain_name])

    for wmo in tqdm(WMO):
        # Processing:
        df = this_index[this_index['wmo'] == wmo]
        x, y = df['longitude'].values, df['latitude'].values
        df['in_area'] = p.contains_points(np.array([x, y]).T)
        # Last point before entering the AREA:
        igoes_in = np.argwhere(np.diff(df['in_area'].astype(int)) == 1)[:, 0]
        # Last point before going out of the AREA:
        igoes_out = np.argwhere(np.diff(df['in_area'].astype(int)) == -1)[:, 0]

        segments = region2segments(regions[domain_name])
        face_entry, face_exit = {}, {}
        [face_entry.update({k: 0}) for k in segments.keys()]
        for ii in igoes_in:
            face = intersectregion(regions[domain_name], df.iloc[ii:ii + 2])
            if face:
                # face_entry[face] += 1
                df.iloc[ii, df.columns.get_loc('face_in')] = face
        [face_exit.update({k: 0}) for k in segments.keys()]
        for ii in igoes_out:
            face = intersectregion(regions[domain_name], df.iloc[ii:ii + 2])
            if face:
                # face_exit[face] += 1
                df.iloc[ii, df.columns.get_loc('face_out')] = face

        this_index.loc[df.index] = df
    return this_index


def area_float_flux_counts(regions, index, domain_name='GSE'):
    # Get list of WMO with profiles in the experiment area:
    WMO = np.unique(squeeze_region(regions, domain_name, index)['wmo'])
    # nfloat = len(np.unique(local_index['wmo']))

    # Squeeze index to these floats only:
    index = index[index.apply(lambda x: x['wmo'] in WMO, axis=1)]
    # index = squeeze_region(regions, 'NATL', index)
    index = index.reset_index(drop=True)

    # Analyse trajectories through each faces of the experiment area:
    index = id_area_float_fluxes(regions, index, WMO, domain_name=domain_name)

    # Count fluxes:
    ct_in = index.groupby('face_in')['date'].count()  # Number of float entry by faces
    ct_out = index.groupby('face_out')['date'].count()  # Number of float exist by faces

    if 'east' not in ct_in:
        ct_in['east'] = 0
    if 'west' not in ct_in:
        ct_in['west'] = 0
    if 'south' not in ct_in:
        ct_in['south'] = 0
    if 'north' not in ct_in:
        ct_in['north'] = 0

    if 'east' not in ct_out:
        ct_out['east'] = 0
    if 'west' not in ct_out:
        ct_out['west'] = 0
    if 'south' not in ct_out:
        ct_out['south'] = 0
    if 'north' not in ct_out:
        ct_out['north'] = 0

    # Flux matrix with signs:
    floats_flux = {'east': {'in': -ct_in['east'], 'out': ct_out['east']},
                   'west': {'in': ct_in['west'], 'out': -ct_out['west']},
                   'south': {'in': ct_in['south'], 'out': -ct_out['west']},
                   'north': {'in': -ct_in['north'], 'out': ct_out['north']}}
    # Overall max amplitude of flux range:
    rge = [floats_flux['east']['out'] - floats_flux['east']['in'],
           floats_flux['west']['out'] - floats_flux['west']['in'],
           floats_flux['south']['out'] - floats_flux['south']['in'],
           floats_flux['north']['out'] - floats_flux['north']['in']]
    delta = np.max(np.abs(rge))

    return floats_flux, delta


def make_harrow(ax, face, value, delta=100, max_size=20):
    """Plot an horizontal arrow to represent a float 'flux'"""
    size = np.abs(value) / delta * max_size

    x = np.mean(face.vertices[:, 0])
    ymin = np.min(face.vertices[:, 1])
    dy = np.max(face.vertices[:, 1]) - ymin
    # print(x, ymin, dy)
    if value > 0:
        ha = 'left'
        boxstyle = "rarrow, pad=0.3"
        y0 = ymin + dy * 2 / 3
    else:
        ha = 'right'
        boxstyle = "larrow, pad=0.3"
        y0 = ymin + dy * 1 / 3
    t = ax.text(
        x, y0, "%i" % value, ha=ha, va="center", rotation=0, size=size,
        bbox=dict(boxstyle=boxstyle, fc="cyan", ec="b", lw=1))


def make_varrow(ax, face, value, delta=100, max_size=20):
    """Plot an vertival arrow to represent a float 'flux'"""
    size = np.abs(value) / delta * max_size

    y = np.mean(face.vertices[:, 1])
    xmin = np.min(face.vertices[:, 0])
    dx = np.max(face.vertices[:, 0]) - xmin

    if value > 0:
        va = 'bottom'
        rotation = 90
        x0 = xmin + dx * 2 / 3
    else:
        va = 'top'
        rotation = -90
        x0 = xmin + dx * 1 / 3
    t = ax.text(
        x0, y, "%i" % value, ha='left', va=va, rotation=rotation, size=size,
        bbox=dict(boxstyle="rarrow, pad=0.3", fc="cyan", ec="b", lw=1))


def get_a_run_template(db_file):
    # jobid = os.getenv('PBS_JOBID').split('.')[0]  # eg: 777994.datarmor0

    # Load the database:
    db = load_json(db_file)

    # Load the template for a new run:
    template = db['simulations']['run-template']

    # Get a new entry for this run:
    run = template.copy()
    run['comment'] = ''
    run['runs-on'] = 'datarmor'

    return run



def simdb_comment(run):
    import re
    from subprocess import Popen, PIPE

    def remove_emojis(data):
        emoj = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                          "]+", re.UNICODE)
        return re.sub(emoj, '', data)

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    proc = Popen("simdb --run %s -c" % run, shell=True, close_fds=True, stdin=PIPE, stdout=PIPE)
    out, err = proc.communicate()
    s = out.decode('utf-8').replace("\n", "").replace("   ", "")
    s = ansi_escape.sub('', s)
    s = remove_emojis(s)
    s = s.split("Comment:")[-1].strip()
    return s


def load_one_experiment_ctl_obs(run, grid_resolution=1):
    import pickle
    from virtualargofleet.utilities import set_WMO, simu2index, splitonprofiles, simu2index_par
    import re
    from subprocess import Popen, PIPE

    def load_observations(dict_regions):
        # Load data:
        dict_regions = get_index_for_regions(dict_regions)
        index = dict_regions['NATL']['index']

        # Ensure domain consistency with simulation results:
        index = squeeze_natl(regions, index)

        # Compute profile density
        print('Compute profile density...')
        obs = compute_prof_density(index, grid_resolution)

        length = index['date'].max() - index['date'].min()
        print("Length of the observed time series: %i days [%s - %s]" % (
        length.total_seconds() / 86400, index['date'].min(), index['date'].max()))
        print("Nb of profiles: %i" % index.shape[0])

        return {'index': index, 'prof_density': obs}

    def load_simulation(simu_file, df_plan=None, par=False):
        # Load float configuration
        print('Load float configuration...')
        cfg_file = simu_file.replace(".nc", "_floatcfg.json")
        if os.path.exists(cfg_file):
            cfg = FloatConfiguration(cfg_file)

        # Load simulation raw data:
        print('Load simulation...')
        if os.path.exists(simu_file):
            simu = xr.open_dataset(simu_file)
            # print("Remove weird floats...")
            # simu = remove_weirdo(simu, dxmax=3, dymax=3)
        else:
            simu = None
            print("Can't find: %s" % simu_file)

        # ID correct WMO:
        if df_plan is not None and simu is not None:
            print('Name virtual floats with real WMO...')
            simu = set_WMO(simu, df_plan)

        # Create index of profiles if not on disk:
        print('Get index of profiles...')
        index_file = simu_file.replace(".nc", "_ar_index_prof.txt")
        get_index = simu2index if par else simu2index_par
        if not os.path.exists(index_file):
            if simu is not None:
                print("Computing index ...")
                simu_index = get_index(simu)
                simu_index['file'] = '?'
            else:
                raise ValueError("There is no way to get the index with the simulation or index files !")
        else:
            print("Loading index from file...")
            simu_index = pd.read_csv(index_file, sep=",", index_col=None, header=0, skiprows=8)
            # simu_index = simu_index[['date', 'latitude', 'longitude', 'institution_code', 'file']]
            simu_index['date'] = pd.to_datetime(simu_index['date'], format='%Y%m%d%H%M%S')
            simu_index['wmo'] = simu_index["file"].apply(lambda x: int(x.split("/")[1]))
            simu_index['cycle_number'] = simu_index['file'].apply(lambda x: int(x.split("_")[-1].split(".nc")[0]))
            simu_index['traj_id'] = simu_index['wmo'] - 9000000
            simu_index = simu_index[['date', 'latitude', 'longitude', 'wmo', 'cycle_number', 'traj_id']]

            # print("Remove profiles from weird floats...")
            # keep = np.unique(simu['traj'])
            # keep = simu_index.apply(lambda x: x['traj_id'] in keep, axis=1)
            # simu_index = simu_index[keep]

        # Remove profiles from the index of those floats that went suddenly from one side of the domain to the other
        if os.path.exists(simu_file):
            data = remove_weirdo_from_index({'simu': simu, 'index': simu_index}, dxmax=3, dymax=3)
        else:
            data = remove_weirdo_from_index({'index': simu_index}, dxmax=9, dymax=9)
        simu_index = data['index']

        # Compute profile density
        print('Compute profile density...')
        obs = compute_prof_density(simu_index, grid_resolution)

        simu_length = simu_index['date'].max() - simu_index['date'].min()
        print("Length of the simulation: %i days [%s - %s]" % (
        simu_length.total_seconds() / 86400, simu_index['date'].min(), simu_index['date'].max()))
        print("Nb of profiles: %i" % simu_index.shape[0])
        print("\n")
        return {'simu': simu, 'index': simu_index, 'prof_density': obs, 'float_cfg': cfg}

    def read_traj_filename(run):

        def remove_emojis(data):
            emoj = re.compile("["
                              u"\U0001F600-\U0001F64F"  # emoticons
                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                              u"\U0001F680-\U0001F6FF"  # transport & map symbols
                              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              u"\U00002500-\U00002BEF"  # chinese char
                              u"\U00002702-\U000027B0"
                              u"\U00002702-\U000027B0"
                              u"\U000024C2-\U0001F251"
                              u"\U0001f926-\U0001f937"
                              u"\U00010000-\U0010ffff"
                              u"\u2640-\u2642"
                              u"\u2600-\u2B55"
                              u"\u200d"
                              u"\u23cf"
                              u"\u23e9"
                              u"\u231a"
                              u"\ufe0f"  # dingbats
                              u"\u3030"
                              "]+", re.UNICODE)
            return re.sub(emoj, '', data)

        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        proc = Popen("simdb --run %s -o --local" % run, shell=True, close_fds=True, stdin=PIPE, stdout=PIPE)
        out, err = proc.communicate()
        s = out.decode('utf-8').replace("\n", "").replace("   ", "")
        s = ansi_escape.sub('', s)
        s = remove_emojis(s)
        s = s.split("Output:")[-1].strip()
        s = s.split("Trajectories: ")[-1].split("Traj.")[0].strip()
        return s

    def run2region(run):
        dr = [
            {'NATL': ['613257']},
            {'GSE': ['614790', '653526', '1009213', '1132758']},
            {'GSEext': ['1009214', '1010801', '1009216']},
            {'ArgoWBC': ['691822', '711242', '1132757']},
        ]
        for region in dr:
            for name in region.keys():
                if run in region[name]:
                    return name
        return None

    control = "613257"
    e_name = "Experiment (#%s) vs Control (#%s)" % (run, control)
    pkfile = os.path.join(EUROARGODEV, "VirtualFleet_GulfStream", "local_work", "data", "%s.pk" % e_name)
    if os.path.exists(pkfile):
        results = pickle.load(open(pkfile, 'rb'))
    else:
        results = {}
        # Observations:
        print('Loading Observations...')
        results['obs'] = {'data': load_observations(dict_regions), 'name': '2008-2018 Observations', 'region': 'NATL'}

        # Control:
        print("\nLoading Control...")
        results['ctl'] = {'data': load_simulation(read_traj_filename(control)),
                          'name': 'Control (#%s)' % control,
                          'region': run2region(control)}
        # Experiment:
        print("\nLoading Experiment...")
        results['exp'] = {'data': load_simulation(read_traj_filename(run)),
                          'name': 'Experiment (#%s)' % run,
                          'region': run2region(run)}

        # Delete profiles out of the domain of statistical analysis, and comparison with observations
        for run in ['ctl', 'exp']:
            results[run]['data']['index'] = squeeze_natl(regions, results[run]['data']['index'])

        # Determine the optimal maximum cycle number to deal with the same nb of profiles than in obs.
        Nobs = results['obs']['data']['index'].shape[0]
        Nrange = np.arange(100,200,1)
        Ncount = np.empty_like(Nrange)
        for ii, N in enumerate(Nrange):
            df = squeeze_max_cyc_number(N, results['ctl']['data']['index'])
            Ncount[ii] = df.shape[0]

        Nmax_optim = Nrange[np.argmin(np.abs(Ncount - Nobs))+1]
        for run in ['ctl', 'exp']:
            results[run]['data']['index'] = squeeze_max_cyc_number(Nmax_optim, results[run]['data']['index'])
        results['Nmax_optim'] = Nmax_optim

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5), dpi=90, facecolor='w', edgecolor='k')
        ax.plot(Nrange, Ncount, '.')
        ax.vlines(Nmax_optim, np.min(Ncount), np.max(Ncount), 'r')
        ax.hlines(Nobs, np.min(Nrange), np.max(Nrange), 'k')
        ax.text(Nmax_optim, np.min(Ncount), 'Optimum = %i' % Nmax_optim, color='r')
        ax.text(Nrange[0], Nobs, 'Real number of profiles\n(%i)' % Nobs, verticalalignment='bottom')
        ax.set_xlabel('Maximum cycle number in the simulation')
        ax.set_ylabel('Simulated number of profiles')
        ax.set_title("Optimal maximum number of profiles to use for %s\n to match the number of profiles in %s" % (results['ctl']['name'], results['obs']['name']))
        ax.grid()
        figfile = os.path.join(EUROARGODEV, "VirtualFleet_GulfStream", "local_work", "img",
                               "Optimal-%s-N%i.png" % (results['ctl']['name'].replace(" ","-"), Nmax_optim))
        plt.savefig(figfile, bbox_inches='tight', pad_inches=0.1)

        # Compute profile density:
        for run in ['ctl', 'exp']:
            results[run]['data']['prof_density'] = compute_prof_density(results[run]['data']['index'], grid_resolution)

        # Compute changes in density for this experiment:
        results['ano'] = {'name': "%s vs %s" % (results['exp']['name'], results['ctl']['name']),
                          'region': results['exp']['region'],
                          'data': {
                              'prof_density': results['exp']['data']['prof_density'] - results['ctl']['data']['prof_density']}}

        # Float "flux" through each faces of the experiment domain:
        print("\n Float 'flux' through each faces of the experiment domain...")
        for run in ['obs', 'ctl', 'exp']:
            floats_flux, delta = area_float_flux_counts(regions,
                                                        results[run]['data']['index'],
                                                        domain_name=results['exp']['region'])
            results[run]['data']['fluxes'] = {'count': floats_flux, 'delta': delta}

        # Compute change in fluxes between experiment and control:
        floats_flux = {'east': {'in': 0, 'out': 0},
                       'west': {'in': 0, 'out': 0},
                       'south': {'in': 0, 'out': 0},
                       'north': {'in': 0, 'out': 0}}
        for face in ['east', 'west', 'south', 'north']:
            for direction in ['in', 'out']:
                floats_flux[face][direction] = results['exp']['data']['fluxes']['count'][face][direction] \
                                               - results['ctl']['data']['fluxes']['count'][face][direction]
        rge = [floats_flux['east']['out'] - floats_flux['east']['in'],
               floats_flux['west']['out'] - floats_flux['west']['in'],
               floats_flux['south']['out'] - floats_flux['south']['in'],
               floats_flux['north']['out'] - floats_flux['north']['in']]
        delta = np.max(np.abs(rge))
        results['ano']['data']['fluxes'] = {'count': floats_flux, 'delta': delta}

        # Save post-processed data for later re-use:
        pkfile = "data/%s.pk" % results['ano']['name']
        pickle.dump(results, open(pkfile, 'wb'))

    return results