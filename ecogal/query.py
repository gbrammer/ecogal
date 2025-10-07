"""
Queries to the ALMA archive with astroquery
"""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from astroquery.alma.core import Alma
from astropy import coordinates
from astropy import units as u
import astropy.io.fits as pyfits
import astropy.wcs as pywcs

from grizli import utils
import ecogal.pbcor

alma = Alma()
alma.archive_url = "https://almascience.eso.org"

archive_mirrors = [
    "https://almascience.eso.org",
    "https://almascience.org",  # ?
    "https://almascience.nrao.edu",  # ?
]

BASEPATH = "/Users/gbrammer/Research/JWST/Projects/AlmaECOCAL/Raw"
if not os.path.exists(BASEPATH):
    BASEPATH = os.getcwd()
    print(f"BASEPATH: {BASEPATH}")

def query_position(
    coord="53.16557 -27.76991",
    radius=0.1 * u.arcmin,
    load_existing_filelist=True,
    download=True,
    skip_downloaded=True,
    payload={},
    skip_mosaics=False,
):
    """ """
    result = utils.GTable(
        alma.query_region(coord, radius=radius, payload=payload)
    )
    for c in result.colnames:
        if result[c][0] in ["T", "F"]:
            result[c] = result[c] == "T"

    print(f"result: {len(result)} rows")

    if skip_mosaics:
        print(f'remove: {result["is_mosaic"].sum()} mosaics')

        result = result[~result["is_mosaic"]]

    uids = utils.Unique(result["member_ous_uid"], verbose=False)

    plt.scatter(
        result["s_ra"],
        result["s_dec"],
        alpha=0.1,
    )
    ax = plt.gca()

    for row in result:
        sr = utils.SRegion(row["s_region"].split(" Not")[0])
        is_mosaic = row["is_mosaic"] == "T"

        # if is_mosaic:
        #     continue

        color = plt.cm.jet((int(row["band_list"]) - 1) / 9)

        sr.add_patch_to_axis(
            ax,
            fc=color if is_mosaic else "None",
            ec=color,
            alpha=0.1,
            zorder=10 - is_mosaic,
        )

    utils.LOGFILE = os.path.join(BASEPATH, "alma_download_log.txt")

    result["ustr"] = [
        uid.split("//")[-1].replace("/", "_")
        for uid in result["member_ous_uid"]
    ]

    if not download:
        return result

    print(f"N = {uids.N} unique uids")

    for ui, uid in enumerate(uids.values):
        ustr = uid.split("//")[-1].replace("/", "_")
        savedir = os.path.join(BASEPATH, ustr)

        if os.path.exists(savedir) & skip_downloaded:
            continue

        ustr_table = os.path.join(BASEPATH, f"{ustr}_files.csv")
        if os.path.exists(ustr_table) & load_existing_filelist:
            uid_url_table = utils.read_catalog(ustr_table)
            write_filelist = False
        else:
            uid_url_table = alma.get_data_info(uid, expand_tarfiles=True)
            write_filelist = True

        uid_url_table["ustr"] = ustr

        has_pb = np.zeros(len(uid_url_table), dtype=bool)
        # test = has_pb | True

        for i, f in enumerate(uid_url_table["access_url"]):
            has_pb[i] = (
                ("cube" not in f)
                & ("sci.spw" in f)
                & ("fits" in f)
                & (".pb" in f)
            )
            has_pb[i] &= "tt1.pb" not in f
            nsub = f.split("spw")[-1].split(".12m")[0].count("_")
            has_pb[i] &= nsub > 2

            has_pb[i] |= "final_cont_image" in f
            has_pb[i] &= ("_bp.spw" not in f) & ("_ph.spw" not in f)

            if not has_pb[i]:
                has_pb[i] |= ("cont.flux.fits.gz" in f) | (
                    "cont.image.pbcor.fits" in f
                )
                has_pb[i] |= "cont.pbcorr" in f
                has_pb[i] |= "cont.I.pb" in f

            has_pb[i] &= ("_bp.spw" not in f) & ("_ph.spw" not in f)

        uid_url_table["has_pb"] = has_pb
        if write_filelist:
            uid_url_table.write(ustr_table, overwrite=True)

        msg = f"\n\n({ui}/{uids.N})  {ustr}: {has_pb.sum()} PB files\n\n"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        if has_pb.sum() > 0:
            do_download = has_pb & True
            for j in np.where(has_pb)[0]:
                do_download[j] &= not os.path.exists(
                    os.path.join(
                        savedir,
                        os.path.basename(uid_url_table["access_url"][j]),
                    )
                )

            if not os.path.exists(savedir):
                os.makedirs(savedir)

            filelist = alma.download_files(
                uid_url_table["access_url"][has_pb & do_download],
                skip_unauthorized=True,
                cache=False,
                savedir=savedir,
            )

    keys = ["{ustr} {target_name}".format(**row) for row in result]
    unk = utils.Unique(keys, verbose=False)
    print(f"N = {unk.N} unique UID + target_name keys")

    res = result[unk.unique_index()]
    pbcor = []
    pb = []
    hrows = []

    ari = []

    for row in res:
        files = glob.glob(
            "{BASEPATH}/{ustr}/*pbco*.fits".format(BASEPATH=BASEPATH, **row)
        )
        if len(files) > 1:
            files = glob.glob(
                "{BASEPATH}/{ustr}/*{target_name}*pbco*.fits".format(
                    BASEPATH=BASEPATH, **row
                )
            )

        files.sort()
        if len(files) > 0:
            print(files)

            ari_i = "N/A"

            for i, f in enumerate(files):
                if "ari_l" in f:
                    ari_i = files.pop(i)
                    if len(files) == 0:
                        files = [ari_i]
                    break

            ari.append(ari_i)

            pbcor.append(os.path.basename(files[0]))
            with pyfits.open(files[0]) as im:
                hrow = {}
                for k in [
                    "NAXIS1",
                    "NAXIS2",
                    "CRVAL1",
                    "CRVAL2",
                    "CDELT1",
                    "CDELT2",
                    "BMAJ",
                    "BMIN",
                    "BPA",
                ]:
                    hrow[k.lower()] = im[0].header[k]

                hrows.append(hrow)
        else:
            pbcor.append("N/A")
            hrows.append({})
            ari.append("N/A")

        pb_names = [
            pbcor[-1].replace("pbcor.fits", "pb.fits"),
            pbcor[-1].replace("pbcor.fits", "pb.fits.gz"),
            pbcor[-1].replace("pbcor.fits", "flux.fits"),
            pbcor[-1].replace("pbcor.fits", "flux.fits.gz"),
            pbcor[-1].replace("pbcorr.fits", "flux.fits"),
            pbcor[-1].replace("pbcorr.fits", "flux.fits.gz"),
            pbcor[-1].replace("image.pbcor.fits", "pb.fits.gz"),
            pbcor[-1].replace("tt0.pbcor", "pb.tt0"),
        ]

        pb_i = "N/A"
        for pbn in pb_names:
            if os.path.exists(os.path.join(BASEPATH, row["ustr"], pbn)) & (
                pbn != pbcor[-1]
            ):
                pb_i = pbn

        pb.append(pb_i)

    htab = utils.GTable(hrows)
    for c in htab.colnames:
        res[c] = htab[c]

    res["bmaj_arcsec"] = (res["bmaj"] * u.degree).to(u.arcsec)
    res["fov_arcmin2"] = [
        utils.SRegion(sreg.split(" Not")[0]).sky_area()[0].value
        for sreg in res["s_region"]
    ]

    res["pbcor"] = pbcor
    res["pbcor_ari"] = ari
    res["pb"] = pb

    so = np.argsort(res["t_min"])
    res = res[so]

    for c in res.colnames:
        if res[c][0] in ["T", "F"]:
            res[c] = res[c] == "T"

    res.write(
        f"{BASEPATH}/alma_query_{coord.replace(' ','_')}.csv", overwrite=True
    )

    for c in res.colnames:
        if c.startswith("t_"):
            res[c].format = ".1f"

    res["bmaj_arcsec"].format = ".2f"
    res["fov_arcmin2"].format = ".2f"

    return res


def load_ecogal(file, file_pb):
    """ """

    meta = alma_file_metadata(file)
    eco = ecogal.pbcor.EcogalFile(file_alma=file, file_pb=file_pb, meta=meta)

    scale_err = utils.nmad(eco.data_sn[np.isfinite(eco.data_sn)])

    eco.data_sn /= scale_err
    eco.meta["noise_tot"] *= scale_err
    eco.meta["noise_fit"] *= scale_err
    eco.meta["scale_noise_keywords"] = scale_err

    return eco


def show_all_sources(
    file,
    file_pb,
    spurious_count=0.02,
    threshold=-1,
    nmax=10,
    npercentile=0.01,
    **kwargs,
):
    """ """
    from scipy.stats import Normal

    meta = alma_file_metadata(file)
    eco = load_ecogal(file, file_pb)

    if threshold < 0:
        spurious_threshold = -Normal().icdf(spurious_count / eco.nbeams * 2)
        print(f"Threshold for spurious sources: {spurious_threshold}")
        threshold = spurious_threshold

    props = eco.blind_detection(threshold=threshold)

    so = np.argsort(props["sn"])[::-1]

    # clip = np.maximum(props["negative_snmax"], props["spurious_threshold"])
    # nmax = np.minimum((props["sn"] > clip).sum(), nmax) + extra

    sn_clip = props["sn"] > props["negative_snmax"]  # [so]
    sn_clip &= props["spurious_count"] < spurious_count  # [so]

    if props["negative_snmax"][0] < 0:
        npercentile *= -1

    if npercentile > 0:
        perc_clip = props["sn"] > -np.nanpercentile(
            eco.data_sn[np.isfinite(eco.data_sn)], npercentile
        )
        perc_clip &= props["area"] > 2
        sn_clip |= perc_clip

    # if 1:
    #     sn_clip = (props["spurious_count"] < spurious_count)
    #     sn_clip &= (props["area"] > 1)

    props["valid"] = sn_clip  # [so]

    figs = []
    for row in props[so[sn_clip[so]][:nmax]]:
        print(row["ra"], row["dec"], row["sn"])
        fig_ = eco.cutout_with_thumb(row["ra"], row["dec"], **kwargs)
        figs.append(fig_)

    show_center = (sn_clip.sum() == 0) | (not eco.meta["is_mosaic"])
    show_center |= len(props) == 0

    if sn_clip.sum() > 0:
        show_center &= props["dx_beam"][sn_clip].min() > 2

    show_center &= not eco.meta["is_mosaic"]

    if show_center:
        bsize = np.sqrt(eco.meta["bmaj"] ** 2 + eco.meta["bmin"] ** 2) * 60
        nbeams = eco.footprint.sky_area(unit="arcmin2")[0].value / (
            np.pi * bsize**2
        )
        print("Empty catalog, show center")
        fig_ = eco.cutout_with_thumb(
            eco.meta["ra_center"], eco.meta["dec_center"], **kwargs
        )
        figs.append(fig_)

    return eco, props[so], figs


def empty():

    # res = utils.read_catalog("alma_query_3.588333^I-30.397250.csv")
    qfiles = glob.glob("*query*csv")
    k = -1

    k += 1
    res = utils.read_catalog(qfiles[k])

    res = res[(res["pbcor"] != "N/A") & (res["pb"] != "N/A")]
    j = -1
    threshold = 4

    j += 1
    row = res[j]
    file = os.path.join(row["ustr"], row["pbcor"])
    file_pb = os.path.join(row["ustr"], row["pb"])

    eco, props, figs = show_all_sources(file, file_pb, threshold=threshold)

    j += 1
    row = res[j]
    print(
        row[
            "proposal_id",
            "ustr",
            "pbcor",
            "band_list",
            "pi_name",
            "t_exptime",
            "t_min",
            "bmaj_arcsec",
            "data_rights",
            "is_mosaic",
            "fov_arcmin2",
        ]
    )

    file = os.path.join(row["ustr"], row["pbcor"])
    file_pb = os.path.join(row["ustr"], row["pb"])

    eco, props, figs = ecogal.query.show_all_sources(
        file, file_pb, threshold=threshold, npercentile=-1.0e-3
    )
    props[
        "id",
        "ra",
        "dec",
        "smax",
        "area",
        "sn",
        "negative_snmax",
        "negative_sn",
        "spurious_count",
        "nbeams",
        "dx_pb",
        "dx_beam",
        "valid",
    ]


def alma_file_metadata(file):
    """ """
    h = pyfits.getheader(file)

    uid = os.path.basename(file)[len("member.uid___") :].split(".")[0]
    meta = get_alma_metadata(
        uid=uid,
        source_name=(
            h["OBJECT"] if h["OBJECT"] else h["FIELD"].replace('"', "")
        ),
        # payload={"pi_userid": h["OBSERVER"]}
    )

    for k in ["BMAJ", "BMIN", "BPA"]:
        if k in h:
            meta[k.lower()] = h[k]

    return meta


def alma_uid_query(
    uid="A001_X87d_Xbc4", source_name="S2COSMOS.850.103", payload={}, **kwargs
):
    """ """
    uid_key = "uid://" + uid.replace("_", "/")
    result = alma.query(
        payload=dict(
            # member_ous_uid=uid_key,
            source_name_alma=source_name,
            **payload,
        )
    )

    result["match_uid"] = result["member_ous_uid"] == uid_key
    return result


def get_alma_metadata(
    uid="A001_X87d_Xbc4", source_name="S2COSMOS.850.103", payload={}, **kwargs
):
    """
    Query archive
    """

    # match = result["member_ous_uid"] == uid_key
    result = alma_uid_query(
        uid=uid, source_name=source_name, payload=payload, **kwargs
    )
    match = result["match_uid"]

    if match.sum() == 0:
        return dict(result)
    else:
        meta = dict(result[match][0])
        meta["ra_center"] = meta["s_ra"]
        meta["dec_center"] = meta["s_dec"]

        meta["noise_tot"] = meta["sensitivity_10kms"] / 1000.0
        meta["noise_fit"] = meta["sensitivity_10kms"] / 1000.0
        meta["band"] = int(meta["band_list"])

        sr = utils.SRegion(meta["s_region"].split(" Not")[0])

        # meta["FoV_sigma"] = np.sqrt(sr.sky_area(unit="arcsec2")[0]).value / np.pi / 2.0

        meta["wavelength"] = (3.0e8 / (meta["frequency"] * 1.0e9)) * u.m
        meta["primary_beam_fwhm"] = (
            1.13 * meta["wavelength"] / (12 * u.m) * u.radian
        ).to(u.arcmin)
        meta["FoV_sigma"] = (
            (meta["primary_beam_fwhm"] / 2.35).to(u.arcsec).value
        )

        meta["footprint"] = sr.polystr(precision=5)[0]

        for k in meta:
            if meta[k] in ["T", "F"]:
                meta[k] = meta[k] == "T"

        return meta
