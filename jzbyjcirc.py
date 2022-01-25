import numpy


def Hz(z, H0, omegaM, omegaK, omegaLb):

    Hz2 = (H0**2) * (omegaM * ((1 + z)**3) + omegaK * ((1 + z)**2) + omegaLb)
    Hz = Hz2**(0.5)
    return Hz


def angmom_getdirdist(cent1, cent2, boxlen1):

    permax = boxlen1 / 2.0
    Rdiff = cent1 - cent2
    Rmask = 1 - (numpy.abs(Rdiff / permax)).astype(numpy.int32)
    Roffset = numpy.ma.masked_array(
        (-1) * boxlen1 * ((Rdiff / permax).astype(numpy.int32)), Rmask)
    Rdiff = numpy.array(Rdiff + Roffset.filled(0)).copy()
    Rmeas = pow(Rdiff, 2.0)
    Rmeas3d = Rmeas[:, 0] + Rmeas[:, 1] + Rmeas[:, 2]
    Rmeas3d = pow(Rmeas3d, 1.0 / 2.0)
    Rmeas2d = Rmeas[:, 0] + Rmeas[:, 1]
    Rmeas2d = pow(Rmeas2d, 1.0 / 2.0)
    return Rdiff, Rmeas3d, Rmeas2d


""" Positions : Comoving (h^{1}kpc), Velocities : Code Units (km/s), Returns : Jz in Comoving (h^{-1}kpc km/s)"""


def getjz(sg_posi, sg_strhmi, star_pos, star_mass, star_vel, boxlen, redshift, Hz1, hrfac=None):
    z = redshift
    if hrfac is None:
        rmax = (boxlen / 2.0)**(pow(3, 1.0 / 2.0))
    else:
        rmax = hrfac * (sg_strhmi)
    fstar_vel = star_vel * (pow(1.0 / (1.0 + z), 0.5))
    star_rdiff, star_r3d, star_r2d = angmom_getdirdist(
        star_pos, sg_posi, boxlen)
    fvel = fstar_vel + numpy.float64(star_rdiff) * (Hz1 / (1000.0 * (1.0 + z)))
    velmp = numpy.array(numpy.matrix(
        numpy.float64(star_mass)).T) * (numpy.float64(fvel))
    star_velmean = [velmp[:, 0].sum(), velmp[:, 1].sum(
    ), velmp[:, 2].sum()] / numpy.float64(star_mass).sum()
    star_vdiff = (fvel - star_velmean)
    chk_1hm = (star_r3d <= (1.0 * sg_strhmi))
    chk_2hm = (star_r3d <= (2.0 * sg_strhmi))
    star_mass_2hm = star_mass[chk_2hm]
    chk_10hm = (star_r3d <= (rmax))
    star_rdiff_10hm = star_rdiff[chk_10hm]
    star_vdiff_10hm = star_vdiff[chk_10hm]
    star_mass_10hm = star_mass[chk_10hm]
    jstar_10hm = numpy.array(numpy.matrix(star_mass_10hm).T) * \
        (numpy.cross(star_rdiff_10hm, star_vdiff_10hm))
    jtot_10hm = numpy.array([jstar_10hm[:, 0].sum(), jstar_10hm[:, 1].sum(
    ), jstar_10hm[:, 2].sum()]) / (star_mass_10hm.sum())
    jtot_10hm_mag = pow((jtot_10hm**2).sum(), 0.5)
    jdir_10hm = jtot_10hm / pow(((jtot_10hm)**2).sum(), 0.5)
    jz_10hm = numpy.dot(numpy.cross(
        star_rdiff_10hm, star_vdiff_10hm), jdir_10hm)
    return star_mass_2hm, star_mass_10hm, jstar_10hm, jtot_10hm, jtot_10hm_mag, jdir_10hm, jz_10hm, chk_10hm, chk_1hm


def getjcirc(sg_posi, sg_strhmi, gas_pos, gas_mass, dm_pos, dm_mass, star_pos, star_mass, star_vel, bh_pos, bh_mass, sg_lentypei, boxlen, redshift, Hz1, hrfac=None, wind_pos=None, wind_mass=None):
    z = redshift
    if hrfac is None:
        rmax = (boxlen / 2.0)**(pow(3, 1.0 / 2.0))
    else:
        rmax = hrfac * (sg_strhmi)

    star_rdiff, star_r3d, star_r2d = angmom_getdirdist(
        star_pos, sg_posi, boxlen)
    chk_10hm = (star_r3d <= (rmax))
    star_r3d_10hm = star_r3d[chk_10hm]
    star_mass_10hm = star_mass[chk_10hm]

    star_pivsort = numpy.argsort(star_r3d_10hm)
    star_r3dsort = star_r3d_10hm[star_pivsort]
    star_msort = star_mass_10hm[star_pivsort]

    lentype_id = sg_lentypei
    in_rad0 = numpy.zeros(lentype_id[0])
    in_rad1 = numpy.zeros(lentype_id[1])
    in_rad4 = numpy.zeros(lentype_id[4])
    in_rad5 = numpy.zeros(lentype_id[5])

    if wind_pos is None:
        in_rad4a = numpy.zeros(0)
        wind_mass = numpy.zeros(0)
    else:
        in_rad4a = numpy.zeros(len(wind_pos))
        wind_mass = numpy.zeros(len(wind_pos))
        if (len(wind_pos) > 0):
            in_dpos4a, in_rad4a, in_rad2d4a = angmom_getdirdist(
                wind_pos, sg_posi, boxlen)
    if (lentype_id[0] > 0):
        in_dpos0, in_rad0, in_rad2d0 = angmom_getdirdist(
            gas_pos, sg_posi, boxlen)
    if (lentype_id[1] > 0):
        in_dpos1, in_rad1, in_rad2d1 = angmom_getdirdist(
            dm_pos, sg_posi, boxlen)
    if (lentype_id[4] > 0):
        in_dpos4, in_rad4, in_rad2d4 = angmom_getdirdist(
            star_pos, sg_posi, boxlen)
    if (lentype_id[5] > 0):
        in_dpos5, in_rad5, in_rad2d5 = angmom_getdirdist(
            bh_pos, sg_posi, boxlen)

    in_radall = numpy.zeros(len(in_rad0) + len(in_rad1) +
                            len(in_rad4) + len(in_rad4a) + len(in_rad5))
    in_massall = numpy.zeros(
        len(in_rad0) + len(in_rad1) + len(in_rad4) + len(in_rad4a) + len(in_rad5))
    in_radall[0:len(in_rad0)] = in_rad0[:]
    in_radall[len(in_rad0): len(in_rad0) + len(in_rad1)] = in_rad1[:]
    in_radall[len(in_rad0) + len(in_rad1): len(in_rad0) +
              len(in_rad1) + len(in_rad4)] = in_rad4[:]
    in_radall[len(in_rad0) + len(in_rad1) + len(in_rad4): len(in_rad0) +
              len(in_rad1) + len(in_rad4) + len(in_rad5)] = in_rad5[:]
    in_radall[len(in_rad0) + len(in_rad1) + len(in_rad4) + len(in_rad5): len(in_rad0) +
              len(in_rad1) + len(in_rad4) + len(in_rad5) + len(in_rad4a)] = in_rad4a[:]
    if (lentype_id[0] > 0):
        in_massall[0:len(in_rad0)] = gas_mass[:]
    if (lentype_id[1] > 0):
        in_massall[len(in_rad0): len(in_rad0) + len(in_rad1)] = dm_mass[:]
    if (lentype_id[4] > 0):
        in_massall[len(in_rad0) + len(in_rad1): len(in_rad0) +
                   len(in_rad1) + len(in_rad4)] = star_mass[:]
    if (lentype_id[5] > 0):
        in_massall[len(in_rad0) + len(in_rad1) + len(in_rad4): len(in_rad0) +
                   len(in_rad1) + len(in_rad4) + len(in_rad5)] = bh_mass[:]
    in_massall[len(in_rad0) + len(in_rad1) + len(in_rad4) + len(in_rad5): len(in_rad0) +
               len(in_rad1) + len(in_rad4) + len(in_rad5) + len(in_rad4a)] = wind_mass[:]
    in_pivall = numpy.argsort(in_radall)
    in_radall_sort = in_radall[in_pivall]
    in_massall_sort = in_massall[in_pivall]
    in_massall_sort_cumsum = numpy.cumsum(numpy.float64(in_massall_sort))
    piv_din = numpy.searchsorted(in_radall_sort, star_r3dsort)
    in_massall_fin = in_massall_sort_cumsum[piv_din]
    Gconst = 6.67384 * (10**(-11))

    star_r3dsortmeter = numpy.float64(
        (star_r3dsort) * (10**3) * (3.08567758 * (10**16)))
    star_r3dpivm = star_r3dsortmeter
    fmassin_pivkg = numpy.float64(
        in_massall_fin * (10**10) * (1.9891 * (10**30)))
    vcirc_piv = pow(Gconst * (fmassin_pivkg) / (star_r3dpivm), 0.5) / 1000.0
    jcircsort_10hm = star_r3dsort * vcirc_piv

    return vcirc_piv, star_pivsort, jcircsort_10hm


def dtot_jzbyjcirc(sg_posi, sg_strhmi, gas_pos, gas_mass, dm_pos, dm_mass, star_pos, star_mass, star_vel, bh_pos, bh_mass, sg_lentypei, boxlen, redshift, Hz1, h):
    star_mass_2hm, star_mass_10hm, jstar_10hm, jtot_10hm, jtot_10hm_mag, jdir_10hm, jz_10hm, chk_10hm, chk_1hm = getjz(
        sg_posi, sg_strhmi, star_pos, star_mass, star_vel, boxlen, redshift, Hz1, hrfac=1)
    vcirc_piv, star_pivsort, jcircsort_10hm = getjcirc(sg_posi, sg_strhmi, gas_pos, gas_mass, dm_pos, dm_mass, star_pos,
                                                       star_mass, star_vel, bh_pos, bh_mass, sg_lentypei, boxlen, redshift, Hz1, hrfac=1, wind_pos=None, wind_mass=None)
    jcirc_10hm = numpy.zeros(len(jz_10hm))
    jcirc_10hm[star_pivsort] = jcircsort_10hm
    ecirc_10hm = (jz_10hm / jcirc_10hm) * (pow(1.0 / (1.0 + redshift), 0.5))
    dt = numpy.float32(len(ecirc_10hm[ecirc_10hm > 0.7])) / len(ecirc_10hm)
    jtot_10hmmag = jtot_10hmmag / ((1.0 + z) * h)
    return dt, jtot_10hm_mag, jdir_10hm


def getspin(sg_posi, sg_massi, sg_veli, sg_poti, dm_pos, dm_mass, dm_vel, boxlen, redshift, Hz1, h):
    dm_mass_2hm, dm_mass_10hm, jdm_10hm, jtot_10hm, jtot_10hm_mag, jdir_10hm, jz_10hm, chk_10hm, chk_1hm = getjz(
        sg_posi, sg_r200i, dm_pos, dm_mass, dm_vel, boxlen, redshift, Hz1, hrfac=None)
    dm_vel = dm_vel * (pow(1.0 / (1.0 + redshift), 0.5))
    sg_veli = sg_veli * (pow(1.0 / (1.0 + redshift), 0.5))
    sg_massi = sg_massi * (1.98855e30)
    dm_mass = dm_mass * (1.98855e30)
    Etothalf = pow((sg_massi * sg_poti + 0.5 *
                    ((dm_mass * ((dm_vel - sg_veli)**2.0)).sum())) * 1e6, 0.5)
    jtot_10hm_mag = jtot_10hm_mag * (3.086e19) * (1.98855e30 * 1e3)
    G = 6.67408 * 1e-11  # in m^3kg^-1s^-2, 1kpc : 3.086e19 meters,
    dmspin = jtot_10hm_mag * Etothalf / (G * (pow(sg_massi, 2.5)))
    return dmspin
