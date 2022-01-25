import numpy
from jzbyjcirc import *
from argparse import ArgumentParser

ap = ArgumentParser("build subhalo catalogue of BlueTides")
ap.add_argument("pig", help="directory of FOF PIG files, e.g. PIG_086")
ap.add_argument("N", help="number of FOF groups to scan e.g. 100", type=int)

ns = ap.parse_args()


def distp(center, pos, boxsize):
    diff = (pos - center)
    diff = numpy.abs(diff)
    diff[diff > 0.5 * boxsize] -= boxsize
    return numpy.einsum('ij,ij->i', diff, diff) ** 0.5


def dist3p(center, pos, boxsize):
    diff = (pos - center)
    diff[diff > 0.5 * boxsize] -= boxsize
    diff[diff < -0.5 * boxsize] -= boxsize
    return diff


def cmp(pos, mass, boxsize):
    diff = (pos - pos[0])
    diff[diff > 0.5 * boxsize] -= boxsize
    diff[diff < -0.5 * boxsize] += boxsize
    cm = numpy.sum(diff * mass, dtype='f8', axis=0) / numpy.sum(mass)
    return (cm + pos[0]) % boxsize


def measure_m200(center, pos, mass, omegam, boxsize):
    rbins = numpy.linspace(0, 1000, 100)

    r = distp(center, pos, boxsize)
    dig = numpy.digitize(r, rbins)
    N = numpy.bincount(dig, weights=mass, minlength=len(rbins) + 1)
    r = rbins
    mcum = N.cumsum()[1:]
    rho = mcum / (4 * numpy.pi / 3 * (rbins)**3)
    rho_c = 27.75 / 1e9 * 200
    rho[0] = numpy.inf
    r200 = numpy.interp(-rho_c, -rho, r)
    m200 = numpy.interp(r200, r, mcum)
    return r200, m200, rho_c, r, rho


def create_subhalos(bf, fofid):
    Nsubhalo = bf['FOFGroups/LengthByType'][fofid][5]
    result = numpy.empty(Nsubhalo,
                         dtype=[
                             ('StarFormationRate',  'f8'),
                             ('Position',  ('f8', 3)),
                             ('DistanceToCenter',  ('f8')),
                             ('BlackholePosition',  ('f8', 3)),
                             ('BlackholeMass',  ('f8')),
                             ('TotalBlackholeAccretionRate',  ('f8')),
                             ('TotalBlackholeMass',  ('f8')),
                             ('NBlackhole',  ('i4')),
                             ('CentralBlackholeID',  ('u8')),
                             ('FOFHaloID', 'i4'),
                             ('BlackholeID', 'u8'),
                             ('Mass200',  'f4'),
                             ('R200',  'f4'),
                             ('DtoT',  'f4'),
                             ('MassByType200',  ('f4', 6)),
                             ('StellarSpecifcAngMom', 'f4'),
                             ('StellarSpinDirection', ('f4', 3)),
                             ('StellarAxisRatios', ('f4', 2)),
                             ('StellarShapeDirections', ('f4', (3, 3))),
                             ('ReducedStellarAxisRatios', ('f4', 2)),
                             ('ReducedStellarShapeDirections', ('f4', (3, 3)))
                         ])
    boxsize = bf['header'].attrs['BoxSize'][0]
    omegam = bf['header'].attrs['Omega0'][0]
    aa = bf['header'].attrs['Time'][0]
    redshift = 1.0 / (1.0 + aa)
    hpar = 0.697
    Hz1 = Hz(redshift, 100.0, omegam, 0, 1.0 - omegam)
    start = bf['FOFGroups/OffsetByType'][fofid]
    end = start + bf['FOFGroups/LengthByType'][fofid]
    result['FOFHaloID'] = fofid

    for k, sel in enumerate(range(start[5], end[5])):
        bhid = bf['5/ID'][sel]
        center = bf['5/Position'][sel]

        pos = numpy.concatenate(
            [bf['%d/Position' % p][start[p]:end[p]] for p in [0, 1, 4, 5]], axis=0)
        mass = numpy.concatenate(
            [bf['%d/Mass' % p][start[p]:end[p]] for p in [0, 1, 4, 5]], axis=0)

        # now do StellarMass in the nearby region:
        r200, m200, rho_c, r, rho = measure_m200(
            center, pos, mass, omegam, boxsize)

        sfr0 = bf['0/StarFormationRate'][start[0]:end[0]]
        pos0 = bf['0/Position'][start[0]:end[0]]
        pos5 = bf['5/Position'][start[5]:end[5]]
        id5 = bf['5/ID'][start[5]:end[5]]
        pos1 = bf['1/Position'][start[1]:end[1]]

        result['BlackholeID'][k] = bhid
        usebh = distp(center, pos5, boxsize) < r200

        result['NBlackhole'][k] = (distp(center, pos5, boxsize) < r200).sum()
        result['BlackholeMass'][k] = bf['5/BlackholeMass'][sel]
        # add the blackholes
        for blk in ['BlackholeAccretionRate', 'BlackholeMass']:
            result['Total' + blk][k] = bf['5/%s' % blk][start[5]:end[5]][distp(center, pos5,
                                                                               boxsize) < r200].sum(dtype='f8')

        pos4 = bf['4/Position'][start[4]:end[4]]
        vel4 = bf['4/Velocity'][start[4]:end[4]]
        usestar = distp(center, pos4, boxsize) < 1 * r200

        mass0 = bf['0/Mass'][start[0]:end[0]]
        mass1 = bf['1/Mass'][start[1]:end[1]]
        mass4 = bf['4/Mass'][start[4]:end[4]]
        mass5 = bf['5/Mass'][start[5]:end[5]]

        if usestar.sum() > 0:
            #result['DtoT'][k] = dtot(center, pos4[usestar], vel4[usestar], boxsize)
            sg_posi = pos4[usestar].mean(dtype='f8', axis=0)
            sg_lentypei = numpy.array(
                [len(pos0), len(pos1), 0, 0, len(pos4), len(pos5)])
            result['Position'][k][:] = sg_posi
            dtr, jtot_10hm_mag, jdir_10hm = dtot_jzbyjcirc(
                sg_posi, r200, pos0, mass0, pos1, mass1, pos4, mass4, vel4, pos5, mass5, sg_lentypei, boxsize, redshift, Hz1, hpar)
            result['DtoT'][k] = dtr
            result['StellarSpecifcAngMom'][k] = jtot_10hm_mag
            result['StellarSpinDirection'][k] = jdir_10hm
        else:
            result['DtoT'][k] = 0.0
            result['StellarSpecifcAngMom'][k] = 0
            (result['StellarSpinDirection'][k])[:] = 0
            result['Position'][k][:] = center

        result['BlackholePosition'][k][:] = center
        result['Mass200'][k] = m200
        result['StarFormationRate'][k] = sfr0[distp(
            center, pos0, boxsize) < r200].sum(dtype='f8')

#        usedm = (distp(center, pos1, boxsize) < r200)
        starcenter = dist3p(center, pos4[usestar], boxsize).mean(
            axis=0, dtype='f8')
        centralbh = distp(center + starcenter, pos5, boxsize).argmin()
        result['CentralBlackholeID'][k] = id5[centralbh]

        # add all masses
        for i in range(6):
            if end[i] != start[i]:
                posi = bf['%d/Position' % i][start[i]:end[i]]
                massi = bf['%d/Mass' % i][start[i]:end[i]]
                result['MassByType200'][k][i] = massi[distp(
                    center, posi, boxsize) < r200].sum(dtype='f8')
            else:
                result['MassByType200'][k][i] = 0

        result['R200'][k] = r200
    return result


def findaxis(dist, mass):
    """ v1, v2, v3 = findaxis(dist, mass) """
    dr = dist
    Imom = numpy.einsum('i,iu,iv->uv', mass, dr, dr,
                        dtype='f8') / mass.sum(dtype='f8')

    w, v = numpy.linalg.eig(Imom)

    v = v[:, numpy.abs(w).argsort()]
    w = w[numpy.abs(w).argsort()] ** 0.5

    return v.T


def dtot(center, pos, vel, boxsize):
    dist = dist3p(center, pos, boxsize)
    v0 = vel.mean(axis=0, dtype='f8')
    va, vb, vc = findaxis(dist, numpy.ones(len(dist)))
    vc = vc / (vc ** 2).sum() ** 0.5
    J = numpy.cross(dist, vel - v0)
    Jz = numpy.dot(J, vc)
    J = numpy.einsum('ij,ij->i', J, J) ** 0.5
    d1 = Jz / J > 0.5
    d2 = Jz / J < -0.5
    d1 = d1.sum()
    d2 = d2.sum()
#    print Jz.shape, J.shape, d.sum(), len(pos), 1.0 * d.sum() / len(pos)
    return 1.0 * max([d1, d2]) / len(pos)


def main():
    import bigfile
    import sharedmem
    import os.path

    bf = bigfile.BigFile(ns.pig)
    a = []
    with sharedmem.MapReduce() as pool:
        def work(i):
            r = create_subhalos(bf, i)
            print i
            return r
        a = pool.map(work, range(ns.N))

    a = numpy.concatenate(a)
    numpy.save(os.path.join(ns.pig, 'subhalos.npy'), a)

    return

    print numpy.bincount(a['NBlackhole'])
    for row in a:
        #        if row['BlackholeMass'] < 1e-3: continue
        if row['MassByType200'][4] < .5e0:
            continue
        print row['FOFHaloID'], row['DtoT'], row['BlackholeMass'],\
            row['BlackholeID'], \
            row['CentralBlackholeID'], row['NBlackhole']


if __name__ == '__main__':
    main()
