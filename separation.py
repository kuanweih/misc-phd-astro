import numpy as np
import sqlutilpy


def main():

    MAG_G_MIN = 17
    MAG_G_MAX = 21

    print('Querying data with g_mag between {} and {}.'.format(MAG_G_MIN, MAG_G_MAX))

    sql_str = """
              WITH sdss AS (SELECT s.ra, s.dec, s.type
                            FROM sdssdr9.phototag as s
                            WHERE s.ra > 0 and s.ra < 1 and s.mode=1)
              SELECT sdss.*, gaia.*
              FROM sdss
              LEFT JOIN LATERAL (SELECT g.ra, g.dec, g.phot_g_mean_mag,
                                        g.astrometric_excess_noise
                                 FROM gaia_dr2.gaia_source as g
                                 WHERE q3c_join(sdss.ra, sdss.dec,
                                                g.ra, g.dec, 1/3600.)
                                       and g.phot_g_mean_mag > {}
                                       and g.phot_g_mean_mag < {}
                                 ORDER BY q3c_dist(sdss.ra, sdss.dec, g.ra, g.dec)
                                 ASC LIMIT 1)
              AS gaia
              ON true
              """.format(MAG_G_MIN, MAG_G_MAX)

    sql_query = sqlutilpy.get(sql_str,
                              host='wsdb.hpc1.cs.cmu.edu',
                              user='kuan_wei', password='')

    print('Done querying and now filtering out nan in ra and dec.')
    print('Be aware of the index here though.')
    mask = (~np.isnan(sql_query[3])) & (~np.isnan(sql_query[4]))

    sql_query_new = []
    for data in sql_query:
        sql_query_new.append(data[mask])
    sql_query_new = np.array(sql_query_new)

    print('Done filtering and now saving.')
    np.save('gaia_sdss_g_{}_{}'.format(MAG_G_MIN, MAG_G_MAX), sql_query_new)

    print('All done :)')

if __name__ == '__main__':
    main()
