import numpy as np
import matplotlib.pyplot as plt


def geometry(n_ring, radius, gap=0, inner_gap=False, orient=0, center=(0, 0),
             missing=[]):
    '''Creates the hexagonal geometry of an segmented telescope given a
    number of rings and the radius of a segment
 
    Parameters
    ----------
    n_ring : int
        Number of segment rings

    radius : float
        Radius of one segment, in arbitrary units

    gap : float
        Size of the inter-segment gap, defined as a fraction of the
        segment radius. Default is 0

    inner_gap : bool
        Definition adopted for the gap (see Notes). The default is 
        an "outer gap"

    orient : float
        Global orientation of the segmented mirror, in
        degrees. Default is 0

    center : tuple
        Center of the central segment, in arbitrary units. Default is
        (0, 0)

    missing : list
        List of missing segment numbers

    Returns
    -------
    segments : reccord array
        Reccord array containing all the information regarding all
        segments. See Notes for a full description.

    system : array
        A matrix poviding the link between segments and borders

    Notes
    -----

    It creates an hexagonal geometry given a number of rings, a radius
    for the segment and an inter-segment gap. The general setup is to
    create a geometry where the segments are flat-side up, with the
    reference being the central segment. Numbering of segments and
    borders are then defined with respect to that reference. An
    additional orientation offset can be provided to rotate the whole
    geometry.
 
    The number of rings defines the total number of circular rings
    made by the segments. It includes the central segment as ring
    number 0. Subsequent rings are numbered increasingly from the
    center.
 
    The radius of the segment defines the distance between the center
    of the hexagon and any of its nodes. It is provided in arbitrary
    units. Although it is not necessary to assume any physical unit
    when building the geometry of the system, it becomes unavoidable
    when introducing tip-tilt aberrations on the segment at a later
    stage.
 
    The inter-segment gap defines the small space that necessarily
    exist between two physically independant segments. It is defined
    as a fraction of the segment radius. It can be either defined as
    an "outer" gap (default) or an "inner" gap. In the former case,
    the effective physical radius of the segment is exactly equal to
    radius, and the gap is added in-between the segments. In the
    latter case, the effective physical radius of the segment becomes
    radius * (1 - gap) so that the space occupied by a segment and the
    gap is equal to radius.
 
    Given the three previously described parameters, the procedure
    creates a geometry starting from the center of the central
    segment, which has coordinates (0,0) by default (a user-defined
    value can be provided). The coordinates of the centers of all
    segments are calculated taking into account the inter-segment
    gap. The coordinates of the middle of each of the borders and of
    each of the nodes are also computed.
 
    Segment numbering follows the following convention: for any ring
    the numbering is started from the top segment, and increases in a
    clockwise manner:
                                __
                             __/ 7\__
                          __/18\__/ 8\__
                         /17\__/ 1\__/ 9\
                         \__/ 6\__/ 2\__/
                         /16\__/ 0\__/10\
                         \__/ 5\__/ 3\__/
                         /15\__/ 4\__/11\
                         \__/14\__/12\__/
                            \__/13\__/
                               \__/
 
    Border number is slighlty less straightforward because there are
    borders between the segments of a same ring, and borders between
    segments from different rings. For a given ring, the convention
    adopted is to:
 
       - first number borders between segments inside the current
         ring, in a clockwise manner, starting from border index 2
         (see below) of the top segment of the ring;
 
       - then number the outer borders (i.e. between the current ring
         and the next one), in a clockwise manner, starting from
         border index 0 of the top segment of the ring.
 
    Inside a given segment, the following convention is adopted for
    the indexing of the nodes and borders:
 
                Node index             Border index
 
                0_______1                ___0___
                /       \               /       \
               /         \            5/         \1
             5/     +     \2          /     +     \
              \           /           \           /
               \         /            4\         /2
                \_______/               \_______/
                4       3                   3
 
    The orientation of this geometry will then change according to the
    orient parameter provided by the user.

    The first main output of the geometry() procedure is a structured
    array, which contain the following tags:
 
       .num - number of the segment
 
       .ring - ring number of the segment
 
       .missing - boolean indicating if the segment is missing
 
       .s - array of 6 integers giving the numbers of the neighbouring
            segments. It uses the same convention as the border
            indices, i.e. S[1] gives the number of the segment sharing
            the border at index 1 with the current segment
 
       .b - array of 6 integers giving the numbers of the borders for
            the current segment. It uses the border index numbering
            convention given above
 
       .center - coordinate of the segment center
 
       .center_border - coordinates of the 6 segment borders

       .center_node - coordinates of the 6 segment nodes
 
    The second main output is the system matrix, which links segment
    borders with segments.

    '''
    
    # number of segments and borders
    n_segment = 3 * n_ring * (n_ring+1) + 1
    n_border  = np.sum(np.arange(n_ring, dtype=np.int)*3*6+6) + 6*n_ring
    
    print('{:03d} segments'.format(n_segment))
    print('{:03d} borders'.format(n_border))
    
    # segments definition
    dtype = np.dtype([('num', np.int),
                      ('ring', np.int),
                      ('missing', np.bool),
                      ('s', np.int, (6, )),
                      ('b', np.int, (6, )),
                      ('center', np.float, (2, )),
                      ('center_border', np.float, (6, 2)),
                      ('center_node', np.float, (6, 2))])
    segments = np.rec.array(np.zeros(n_segment, dtype=dtype))

    # default values
    segments.num  = -1
    segments.ring = -1
    segments.s    = -1
    segments.b    = -1
    
    # central segment
    segments[0].num    = 0
    segments[0].ring   = 0
    segments[0].s      = np.arange(6)+1
    segments[0].b      = np.arange(6)
    segments[0].center = center

    # missing segments
    for m in missing:
        segments[m].missing = True
    
    # some constants
    fact = np.sqrt(3)

    # define rad depending on gap option
    if inner_gap:
        rad  = radius
    else:
        rad  = radius * (1 + gap)
        
    i = 1
    b = 6
    for r in range(1, n_ring+1):
        n_seg_ring = 6*r
    
        #
        # center + internal borders
        #
        start = i
    
        # top segment
        refc = (center[0], center[1] + r*rad*fact)
        segments[i].num    = i
        segments[i].ring   = r
        segments[i].center = refc
        segments[i].b[2]   = b
        b += 1
        i += 1
            
        # going South-East
        refc = segments[i-1].center
        for s in range(1, r+1):
            cx = refc[0] + s*3*rad/2
            cy = refc[1] - s*rad*fact/2
            segments[i].num    = i
            segments[i].ring   = r
            segments[i].center = (cx, cy)
            if s < r:
                segments[i].b[2] = b
            else:
                segments[i].b[3] = b
            b += 1
            i += 1
    
        # going South
        refc = segments[i-1].center
        for s in range(1, r+1):
            cx = refc[0] + 0
            cy = refc[1] - s*rad*fact
            segments[i].num    = i
            segments[i].ring   = r
            segments[i].center = (cx, cy)
            if s < r:
                segments[i].b[3] = b
            else:
                segments[i].b[4] = b
            b += 1
            i += 1
    
        # going South-West
        refc = segments[i-1].center
        for s in range(1, r+1):
            cx = refc[0] - s*3*rad/2
            cy = refc[1] - s*rad*fact/2
            segments[i].num    = i
            segments[i].ring   = r
            segments[i].center = (cx, cy)
            if s < r:
                segments[i].b[4] = b
            else:
                segments[i].b[5] = b
            b += 1
            i += 1
    
        # going North-West
        refc = segments[i-1].center
        for s in range(1, r+1):
            cx = refc[0] - s*3*rad/2
            cy = refc[1] + s*rad*fact/2
            segments[i].num    = i
            segments[i].ring   = r
            segments[i].center = (cx, cy)
            if s < r:
                segments[i].b[5] = b
            else:
                segments[i].b[0] = b
            b += 1
            i += 1
    
        # going North
        refc = segments[i-1].center
        for s in range(1, r+1):
            cx = refc[0] + 0
            cy = refc[1] + s*rad*fact
            segments[i].num    = i
            segments[i].ring   = r
            segments[i].center = (cx, cy)
            if s < r:
                segments[i].b[0] = b
            else:
                segments[i].b[1] = b
            b += 1
            i += 1
    
        # going North-East
        refc = segments[i-1].center
        for s in range(1, r):
            cx = refc[0] + s*3*rad/2
            cy = refc[1] + s*rad*fact/2
            segments[i].num    = i
            segments[i].ring   = r
            segments[i].center = (cx, cy)
            segments[i].b[1]   = b
            b += 1
            i += 1

        #
        # external borders
        #
        i = start
        
        # top segment
        segments[i].b[0] = b
        b += 1
        segments[i].b[1] = b
        b += 1
        i += 1

        # going South-East
        for s in range(1, r+1):
            if s < r:
                segments[i].b[0] = b
                b += 1
                segments[i].b[1] = b
                b += 1
            else:
                segments[i].b[0] = b
                b += 1
                segments[i].b[1] = b
                b += 1
                segments[i].b[2] = b
                b += 1
            i += 1
        
        # going South
        for s in range(1, r+1):
            if s < r:
                segments[i].b[1] = b
                b += 1
                segments[i].b[2] = b
                b += 1
            else:
                segments[i].b[1] = b
                b += 1
                segments[i].b[2] = b
                b += 1
                segments[i].b[3] = b
                b += 1
            i += 1
    
        # going South-West
        for s in range(1, r+1):
            if s < r:
                segments[i].b[2] = b
                b += 1
                segments[i].b[3] = b
                b += 1
            else:
                segments[i].b[2] = b
                b += 1
                segments[i].b[3] = b
                b += 1
                segments[i].b[4] = b
                b += 1
            i += 1
        
        # going North-West
        for s in range(1, r+1):
            if s < r:
                segments[i].b[3] = b
                b += 1
                segments[i].b[4] = b
                b += 1
            else:
                segments[i].b[3] = b
                b += 1
                segments[i].b[4] = b
                b += 1
                segments[i].b[5] = b
                b += 1
            i += 1
    
        # going North
        for s in range(1, r+1):
            if s < r:
                segments[i].b[4] = b
                b += 1
                segments[i].b[5] = b
                b += 1
            else:
                segments[i].b[4] = b
                b += 1
                segments[i].b[5] = b
                b += 1
                segments[i].b[0] = b
                b += 1
            i += 1
        
        # going North-East
        for s in range(1, r):
            segments[i].b[5] = b
            b += 1
            segments[i].b[0] = b
            b += 1
            i += 1
        
        # last border is on first segment of current ring!
        segments[i-n_seg_ring].b[5] = b
        b += 1
        
    # neighbors
    segs = np.arange(n_segment)
    for s in range(n_segment):
        d = segments.center - segments[s].center
        sep = np.sqrt(np.sum(d**2, axis=1))
        pa  = np.rad2deg(np.arctan2(d[:, 1], d[:, 0]))
    
        segments[s].s[:] = -1
        
        cseg = segs[(np.abs(sep-fact*rad) <= 0.1) & (np.abs(pa-90)  <= 0.1)]
        if len(cseg) != 0:
            segments[s].s[0] = cseg
    
        cseg = segs[(np.abs(sep-fact*rad) <= 0.1) & (np.abs(pa-30)  <= 0.1)]
        if len(cseg) != 0:
            segments[s].s[1] = cseg
    
        cseg = segs[(np.abs(sep-fact*rad) <= 0.1) & (np.abs(pa+30)  <= 0.1)]
        if len(cseg) != 0:
            segments[s].s[2] = cseg
    
        cseg = segs[(np.abs(sep-fact*rad) <= 0.1) & (np.abs(pa+90)  <= 0.1)]
        if len(cseg) != 0:
            segments[s].s[3] = cseg
    
        cseg = segs[(np.abs(sep-fact*rad) <= 0.1) & (np.abs(pa+150) <= 0.1)]
        if len(cseg) != 0:
            segments[s].s[4] = cseg
    
        cseg = segs[(np.abs(sep-fact*rad) <= 0.1) & (np.abs(pa-150) <= 0.1)]
        if len(cseg) != 0:
            segments[s].s[5] = cseg
            
    # fill missing borders
    for s in range(n_segment):
        if segments[s].b[0] == -1:
            segments[s].b[0] = segments[segments[s].s[0]].b[3]
        if segments[s].b[1] == -1:
            segments[s].b[1] = segments[segments[s].s[1]].b[4]
        if segments[s].b[2] == -1:
            segments[s].b[2] = segments[segments[s].s[2]].b[5]
        if segments[s].b[3] == -1:
            segments[s].b[3] = segments[segments[s].s[3]].b[0]
        if segments[s].b[4] == -1:
            segments[s].b[4] = segments[segments[s].s[4]].b[1]
        if segments[s].b[5] == -1:
            segments[s].b[5] = segments[segments[s].s[5]].b[2]

    # redefine rad depending on gap option
    if inner_gap:
        rad = radius * (1 - gap)
    else:
        rad = radius
    
    # borders centers
    for s in range(n_segment):
        segments[s].center_border[0, :] = segments[s].center + (       0,  rad*fact/2)
        segments[s].center_border[1, :] = segments[s].center + ( 3*rad/4,  rad*fact/4)
        segments[s].center_border[2, :] = segments[s].center + ( 3*rad/4, -rad*fact/4)
        segments[s].center_border[3, :] = segments[s].center + (       0, -rad*fact/2)
        segments[s].center_border[4, :] = segments[s].center + (-3*rad/4, -rad*fact/4)
        segments[s].center_border[5, :] = segments[s].center + (-3*rad/4,  rad*fact/4)

    # nodes centers
    for s in range(n_segment):
        segments[s].center_node[0, :] = segments[s].center + (-rad/2,  rad*fact/2)
        segments[s].center_node[1, :] = segments[s].center + ( rad/2,  rad*fact/2)
        segments[s].center_node[2, :] = segments[s].center + (   rad,           0)
        segments[s].center_node[3, :] = segments[s].center + ( rad/2, -rad*fact/2)
        segments[s].center_node[4, :] = segments[s].center + (-rad/2, -rad*fact/2)
        segments[s].center_node[5, :] = segments[s].center + (  -rad,           0)
        
    # apply orientation
    if orient != 0:
        rot = -np.deg2rad(orient)
        matrix = np.matrix([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])

        cref = segments[0].center
        
        # segments
        c  = segments.center - cref
        nc = c @ matrix
        segments.center = nc + cref

        # borders & nodes
        for e in range(6):
            c  = segments.center_border[:, e, :] - cref
            nc = c @ matrix
            segments.center_border[:, e, :] = nc + cref

            c  = segments.center_node[:, e, :] - cref
            nc = c @ matrix
            segments.center_node[:, e, :] = nc + cref
            
    # matrix of the system
    system = np.zeros((n_segment, n_border))
    for s in range(n_segment):
        if (segments[s].missing):
            continue
        
        if (segments[s].b[3] < n_border) and (not segments[segments[s].s[3]].missing):
            system[s, segments[s].b[3]] = 1
        if (segments[s].b[2] < n_border) and (not segments[segments[s].s[2]].missing):
            system[s, segments[s].b[2]] = 1
        if (segments[s].b[1] < n_border) and (not segments[segments[s].s[1]].missing):
            system[s, segments[s].b[1]] = 1
    
        if (segments[s].b[0] < n_border) and (not segments[segments[s].s[0]].missing):
            system[s, segments[s].b[0]] = -1
        if (segments[s].b[5] < n_border) and (not segments[segments[s].s[5]].missing):
            system[s, segments[s].b[5]] = -1
        if (segments[s].b[4] < n_border) and (not segments[segments[s].s[4]].missing):
            system[s, segments[s].b[4]] = -1
    # for s in range(n_segment):
    #     if segments[s].b[3] < n_border:
    #         system[s, segments[s].b[3]] = 1
    #     if segments[s].b[2] < n_border:
    #         system[s, segments[s].b[2]] = 1
    #     if segments[s].b[1] < n_border:
    #         system[s, segments[s].b[1]] = 1
    
    #     if segments[s].b[0] < n_border:
    #         system[s, segments[s].b[0]] = -1
    #     if segments[s].b[5] < n_border:
    #         system[s, segments[s].b[5]] = -1
    #     if segments[s].b[4] < n_border:
    #         system[s, segments[s].b[4]] = -1

    return segments, system


def plot(segments, system, margin=0.05):
    '''
    Plot segments created by the geometry() function

    Parameters
    ----------
    segments : rec array
        Segments information

    margin : float 
        Margin around the external border of the mirror, in fraction
        of the total with of the pupil (approximately). Default is 
        0.05 (5%).
    '''

    nring = np.max(segments.ring)
    nseg  = len(segments)
    nbord = segments.b.max()
    
    cmin = np.min(segments.center_node, axis=(0, 1))
    cmax = np.max(segments.center_node, axis=(0, 1))
    ext  = margin*np.max(cmax - cmin)
    
    plt.figure('Segments', figsize=(10, 10))
    plt.clf()

    for s in range(nseg):
        if segments[s].missing:
            color = 'r'
        else:
            color = 'g'
            
        plt.text(segments[s].center[0], segments[s].center[1], segments[s].num,
                 color=color, ha='center',
                 va='center', weight='bold', size='xx-large')

        cx = segments.center_node[s, :, 0]
        cy = segments.center_node[s, :, 1]
        plt.plot(np.append(cx, cx[0]), np.append(cy, cy[0]), color='k')
        
    status = np.abs(sys).sum(axis=0)
    for b in range(nbord):
        idx_x, idx_y = np.where(seg.b == b)
        if len(idx_x):
            if status[b]:
                color = 'g'
            else:
                color = 'r'
            
            cc = segments.center_border[idx_x[0], idx_y[0]]
            plt.plot(cc[0], cc[1], linestyle='none', marker='o', color=color)
        
    plt.xlim(cmin[0]-ext, cmax[0]+ext)
    plt.ylim(cmin[1]-ext, cmax[1]+ext)

    plt.title(r'$N_{{ring}}$={:d} - $N_{{seg}}$={:d}'.format(nring, nseg))
    
    plt.tight_layout()


if __name__ == '__main__':
    seg, sys = geometry(4, 1, gap=0, inner_gap=False, orient=0, center=(0, 0),
                        missing=[7])
    
    plot(seg, sys)
