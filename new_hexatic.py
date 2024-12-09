import numpy as np
import pickle
from load_coords4 import load
from scipy import spatial
import sys


cutoff = 2 #upper bound for allowing a nearest neighbor to be counted
fname = 'trajectory.vtf'
bfname = None
begin = 0
end = -1
inter = 1

# return a list of points to add to data set as a buffer
# region of depth d surrounding the real data in order
# to make the voronoi calculation compatible with PBC
# x_tree and y_tree are cKDTree objects for the x and
# y coordinates of the data points
#
# this could potentially be faster if I find a quick
# way to find overlapping points for the corner buffers.
# at the moment, it adds more buffer points than is
# strictly necessary, which will make the voronoi
# calculation slower
def buffer_points(points, x_tree, y_tree, d, box):
    left = x_tree.query_ball_point([d/2.],d/2.)
    right = x_tree.query_ball_point([box[0]-(d/2.)],d/2)
    bot = y_tree.query_ball_point([d/2.],d/2.)
    top = y_tree.query_ball_point([box[1]-d/2.],d/2.)
    
    left_pts = [p+np.array([box[0],0.]) for p in points[left]]
    right_pts = [p-np.array([box[0],0.]) for p in points[right]]
    bot_pts = [p+np.array([0.,box[1]]) for p in points[bot]]
    top_pts = [p-np.array([0.,box[1]]) for p in points[top]]
    
    r_corners = [p-np.array([box[0],box[1]]) for p in points[right]]
    r_corners += [p-np.array([box[0],-box[1]]) for p in points[right]]
    l_corners = [p+np.array([box[0],box[1]]) for p in points[left]]
    l_corners += [p+np.array([box[0],-box[1]]) for p in points[left]]
    
    # adding in next line is list concatenation
    return left_pts + right_pts + bot_pts + top_pts + r_corners + l_corners

# compute the voronoi cell area using the shoelace formula
def cell_area(vor, p_ind):
    reg = vor.regions[vor.point_region[p_ind]]
    A = 0.0
    for i in range(len(reg)):
        vert = vor.vertices[reg[i]]
        n_vert = vor.vertices[reg[(i+1)%len(reg)]]
        A += (n_vert[0]+vert[0])*(n_vert[1]-vert[1])
    return np.abs(0.5*A)

step, box, types = load(fname,start=begin,stop=end,interval=inter,box=True,type=True)
print('Data loading done.')
print("len(step)={}, len(box)={}".format(len(step),len(box)))

i=1
if bfname != None:
    with open(bfname,"r") as bf:
        for line in bf:
            if (i%inter == 0) and (i >= begin) and (i <= end):
                box.append([float(x) for x in line.strip().split(",")])
            i += 1

#minimum image displacement vector from raw displacement dr
#inputs are expected to be numpy arrays
def mind(dr, box_l):
    return dr - box_l*np.floor(dr/box_l + 0.5)
    
#function to fold x,y coordinates into the simulation box
def fold(a,b):
    return [(a[0]+(2.*b[0]))%b[0], (a[1]+(2.*b[1]))%b[1]]


u_indices = []
d_indices = []
strays = []
    
for k in range(min(len(step),len(box))):
    zprod = np.array([np.dot(step[k][i][0]-step[k][i][-1],[0,0,1]) for i in range(len(step[k]))])
    
    #approximate - now we need to check for strays and vampires
    u_indices.append(np.where(zprod >= 0)[0])
    d_indices.append(np.where(zprod < 0)[0])
    
    z_folded = [[((step[k][i][1][2]+step[k][i][2][2]+step[k][i][3][2])/3.0)%box[k][2]] for i in range(len(step[k]))]
    z_folded = np.array(z_folded)

    #use kdtree in z direction to find strays
    z_tree = spatial.cKDTree(z_folded, boxsize=[box[k][-1]])
    strays.append([])
    for i in range(len(step[k])):
        inds = z_tree.query_ball_point(z_folded[i],1.5)
        if len(inds) == 1:
            #no neighbors, add index to stray list
            strays[k].append(inds[0])
            #remove stray index from up/down lists
            u_indices[k] = u_indices[k][u_indices[k] != inds[0]]
            d_indices[k] = d_indices[k][d_indices[k] != inds[0]]
               
 
print('Finished leaflet assignment for all lipids.\nStarting the calculation of psi6,areas,nearest-neighbors-dist and list of neighbors.')

with open("indices.pkl", 'wb') as file:
    pickle.dump([u_indices,d_indices], file)
with open("strays.pkl", 'wb') as file:
    pickle.dump(strays, file)
    
 
order_params = []
areas = []
nn_dists = []
nblist = []

for k in range(min(len(step),len(box))):
    #populate data arrays so they can be indexed out of order
    order_params.append([0.0]*len(step[0]))
    nn_dists.append([0.0]*len(step[0]))
    nblist.append([[]]*len(step[0]))      
    areas.append([0.0]*len(step[0]))
    

    u_folded = [fold(step[k][i][1][:-1],box[k][:-1]) for i in u_indices[k]]
    d_folded = [fold(step[k][i][1][:-1],box[k][:-1]) for i in d_indices[k]]
 
    u_xtree = spatial.cKDTree([[p[0]] for p in u_folded], boxsize=[box[k][0]])
    u_ytree = spatial.cKDTree([[p[1]] for p in u_folded], boxsize=[box[k][1]])
    u_padding = buffer_points(np.array(u_folded), u_xtree, u_ytree, 2., box[k][:-1])
    u_pts = np.append(u_folded, u_padding, axis=0)
    
    d_xtree = spatial.cKDTree([[p[0]] for p in d_folded], boxsize=[box[k][0]])
    d_ytree = spatial.cKDTree([[p[1]] for p in d_folded], boxsize=[box[k][1]])
    d_padding = buffer_points(np.array(d_folded), d_xtree, d_ytree, 2., box[k][:-1])
    d_pts = np.append(d_folded, d_padding, axis=0)
    
    u_vor = spatial.Voronoi(u_pts)
    d_vor = spatial.Voronoi(d_pts)
    
    u_folded = np.array(u_folded)
    d_folded = np.array(d_folded)

    u_tree = spatial.cKDTree(u_folded, boxsize=box[k][:-1]+0.1)
    d_tree = spatial.cKDTree(d_folded, boxsize=box[k][:-1]+0.1)


    for i in range(len(u_folded)):
        areas[k][u_indices[k][i]] = cell_area(u_vor,i)
        nb_angles = []
        dists, arr_inds = u_tree.query(u_folded[i],k=7)
        inds = np.array([u_indices[k][m] for m in arr_inds])
               
        nn_dists[k][u_indices[k][i]] = np.amin(dists[1:])
        nblist[k][u_indices[k][i]] = inds[1:]
        for pair in zip(dists[1:],arr_inds[1:]):
            if pair[0] > cutoff:
                continue
            md = mind(u_folded[pair[1]]-u_folded[i], box[k][:-1])
            nb_angles.append(np.arctan2(md[1],md[0])) #note that y is first arg of arctan2
        
        op = np.array([0.0,0.0])
        for a in nb_angles:
            op += np.array([np.cos(6.0*a),np.sin(6.0*a)])
        op = op / float(len(nb_angles))
        op_scalar = np.linalg.norm(op)
        order_params[k][u_indices[k][i]] = op_scalar


    for i in range(len(d_folded)):
        areas[k][d_indices[k][i]] = cell_area(d_vor,i)
        nb_angles = []
        dists, arr_inds = d_tree.query(d_folded[i],k=7)
        inds = np.array([d_indices[k][m] for m in arr_inds])

        nn_dists[k][d_indices[k][i]] = np.amin(dists[1:])
        nblist[k][d_indices[k][i]] = inds[1:]
        for pair in zip(dists[1:],arr_inds[1:]):
            if pair[0] > cutoff:
                continue
            md = mind(d_folded[pair[1]]-d_folded[i], box[k][:-1])
            nb_angles.append(np.arctan2(md[1],md[0])) #note that y is first arg of arctan2
        
        op = np.array([0.0,0.0])
        for a in nb_angles:
            op += np.array([np.cos(6.0*a),np.sin(6.0*a)])
        op = op / float(len(nb_angles))
        op_scalar = np.linalg.norm(op)
        order_params[k][d_indices[k][i]] = op_scalar


np.save("order_params", order_params)
np.save("steps", step)
np.save("box_data", box)
np.save("nn_dists", nn_dists)
np.save("areas", areas)
np.save("types",types)


print("Looped over all lipids.\n Moving on to calculation of (time) avg psi6.")


op_avgs = []
for i in range(min(len(step),len(box))):
    op_avgs.append([])
    for j in range(len(nblist[i])):
        ops = [order_params[i][k] for k in nblist[i][j]]
        if len(ops) > 0:
            op_avgs[-1].append(np.mean(ops))
        else:
            op_avgs[-1].append(0.0)
np.save("op_avgs",op_avgs)
print("Saved avg psi6 order parameters.")


nbtype =[]
for i in range(min(len(step),len(box))):
    nbtype.append([])
    for j in range(len(nblist[i])):
        nbtypelist = [types[4*k + 1] for k in nblist[i][j]]
      
        if len(nbtypelist)>0:
            type, count = np.unique(nbtypelist,return_counts=True)
            nbtype[-1].append(int(type[np.argmax(count)])) 
        else:
            nbtype[-1].append(types[4*j + 1])
            print(f"Zero neighbors found for lipid {j}. Assigning self type as neighbor type.")
            
np.save("nbtype",nbtype)

with open("nb_inds.pkl", 'wb') as file:
    pickle.dump(nblist, file)
    
print("Saved the list of most frequent neighbor types and indices of neighbors.")



print("Finished analysing this trajectory.")