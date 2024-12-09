import sys
import os
import time
import numpy as np
from random import shuffle
from datetime import datetime

import espressomd

#For VMD
from espressomd.io.writer import vtf

#Our libraries
from mbtools import *
from Lipid import Lipid

verb = True
jobname = "data"

#thermostat indicator - 0 (default) is set to run NPT simulation.
ts_flag = 0 
temperature = 1.4

nargs = 6 #number of command line arguments
#taking input arguments from command line
if (len(sys.argv)-1) < nargs:
    print("Not all commandline arguments were entered...")
    quit()
else:
    nut = int(sys.argv[1])
    nst = int(sys.argv[2])
    nct = int(sys.argv[3])
    nub = int(sys.argv[4])
    nsb = int(sys.argv[5])
    ncb = int(sys.argv[6])
    iterations = int(sys.argv[7])
    

# System parameters
#############################################################
#Main simulation parameters:
skin 		  = 0.4
time_step 	  = 0.005

#Look at the main integration loop near the end to make sense of these:
integrator_steps  = 200
sampling_interval = 20
if iterations<200:
  iterations      = 150000
  
print(f'Iterations = {iterations}')

#Warmup parameters:
warm_tstep  = 0.001
warm_steps  = 10
warm_n_time = 10000
min_dist    = 0.90
lj_cap = 4.

#############################################################
# Standard Cooke Interaction parameters (see paper)
#############################################################					
lj_eps = 1.0
cshift = 1./4.
#lipid bead radii = h-head,m-middle,t-tail
rh = 0.95
rm = 1.
rt = 1.
#chol bead radii
sm_fr = 0.2
ch = (1.- sm_fr)*rh
cm = (1.- sm_fr)*rm
ct = (1.- sm_fr)*rt
#all bead radii
r = [rh,rt,rm,rm,rm,rm,cm,rh,ch,ct]
global b
b = []
for i in range(len(r)):
    b.append([]*len(r))
    for j in range(len(r)):
        b[i].append((r[i]+r[j])/2.)
        

#############################################################
#wc parameters for different types of interactions
#############################################################
wc_0 = 1.6 ##default interaction parameter
wc_us = 1.4
wc_uc = 1.68
wc_sc = 1.75
global wc
wc = np.full((len(r),len(r)), wc_0)
for i in range(len(wc)):
    for j in range(len(wc)):
        if (i,j) in [(3,2),(5,4),(2,3),(4,5)]:
            wc[i][j] = wc_us
        elif (i,j) in [(6,2),(6,4),(2,6),(4,6)]:
            wc[i][j] = wc_uc
        elif (i,j) in [(6,3),(6,5),(3,6),(5,6)]:
            wc[i][j] = wc_sc
          

#####################
##spring constants###
#####################
kval_u = 5.0
kval_s = 25.0

print("kT={}".format(temperature))
print("wSC={}".format(wc_uc))
print("wUC={}".format(wc_sc))
print("wUS={}".format(wc_us))
print("kbend_U={}".format(kval_u))
print("kbend_S={}".format(kval_s))


ntop = nut + nst + nct
nbot = nub + nsb + ncb 
ntot = ntop + nbot


#box dimensions
lx=round(np.sqrt(1.16*ntot/2))
ly=lx
lz=20.0

def main():
        # Set Up
        #############################################################
        system = espressomd.System(box_l = [lx,ly,lz]) #periodic BC by default
        # system.set_random_state_PRNG()
        system.time_step = warm_tstep
        system.cell_system.skin = skin

        #Membrane
        ############################################################
        typeOne = {"Head":0,"Mid1":2,"Mid2":2,"Tail":1} #At
        typeTwo = {"Head":7,"Mid1":3,"Mid2":3,"Tail":1} #Bt
        typeThr = {"Head":0,"Mid1":4,"Mid2":4,"Tail":1} #Ab
        typeFour = {"Head":7,"Mid1":5,"Mid2":5,"Tail":1} #Bb
        typeFive = {"Head":8,"Mid1":6,"Mid2":6,"Tail":9} #C


        uPos, dPos, uAngle, dAngle = flatBilayer(system, numA=ntop, numB=nbot, verbose=verb, z0=2.5)
        
        ulipids = [Lipid(system,lipidType=typeOne,k_bend=kval_u) for i in range(int(nut))]+[Lipid(system,lipidType=typeTwo,k_bend=kval_s) for i in range(int(nst))]+[Lipid(system,lipidType=typeFive) for i in range(int(nct))]
        dlipids = [Lipid(system,lipidType=typeThr,k_bend=kval_u) for i in range(int(nub))]+[Lipid(system,lipidType=typeFour,k_bend=kval_s) for i in range(int(nsb))]+[Lipid(system,lipidType=typeFive) for i in range(int(ncb))]
        
        shuffle(ulipids)
        shuffle(dlipids)

        placeLipids(ulipids, dlipids, uPos, dPos, uAngle, dAngle)
    
        lipids = ulipids + dlipids
         
        
        if verb: 
            print("Total number of lipids: " + str(len(lipids)))
            print(f"Placing {nut} unsaturated lipids, {nst} saturated lipids and {nct} cholesterol in the top leaflet.")
            print(f"Placing {nub} unsaturated lipids, {nsb} saturated lipids and {ncb} cholesterol in the bottom leaflet.")

        # Non bonded Interactions between the beads
        # See espressomd documentation for details
        #############################################################
        global b
        global wc
        for i in range(len(r)):
            # note: j < i
            for j in range(i+1):
                if (i in [0,7,8]) or (j in [0,7,8]):
                    # purely repulsive WCA potential
                    system.non_bonded_inter[i, j].lennard_jones.set_params(
                                        epsilon=lj_eps, sigma=b[i][j],
                                        cutoff=np.power(2.0, 1.0/6.0)*b[i][j], shift=cshift)
                elif (i,j) in [(4,2),(5,2),(4,3),(5,3)]:
                    # flip-fix repulsive cross-leaflet midbead interaction
                    system.non_bonded_inter[i, j].lennard_jones.set_params(
                                        epsilon=lj_eps, sigma=b[i][j],
                                        cutoff=np.power(2.0, 1.0/6.0)*b[i][j], shift=cshift)
                else:
                    system.non_bonded_inter[i, j].lennard_jones_cos2.set_params(
                                        epsilon=lj_eps, sigma=b[i][j],
                                        width=wc[i][j], offset=0.)


        #############################################################
        #  Output Files                                             #
        #############################################################
        try:
                os.mkdir(jobname)
        except OSError:
                print("Directory {} already exists or could not be created.".format(jobname))

        with open(jobname+"/trajectory.vtf", "w") as vtf_fp, open(jobname+"/energy.txt","w") as en_fp, \
                open(jobname+"/box.txt","w") as box_fp, open(jobname+"/flipflop.txt","w") as ff_fp, \
                open(jobname+"/pressure.txt","w") as pr_fp, open(jobname+"/lipidcount.txt","w") as lc_fp:

                #############################################################
                #  Warmup Integration                                       #
                #############################################################

                # write structure block as header
                vtf.writevsf(system, vtf_fp)
                # write initial positions as coordinate block
                vtf.writevcf(system, vtf_fp)
                vtf_fp.write("unitcell {} {} {}\n".format(*np.copy(system.box_l)))
                #NOTE: WRITING INITIAL CONFIG FIRST

                if verb: print("""
                Start warmup integration:
                At maximum {} times {} steps
                Stop if minimal distance is larger than {}
                """.format(warm_n_time, warm_steps, min_dist))

                i = 0
                global lj_cap
                system.force_cap = lj_cap
                act_min_dist = system.analysis.min_dist()
                while i < warm_n_time and act_min_dist < min_dist:
                        system.integrator.run(warm_steps)
                        act_min_dist = system.analysis.min_dist()
                        if verb: print("run {} at time = {} (LJ cap= {} ) min dist = {}".strip().format(i, system.time, lj_cap, act_min_dist))
                        i += 1
                        system.force_cap = lj_cap #can choose to increment this here if desired

                system.force_cap = 0.         #disable force capping
                system.time_step = time_step  #increase timestep after warmup

                if verb: print("\nWarm up finished\n")
                if verb: print("The box is {} by {}".format(system.box_l[0],system.box_l[1]))

                #if loop to determine whether the simulation is NPT or NVT.
                if ts_flag==0:
                    print("\nRunning NPT simulation.\n")
                    system.thermostat.set_npt(kT=temperature, gamma0=1.0, gammav=0.0002, seed=int(np.random.rand()*100000))
                    system.integrator.set_isotropic_npt(ext_pressure=0, piston=0.01,direction=[1,1,0])
                elif ts_flag==1:
                    print("\nRunning NVT simulation.\n")
                    system.thermostat.set_langevin(kT=temperature, gamma=1.0, seed=int(np.random.rand()*100000))
                else:
                    print("\nThermostat not set.\n")
                    quit()


                #############################################################
                #  Main Simulation                                          #
                #############################################################
                start_time = time.time()

                for i in range(1, iterations+1):
                        system.integrator.run(integrator_steps)

                        if(i % sampling_interval == 0):
                                vtf.writevcf(system, vtf_fp)
                                vtf_fp.write("unitcell {} {} {}\n".format(*np.copy(system.box_l)))

                                e = system.analysis.energy()
                                en_fp.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(e["total"],e["kinetic"],e["bonded"],e["non_bonded"],
                                        e["non_bonded",1,1], e["non_bonded",1,2],e["non_bonded",1,3], e["non_bonded",1,4],e["non_bonded",1,5],e["non_bonded",1,6],
                                        e["non_bonded",2,2], e["non_bonded",2,3],e["non_bonded",2,4],e["non_bonded",2,5],e["non_bonded",2,6],
                                        e["non_bonded",3,3],e["non_bonded",3,4],e["non_bonded",3,5],e["non_bonded",3,6],
                                        e["non_bonded",4,4],e["non_bonded",4,5],e["non_bonded",4,6],
                                        e["non_bonded",5,5],e["non_bonded",5,6],
                                        e["non_bonded",6,6]))

                                b = np.copy(system.box_l)
                                box_fp.write("{},{},{}\n".format(b[0],b[1],b[2]))

                                f = leafletContent3types(system, lipids, typeOne, typeTwo,typeThr,typeFour,typeFive,typeFive)
                                ff_fp.write("{},{},{},{},{},{},{},{},{}\n".format(*f))
                                
                                lc = leafletContent_generic(system, lipids, [typeOne, typeTwo, typeThr, typeFour, typeFive])                                
                                lc_fp.write(','.join(map(str, lc)) + '\n')
                                
                                pt = system.analysis.pressure_tensor()
                                pr_fp.write("{},{},{},{},{},{},{},{},{}\n".format(pt["total"][0][0],pt["total"][0][1],
                                pt["total"][0][2],pt["total"][1][0],pt["total"][1][1],pt["total"][1][2],
                                pt["total"][2][0],pt["total"][2][1],pt["total"][2][2]))

                                vtf_fp.flush()
                                en_fp.flush()
                                box_fp.flush()
                                ff_fp.flush()
                                pr_fp.flush()
                                lc_fp.flush()

                                print("Completed {} iterations".format(i))

        elapsed = time.time() - start_time
        print(jobname + " Done. Elapsed time {} seconds".format(elapsed))

main()
