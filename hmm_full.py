import numpy as np
import hmmlearn.hmm as hmm
from scipy.spatial import distance
import pickle
import time
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys

####################################################################################
### THERE ARE MULTIPLE OPTIONS TO CUSTOMIZE THE CODE TO WORK FOR A SPECIFIC CASE ###
###         THE FOLLOWING FLAGS WILL HELP TURN THESE OPTIONS ON AND OFF          ###
####################################################################################
###If all options are False, it loads existing label and types###
### from the folder and prints the composition of the phases ###
# upper=True #if True, runs the analysis for the upper leaflet; if False, runs it for the lower leaflet
train = False ##if True, a new model is trained using the specified training set; else, an existing model is loaded
testing = True ##if True, labeling is done using model; else, moves directly to plots and animation by loading existing labels in the directory
makeplt = True ##if True, generates plots of the specified frame (last frame by default)
makeanim = True ##if True, generates an animation with the last 500 frames
onlyphase = True ##if True, animation is generated with phase information only (black and yellow)


#read command line arguments
sim = sys.argv[1]
filenum = int(sys.argv[2])
if sys.argv[3]=='u':
    upper = True
    print("All analysis will be performed on the upper leaflet.")
elif sys.argv[3]=='l':
    upper = False
    print("All analysis will be performed on the lower leaflet.")
stval = int(sys.argv[4])

## going to use features: nnd_dist, apl, nbtype
n_feats = 3
t_date = "jan20-0" ##training data
# t_date = 'april20-2'
leaflet = "" if upper==True else "lower"

sta = time.time()
    
date = [f"{sim}-{i}" for i in range(stval,filenum)]
traj_files = [f"/export/home/malavikv/binary_mixtures/simulation/{date}/data/trajectory.vtf" for date in date]
odirs = [f"/export/home/malavikv/binary_mixtures/simulation/{date}/data" for date in date]
  
    
if train==True:
    # training directory
    t_dir = f"/export/home/malavikv/binary_mixtures/simulation/{t_date}/data"

    ############################################################
    ## train the HMM
    ############################################################

    # load the data for the training sim
    print(f"Loading training feature data from {t_date}...")

    # indices of lipids in top/bot leaflet
    try:
        with open("{}/indices.pkl".format(t_dir),'rb') as file:
            t_u_inds, t_d_inds = pickle.load(file)
    except:
        t_u_inds, t_d_inds = np.load("{}/indices.npy".format(t_dir),allow_pickle=True,encoding='latin1')
            
    # most occuring type for nearest neighbor of each lipid
    t_nbtype = np.load("{}/nbtype.npy".format(t_dir),allow_pickle=True).transpose()
    t_nbtype[t_nbtype == 4] = 2
    t_nbtype[t_nbtype == 5] = 3
    # distance to nearest neighbor for each lipid
    t_nnd = np.load("{}/nn_dists.npy".format(t_dir),allow_pickle=True).transpose()
    #area per lipid
    t_apl = np.load("{}/areas.npy".format(t_dir),allow_pickle=True).transpose()
    #hexatic order parameter
    t_psi6 = np.load("{}/op_avgs.npy".format(t_dir),allow_pickle=True).transpose()
        
    print("Formatting HMM input...")


    lengths = []
    Xs = []

    inds_sequence = t_u_inds if upper else t_d_inds

    for ts, inds in enumerate(inds_sequence): # for each timestep/sequence
        lengths.append(len(inds))             # append the length of this particular sequence
        selected_nbtype = t_nbtype[inds, ts]  # shape is (sequence length,)
        selected_nnd = t_nnd[inds, ts]        # shape is (sequence length,)
        selected_apl = t_apl[inds, ts]        # shape is (sequence length,)
        selected_psi6 = t_psi6[inds, ts]        # shape is (sequence length,)
        # selected_features = np.column_stack((selected_nbtype, selected_nnd, selected_apl)) # shape is (sequence length, 4)
        selected_features = np.column_stack((selected_nnd, selected_apl, selected_nbtype)) # shape is (sequence length, 4)
        Xs.append(selected_features)          # append the features for this particular sequence
    X = np.concatenate(Xs, axis=0)            # join the features along the sequence-related axis,
                                              # we kept track of the individual sequence lengths in `lengths`
                                              # shape of X is (nfeatures = sum(# selected features for each timestep), 4)

    X = np.nan_to_num(X)

    print("Fitting model...")

    # create the model and train it
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
    model.fit(X, lengths)

    print("Saving trained model...")

    with open(f"hmm_models/hmm_2phase_{t_date}{leaflet}.pkl", "wb") as outfile:
        pickle.dump(model, outfile)

    print("Done with training phase.")

    sto = time.time()
    
    print(f"training time taken: {sto-sta}")

else:
    with open(f"hmm_models/hmm_2phase_{t_date}{leaflet}.pkl", "rb") as infile:
        model = pickle.load(infile)
    
############################################################
## Now, use trained model to label states in other sims
############################################################

for k in range(len(traj_files)):
    if testing == True:
        # load the data for the sim
        print(f"Loading feature data from {odirs[k]}....")
        
        sta = time.time()
        
        # indices of lipids in top/bot leaflet
        try:
            t_u_inds, t_d_inds = np.load(f"{odirs[k]}/indices.pkl",allow_pickle=True,encoding='latin1')
        except:
            with open(f"{odirs[k]}/indices.pkl", 'rb') as file:
                t_u_inds, t_d_inds = pickle.load(file)
        #distance to nearest neighbor
        t_nnd = np.load("{}/nn_dists.npy".format(odirs[k]),allow_pickle=True).transpose()
        # most occuring type for nearest neighbor of each lipid
        t_nbtype = np.load("{}/nbtype.npy".format(odirs[k]),allow_pickle=True).transpose()
        t_nbtype[t_nbtype == 4] = 2
        t_nbtype[t_nbtype == 5] = 3
        #area per lipid for each lipid
        t_apl = np.load("{}/areas.npy".format(odirs[k]),allow_pickle=True).transpose()
        #hexatic order parameter
        t_psi6 = np.load("{}/op_avgs.npy".format(odirs[k]),allow_pickle=True).transpose()
        

        print("Formatting HMM input...")

        lengths = []
        Xs = []
        inds_sequence = t_u_inds if upper else t_d_inds

        for ts, inds in enumerate(inds_sequence): # for each timestep/sequence
            lengths.append(len(inds))             # append the length of this particular sequence
            selected_nbtype = t_nbtype[inds, ts]  # shape is (sequence length,)
            selected_nnd = t_nnd[inds, ts]        # shape is (sequence length,)
            selected_apl = t_apl[inds, ts]        # shape is (sequence length,)
            selected_psi6 = t_psi6[inds, ts]        # shape is (sequence length,)
            selected_features = np.column_stack((selected_nnd, selected_apl, selected_nbtype)) # shape is (sequence length, 4)
            Xs.append(selected_features)          # append the features for this particular sequence
        X = np.concatenate(Xs, axis=0)            # join the features along the sequence-related axis,
                                                  # we kept track of the individual sequence lengths in `lengths`
                                                  # shape of X is (nfeatures = sum(# selected features for each timestep), 4)
        X = np.nan_to_num(X)

        print("Labeling lipid phase using HMM...")

        # label the data according to the trained HMM
        Y = model.predict(X, lengths)

        # each element of this array is a list of the state (1 or 0) for each lipid
        # at the corresponding time index
        state_predictions = []
        sq_start = 0
        for sq_len in lengths:
            sq_end = sq_start + sq_len
            state_predictions.append(Y[sq_start:sq_end])
            sq_start = sq_end
            

        print("Saving state_predictions...")

        # np.save("{}/labels2phase{}".format(odirs[k],leaflet),state_predictions)
        with open("{}/labels2phase{}.pkl".format(odirs[k],leaflet), "wb") as outfile:
            pickle.dump(state_predictions, outfile)

        print("Done with this simulation.")
        
        sto = time.time()
        print(f"testing time taken for this trajectory: {sto-sta}")
    
    
######getting the compositions of different phases and creating visualizations
    print(f"Loading data for {date[k]} trajectory....")
    
    #plotting and visualization
    try:
        u_ind, d_ind = np.load(f"/export/home/malavikv/binary_mixtures/simulation/{date[k]}/data/indices.npy",allow_pickle=True,encoding='latin1')
    except:
        with open(f"/export/home/malavikv/binary_mixtures/simulation/{date[k]}/data//indices.pkl", 'rb') as file:
            u_ind, d_ind = pickle.load(file)
    with open(f"/export/home/malavikv/binary_mixtures/simulation/{date[k]}/data//labels2phase{leaflet}.pkl", 'rb') as file:
        labels = pickle.load(file)
    stepss = np.load(f"/export/home/malavikv/binary_mixtures/simulation/{date[k]}/data/steps.npy",allow_pickle=True,encoding='latin1')
    types = np.load(f"/export/home/malavikv/binary_mixtures/simulation/{date[k]}/data/types.npy",allow_pickle=True)[1:-1:4]
    box = np.load(f"/export/home/malavikv/binary_mixtures/simulation/{date[k]}/data/box_data.npy",allow_pickle=True)

    #function to fold coordinates into box
    def mind(dr, box_l):
        return (dr + (2.*box_l))%box_l

    tsteps = len(labels)
    typea = [2,4]
    typeb = [3,5]
    typec = 6
    #types and coordinates of all lipids in specified leaflet
    if upper==True:
        liptype = [[types[i] for i in u_ind[t]] for t in range(tsteps)]
        steps = [[stepss[t][i] for i in u_ind[t]] for t in range(tsteps)]
    else:
        liptype = [[types[i] for i in d_ind[t]] for t in range(tsteps)]
        steps = [[stepss[t][i] for i in d_ind[t]] for t in range(tsteps)]

    #x,y,z coordinates of middle beads of the lipid at index i for each time step t
    sites0a = [[(steps[t][i][1]+steps[t][i][2])/2. for i,(a,b) in enumerate(zip(liptype[t],labels[t])) if a in typea and b==0] for t in range(tsteps)]
    sites0b = [[(steps[t][i][1]+steps[t][i][2])/2. for i,(a,b) in enumerate(zip(liptype[t],labels[t])) if a in typeb and b==0] for t in range(tsteps)]
    sites0c = [[(steps[t][i][1]+steps[t][i][2])/2. for i,(a,b) in enumerate(zip(liptype[t],labels[t])) if a==typec and b==0] for t in range(tsteps)]
    sites1a = [[(steps[t][i][1]+steps[t][i][2])/2. for i,(a,b) in enumerate(zip(liptype[t],labels[t])) if a in typea and b==1] for t in range(tsteps)]
    sites1b = [[(steps[t][i][1]+steps[t][i][2])/2. for i,(a,b) in enumerate(zip(liptype[t],labels[t])) if a in typeb and b==1] for t in range(tsteps)]
    sites1c = [[(steps[t][i][1]+steps[t][i][2])/2. for i,(a,b) in enumerate(zip(liptype[t],labels[t])) if a==typec and b==1] for t in range(tsteps)]
   
    #calculating concentrations of different species in each phase
    a0_conc = np.mean(np.array([len(sites0a[i]) for i in range(len(sites0a))]))
    b0_conc = np.mean(np.array([len(sites0b[i]) for i in range(len(sites0b))]))
    c0_conc = np.mean(np.array([len(sites0c[i]) for i in range(len(sites0c))]))
    a1_conc = np.mean(np.array([len(sites1a[i]) for i in range(len(sites1a))]))
    b1_conc = np.mean(np.array([len(sites1b[i]) for i in range(len(sites1b))]))
    c1_conc = np.mean(np.array([len(sites1c[i]) for i in range(len(sites1c))]))

    tot_0 = a0_conc+b0_conc+c0_conc
    tot_1 = a1_conc+b1_conc+c1_conc
    
    print(f"{tot_0} in phase0: {round(a0_conc/tot_0,4)} unsat, {round(b0_conc/tot_0,4)} sat, {round(c0_conc/tot_0,4)} chol\
        \n {tot_1} in phase1: {round(a1_conc/tot_1,4)} unsat, {round(b1_conc/tot_1,4)} sat, {round(c1_conc/tot_1,4)} chol")
    
    if len(types)>2500:
        msize = 3 
        pixels = 300 
    else:
        msize = 6
        pixels = 100
    
    es = 1
    if makeplt==True:
        print("Now creating plots.")
        fr = -1
        
        ##plots
        fig,ax = plt.subplots(dpi=pixels)

        # ax.plot([mind(x[0],box[fr][0]) for x in sites0a[fr]],[mind(x[1],box[fr][1]) for x in sites0a[fr]],'ro',ms=msize,mfc='none',mew=es)
        ax.plot([mind(x[0],box[fr][0]) for x in sites0a[fr]],[mind(x[1],box[fr][1]) for x in sites0a[fr]],'ro',ms=msize,mfc='none',mew=es)
        ax.plot([mind(x[0],box[fr][0]) for x in sites0b[fr]],[mind(x[1],box[fr][1]) for x in sites0b[fr]],'bo',ms=msize,mfc='none',mew=es)
        ax.plot([mind(x[0],box[fr][0]) for x in sites0c[fr]],[mind(x[1],box[fr][1]) for x in sites0c[fr]],'go',ms=0.8*msize,mfc='none',mew=es)

        # ax.plot([mind(x[0],box[fr][0]) for x in sites1a[fr]],[mind(x[1],box[fr][1]) for x in sites1a[fr]],'ro',ms=msize)
        ax.plot([mind(x[0],box[fr][0]) for x in sites1a[fr]],[mind(x[1],box[fr][1]) for x in sites1a[fr]],'ro',ms=msize)
        ax.plot([mind(x[0],box[fr][0]) for x in sites1b[fr]],[mind(x[1],box[fr][1]) for x in sites1b[fr]],'bo',ms=msize)
        ax.plot([mind(x[0],box[fr][0]) for x in sites1c[fr]],[mind(x[1],box[fr][1]) for x in sites1c[fr]],'go',ms=0.8*msize)

        ax.set_aspect(1)
        
        plt.title(f"filled/unfilled=HMM class, red=unsat, blue=sat, green=chol\
        \n {tot_0} in phase0: {round(a0_conc/tot_0,2)} unsat, {round(b0_conc/tot_0,2)} sat, {round(c0_conc/tot_0,2)} chol\
        \n {tot_1} in phase1: {round(a1_conc/tot_1,2)} unsat, {round(b1_conc/tot_1,2)} sat, {round(c1_conc/tot_1,2)} chol"\
        ,fontdict={'fontsize': 8})
        
        plt.savefig(f"/export/home/malavikv/binary_mixtures/plots/ternary_hmm/{date[k]}{leaflet}_{t_date}_{n_feats}feats.png")
        # plt.savefig(f"/export/home/malavikv/binary_mixtures/plots/ternary_hmm/{date[k]}{leaflet}_lipids.pdf",bbox_inches='tight')
        
        print("Plots done.")
        plt.show()
        

    if makeanim == True:

    ###animation###
    ## ims is a list of lists, each row is a list of artists to draw in the
    ## current frame; here we are just animating one artist, the image, in
    ## each frame
        print("Now creating animation.")
        fig, ax = plt.subplots(dpi=pixels)
        ax.set_aspect(1)
        ims = []
        
        if onlyphase==True:
            plt.title("black: phase 0, yellow: phase 1")
            for fr in range(-600,-1):
                im1, = ax.plot([mind(x[0],box[fr][0]) for x in sites0a[fr]],[mind(x[1],box[fr][1]) for x in sites0a[fr]],'ko',ms=msize,mew=es)
                im2, = ax.plot([mind(x[0],box[fr][0]) for x in sites0b[fr]],[mind(x[1],box[fr][1]) for x in sites0b[fr]],'ko',ms=msize,mew=es)
                im3, = ax.plot([mind(x[0],box[fr][0]) for x in sites0c[fr]],[mind(x[1],box[fr][1]) for x in sites0c[fr]],'ko',ms=0.8*msize,mew=es)

                im4, = ax.plot([mind(x[0],box[fr][0]) for x in sites1a[fr]],[mind(x[1],box[fr][1]) for x in sites1a[fr]],'yo',ms=msize)
                im5, = ax.plot([mind(x[0],box[fr][0]) for x in sites1b[fr]],[mind(x[1],box[fr][1]) for x in sites1b[fr]],'yo',ms=msize)
                im6, = ax.plot([mind(x[0],box[fr][0]) for x in sites1c[fr]],[mind(x[1],box[fr][1]) for x in sites1c[fr]],'yo',ms=0.8*msize)
                ims.append([im1,im2,im3,im4,im5,im6])
        else:
            plt.title("filled/unfilled=HMM class, red=typeA, blue=typeB, green=typeC")
            for fr in range(-600,-1):
                im1, = ax.plot([mind(x[0],box[fr][0]) for x in sites0a[fr]],[mind(x[1],box[fr][1]) for x in sites0a[fr]],'ro',ms=msize,mfc='none',mew=es)
                im2, = ax.plot([mind(x[0],box[fr][0]) for x in sites0b[fr]],[mind(x[1],box[fr][1]) for x in sites0b[fr]],'bo',ms=msize,mfc='none',mew=es)
                im3, = ax.plot([mind(x[0],box[fr][0]) for x in sites0c[fr]],[mind(x[1],box[fr][1]) for x in sites0c[fr]],'go',ms=0.8*msize,mfc='none',mew=es)

                im4, = ax.plot([mind(x[0],box[fr][0]) for x in sites1a[fr]],[mind(x[1],box[fr][1]) for x in sites1a[fr]],'ro',ms=msize)
                im5, = ax.plot([mind(x[0],box[fr][0]) for x in sites1b[fr]],[mind(x[1],box[fr][1]) for x in sites1b[fr]],'bo',ms=msize)
                im6, = ax.plot([mind(x[0],box[fr][0]) for x in sites1c[fr]],[mind(x[1],box[fr][1]) for x in sites1c[fr]],'go',ms=0.8*msize)
                ims.append([im1,im2,im3,im4,im5,im6])
            
        ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True,
                                        repeat_delay=800)

        writer = animation.FFMpegWriter(fps=8, metadata=dict(artist='Malavika Varma'), bitrate=500)
        
        if onlyphase==True:
            ani.save(f"/export/home/malavikv/binary_mixtures/plots/ternary_hmm/hmm-OP-{date[k]}{leaflet}-{t_date}-{n_feats}.mp4",writer=writer)
        else:
            ani.save(f"/export/home/malavikv/binary_mixtures/plots/ternary_hmm/hmm-{date[k]}{leaflet}-{t_date}-{n_feats}.mp4",writer=writer)
        
        print('Animation done.')
        
        plt.show()
        
