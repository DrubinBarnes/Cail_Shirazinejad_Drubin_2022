# Cyna Shirazinejad July 15, 2020
# utilities to build KD tree from 

import numpy as np
from sklearn.neighbors import KDTree
from collections import Counter
from scipy.stats import mode
import scipy.stats as stats
from return_track_attributes import (return_track_category, return_track_lifetime, return_track_amplitude, 
                                     return_track_x_position, return_is_CCP, return_track_amplitude_one_channel,
                                     return_puncta_x_position_whole_track, return_puncta_y_position_whole_track,
                                     return_distance_traveled_from_origin, return_frames_in_track,
                                     return_distance_between_two_channel)




# build_kd_tree_channel(): takes all tracks in one channel, and builds N trees for every N frame in the movie

# input: - tracks: (can be multiple channels) from load_tracks() in cmeAnalysisPostProcessingPythonScripts/display_tracks.py
#                 e.g. imported MATLAB output from cmeAnalysis (ProcessedTracks.mat) converted to Python-compatible object

#        - channel: identity of channel to build tree from in tracks
#        - len_movie: length of movie (in frames)
#        - track_categories: categories of tracks from cmeAnalysis to use for tree construction
#
# output: - trees_per_frame: a list of KDTrees of puncta found in each frame, with a tree for every frame;
#           each index i contains a tree built from all puncta in frame i
#         - tree_vals: a list of values the tree is built from, with an additional index to designate the puncta's track identity;
#                      each index i contains a matrix (nx3) of [track_ID, x, y] for n puncta present in frame i
def build_kd_tree_channel(tracks,
                          channel,
                          len_movie,
                          track_categories):
    
    track_positions = []
    
    for i in range(len(tracks)):
        
        # only category 1 and 2 tracks for now
        if return_track_category(tracks,i) in track_categories:
            
            track_x_positions = return_puncta_x_position_whole_track(tracks,i,channel)
            track_y_positions = return_puncta_y_position_whole_track(tracks,i,channel)
            
            # make sure every tracks dimensions add up between fitted positions and frames present (may be a moot problem)
            frames = return_frames_in_track(tracks,i)-1
            
            if len(track_x_positions) != len(frames):
                print('Houston, we got a problem')
                
            # each item in track positions is [track_number, frame, frame,x_pos,y_pos]
            track_positions.append(list(zip([i for j in range(len(track_x_positions))],frames,track_x_positions, track_y_positions)))

    
    # each index i contains a tree built from all puncta in frame i
    trees_per_frame=[]
    
    # each index i contains a matrix (nx3) of [track_ID, x, y] for n puncta present in frame i
    tree_vals = []
    
    for frame in range(len_movie):

        current_tree_vals = []
        
        for track in track_positions:

            for time_point in track:
                
                # check whether frame contains track 
                if frame == time_point[1]:
                    
                    # check for erronous nan fits of x and y 
                    if time_point[2] < np.Inf and time_point[3] < np.Inf:
                        
                        # adding a list [track_ID, x, y]
                        current_tree_vals.append((time_point[0],time_point[2],time_point[3]))
                        
        current_tree_vals=np.array(current_tree_vals)
        
        tree_vals.append(current_tree_vals)
        
        # form a tree just from x y positions (excluding track idendity)
        current_tree = KDTree(current_tree_vals[:,1:3])               

        trees_per_frame.append(current_tree)
        
    return trees_per_frame, tree_vals




def associate_tracks(tracks,kd_trees,vals_tree,distance_query):
    associated_tracks = []
    frac_associated=[]
    distances=[]
    distances_per_track = []
    distance_mode_all = []
    distance_mode_per_track = []
    for i in range(len(tracks)):
        
        frames = return_frames_in_track(tracks,i)-1

        track_category = return_track_category(tracks,i)
        x_positions = return_puncta_x_position_whole_track(tracks,i,0)
        y_positions = return_puncta_y_position_whole_track(tracks,i,0)
        
        cons_order = True
        
        if cons_order:
            dist_individual_track = []
            current_track_positions = list(zip(x_positions,y_positions))
            track_associated = []
            for i,frame in enumerate(frames):
                current_tree = kd_trees[int(frame)]

                ind,dist = current_tree.query_radius(np.array(current_track_positions[i]).reshape(1,-1),r=distance_query,return_distance=True,sort_results=True)

                if ind[0].size>0:

                    track_associated.append(int(vals_tree[int(frame)][ind[0][0]][0]))
                    distances.append(dist[0][0])
                    dist_individual_track.append(dist[0][0])
                else:
                    track_associated.append(-1)
                    distances.append(-1)
                    dist_individual_track.append(-1)
                    
            distances_per_track.append(dist_individual_track)
            associated_tracks.append(track_associated)
            mode = stats.mode(track_associated)

            indices_mode = [i for i, x in enumerate(track_associated) if x == mode[0][0]]

            distance_mode_per_track.append(np.array(dist_individual_track)[indices_mode])
            for dist in np.array(dist_individual_track)[indices_mode]:
                distance_mode_all.append(dist)
            num_associations=mode[1][0]
            frac_associated.append(num_associations/len(frames))
            
    return associated_tracks, frac_associated, distances, distances_per_track, distance_mode_all, distance_mode_per_track