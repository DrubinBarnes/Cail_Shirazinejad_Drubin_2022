def filter_for_ccps(tracks_to_check,
                    minimum_dnm2_counts=3,
                    sig_pval_cutoff=0.01,
                    minimum_lifetime=20,
                    maximum_lifetime=180,
                    initial_msd_threshold=0.02,
                    pixel_size=0.103,
                    mask_for_substrate=[],
                    pixel_tolerance=1,
                    substrate_frame_overlap=5):
    
    """return tracks that are sufficiently DNM2 rich, not too long or not too short, and aren't too speedy"""
    
    tracks_kept = [] # the CCPs we're keeping
    tracks_discarded = []

    for i in tqdm(range(len(tracks_to_check))): # iterate through all tracks
        # iterate through tracks that satisfy a condition:

        # and make sure track is category 1 (valids with good gaps)
        track_lifetime = return_track_attributes.return_track_lifetime(tracks_to_check, i)

        pvals = tracks_to_check[i][index_dictionary['index_pval_Ar']][1]
        significant_pval_indices = [1 if pval < sig_pval_cutoff else 0 for pval in pvals]
        repeated_indices = [(x[0], len(list(x[1]))) for x in itertools.groupby(significant_pval_indices)]
        max_1s = 0
        for itm in repeated_indices:
            if itm[0] == 1:
                if itm[1]>max_1s:
                    max_1s=itm[1]

        max_dnm2_repeats = max_1s

        initial_msd = tracks_to_check[i][index_dictionary['index_MotionAnalysis']][0][0][1][0][0]
        initial_msd = pixel_size**2 * initial_msd
            
        if mask_for_substrate != []:
            
            mask = mask_for_substrate
            mask_nonzero = np.nonzero(mask) # the mask of interest's nonzero positions (where the pillar pixels are)
            mask_x = mask_nonzero[0] # the x positions
            mask_y = mask_nonzero[1] # the y positions
            
            track_x = return_track_attributes.return_track_x_position(tracks_to_check, i)[0] # x and y positions for every frame of the track (not including buffer)
        
            track_y = return_track_attributes.return_track_y_position(tracks_to_check, i)[0]
        
            pillar_test= test_on_pillar_loop(track_x,
                                             track_y,
                                             mask_x,
                                             mask_y,
                                             pixel_tolerance, # 1 pixel tolerance for belonging to a pixel-neighbor
                                             substrate_frame_overlap) # be associated with mask for at least 5 FRAMES 
            
            if track_lifetime >= minimum_lifetime and \
               track_lifetime <= maximum_lifetime and \
               max_dnm2_repeats >= minimum_dnm2_counts and \
               initial_msd <= initial_msd_threshold and \
               pillar_test[0] == 1:
                
                tracks_kept.append(tracks_to_check[i])
            else:
                tracks_discarded.append(tracks_to_check[i])
                
        elif track_lifetime >= minimum_lifetime and \
           track_lifetime <= maximum_lifetime and \
           max_dnm2_repeats >= minimum_dnm2_counts and \
           initial_msd <= initial_msd_threshold:
            
            
            tracks_kept.append(tracks_to_check[i])
        else:
            tracks_discarded.append(tracks_to_check[i])
    return tracks_kept, tracks_discarded