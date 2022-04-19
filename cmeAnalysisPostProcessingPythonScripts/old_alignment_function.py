def align_with_optional_scaling(tracks,ch1_color,ch2_color,selected_tracks,alignment_protocol='maxCh2',scale_to_1=False,percent_of_max = 10,padding = False,padding_amount = 0,display_channels = 'both',ylimits_plot = [-0.1,1.1]):
    # import all necessary Python libraries
    import numpy as np
    import scipy.io as sio
    import os
    import sys
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    # import functions to use for visualization/analysis
    from display_tracks import display_all_tracks, display_selected_tracks, sort_tracks_descending_lifetimes, display_one_track
    from alignment_function import align_with_optional_scaling
    from maximum_intensity_analysis import display_intensity_maxima
    from lifetime_modification import display_tracks_with_lifetimes, moving_average
    from generate_index_dictionary import return_index_dictionary
    from return_amplitudes import return_ch1_ch2_amplitudes
    number_of_tracks = len(tracks['tracks'][0])

    index_dictionary = return_index_dictionary()

    tracks = sort_tracks_descending_lifetimes(tracks)

    # unzip the tracks from the indices
    (tracks, indices) = zip(*tracks)
    ch1_amplitudes_kept = []
    ch2_amplitudes_kept = []
    for i in range(0,number_of_tracks):
        
        # check if current track is categorized as CCP and has 2 channels
        if i in selected_tracks:
            
            ch1_all_amplitudes, ch2_all_amplitudes = return_ch1_ch2_amplitudes(tracks,i)
        
            ch1_amplitudes_kept.append(ch1_all_amplitudes)
            ch2_amplitudes_kept.append(ch2_all_amplitudes)
            
    all_track_lengths = []
    index_for_alignment_protocol = []
    number_of_tracks_kept = len(ch1_amplitudes_kept)
    frames_before_peak = []
    frames_after_peak = []

    print('The number of tracks kept: ' + str(number_of_tracks_kept))



    if scale_to_1 == True: 
        
        for i in range(0,len(ch1_amplitudes_kept)):
            
            ch1_current_max = np.max(ch1_amplitudes_kept[i])
            ch2_current_max = np.max(ch2_amplitudes_kept[i])
            
            ch1_amplitudes_kept[i] = 1/ch1_current_max*ch1_amplitudes_kept[i]
            ch2_amplitudes_kept[i] = 1/ch2_current_max*ch2_amplitudes_kept[i]

    for i in range(len(ch1_amplitudes_kept)):
        all_track_lengths.append(len(ch1_amplitudes_kept[i]))

        if alignment_protocol == 'maxCh2':
            index_for_alignment_protocol.append(np.argmax(ch2_amplitudes_kept[i]))
            frames_before_peak.append(index_for_alignment_protocol[-1]-1)
            frames_after_peak.append(all_track_lengths[i]-index_for_alignment_protocol[-1])

        
        if alignment_protocol == 'maxCh1':
#             print('test')
            index_for_alignment_protocol.append(np.argmax(ch1_amplitudes_kept[i]))
            frames_before_peak.append(index_for_alignment_protocol[-1]-1)
            frames_after_peak.append(all_track_lengths[i]-index_for_alignment_protocol[-1])

        elif alignment_protocol == 'NpercentCh1PriorToPeak':

            index_of_maximum_ch1 = np.argmax(ch1_amplitudes_kept[i])
            value_fraction = ch1_amplitudes_kept[i][index_of_maximum_ch1]*percent_of_max/100
            array_current = np.asarray(ch1_amplitudes_kept[i])
            # print(ch1_amplitudes_kept[i])
            # print(index_of_maximum_ch1)
            # print(value_fraction)
            # print(array_current)
            if index_of_maximum_ch1 == 0:
                idx = 0
            else:
                idx = (np.abs(array_current[0:index_of_maximum_ch1] - value_fraction)).argmin()
            index_for_alignment_protocol.append(idx)
            frames_before_peak.append(index_for_alignment_protocol[-1]-1)
            frames_after_peak.append(all_track_lengths[i]-index_for_alignment_protocol[-1])

        elif alignment_protocol == 'NpercentCh2PriorToPeak':

            index_of_maximum_ch2 = np.argmax(ch2_amplitudes_kept[i])
            value_fraction = ch2_amplitudes_kept[i][index_of_maximum_ch2]*percent_of_max/100
            array_current = np.asarray(ch2_amplitudes_kept[i])
            if index_of_maximum_ch2 == 0:
                idx = 0
            else:
                idx = (np.abs(array_current[0:index_of_maximum_ch2] - value_fraction)).argmin()
            index_for_alignment_protocol.append(idx)
            frames_before_peak.append(index_for_alignment_protocol[-1]-1)
            frames_after_peak.append(all_track_lengths[i]-index_for_alignment_protocol[-1])

        elif alignment_protocol == 'NpercentCh1FollowingPeak':

            index_of_maximum_ch1 = np.argmax(ch1_amplitudes_kept[i])
            value_fraction = ch1_amplitudes_kept[i][index_of_maximum_ch1]*percent_of_max/100
            array_current = np.asarray(ch2_amplitudes_kept[i])
            if index_of_maximum_ch1 == 0:
                idx = 0
            else:
                idx = (np.abs(array_current[index_of_maximum_ch1:] - value_fraction)).argmin()
            index_for_alignment_protocol.append(idx+index_of_maximum_ch1)
            frames_before_peak.append(index_for_alignment_protocol[-1]-1)
            frames_after_peak.append(all_track_lengths[i]-index_for_alignment_protocol[-1])

        elif alignment_protocol == 'NpercentCh2FollowingPeak':

            index_of_maximum_ch2 = np.argmax(ch2_amplitudes_kept[i])
            value_fraction = ch2_amplitudes_kept[i][index_of_maximum_ch2]*percent_of_max/100
            array_current = np.asarray(ch2_amplitudes_kept[i])
            if index_of_maximum_ch2 == 0:
                idx = 0
            else:
                idx = (np.abs(array_current[index_of_maximum_ch2:] - value_fraction)).argmin()
            index_for_alignment_protocol.append(idx+index_of_maximum_ch2)
            frames_before_peak.append(index_for_alignment_protocol[-1]-1)
            frames_after_peak.append(all_track_lengths[i]-index_for_alignment_protocol[-1])
#     print(frames_before_peak)           


    maximum_frames_before_peak = np.max(frames_before_peak)
    maximum_frames_after_peak = np.max(frames_after_peak)   
        
    new_aligned_ch1_amplitudes = []
    new_aligned_ch2_amplitudes = []


    # max_frames_before = np.max(frames_before_peak)
    # max_frames_after = np.max(frames_after_peak)
    print('the maximum number of frames padded before aligned point: ' + str(maximum_frames_before_peak))
    print('the maximum number of frames padded after aligned point: ' + str(maximum_frames_after_peak))

    if padding and (maximum_frames_before_peak > padding_amount or maximum_frames_after_peak > padding_amount):

        raise Exception('the number of padded frames to center your aligned feature is less than the amount of default padded frames. increase the padded value.')


    for i in range(len(ch1_amplitudes_kept)):

        current_ch1_amplitudes = ch1_amplitudes_kept[i]
        current_ch2_amplitudes = ch2_amplitudes_kept[i]
        curr_frames_before_peak = frames_before_peak[i]
        curr_frames_after_peak = frames_after_peak[i]
        
        if padding:

            frames_needed_before = padding_amount - curr_frames_before_peak
            frames_needed_after = padding_amount - curr_frames_after_peak


            vector_before = np.zeros(frames_needed_before)
            vector_after = np.zeros(frames_needed_after)
            new_vector = np.concatenate((vector_before, current_ch2_amplitudes, vector_after), axis = 0)
            new_aligned_ch2_amplitudes.append(new_vector)
            new_vector = np.concatenate((vector_before, current_ch1_amplitudes, vector_after), axis = 0)
            new_aligned_ch1_amplitudes.append(new_vector)
            
        else:
            if curr_frames_before_peak < maximum_frames_before_peak and curr_frames_after_peak < maximum_frames_after_peak:

                frames_needed_before = maximum_frames_before_peak - curr_frames_before_peak
                frames_needed_after = maximum_frames_after_peak - curr_frames_after_peak


                vector_before = np.zeros(frames_needed_before)
                vector_after = np.zeros(frames_needed_after)
                new_vector = np.concatenate((vector_before, current_ch2_amplitudes, vector_after), axis = 0)
                new_aligned_ch2_amplitudes.append(new_vector)
                new_vector = np.concatenate((vector_before, current_ch1_amplitudes, vector_after), axis = 0)
                new_aligned_ch1_amplitudes.append(new_vector)

            elif curr_frames_before_peak < maximum_frames_before_peak and curr_frames_after_peak >= maximum_frames_after_peak:

                frames_needed_before = maximum_frames_before_peak - curr_frames_before_peak

                # frames_needed_before = maximum_frames_before_peak - curr_frames_before_peak
                vector_before = np.zeros(frames_needed_before)
                new_vector = np.concatenate((vector_before, current_ch2_amplitudes), axis = 0)
                new_aligned_ch2_amplitudes.append(new_vector)
                new_vector = np.concatenate((vector_before, current_ch1_amplitudes), axis = 0)
                new_aligned_ch1_amplitudes.append(new_vector)

            elif curr_frames_before_peak >= maximum_frames_before_peak and curr_frames_after_peak < maximum_frames_after_peak:

                # frames_needed_before = maximum_frames_before_peak - curr_frames_before_peak
                frames_needed_after = maximum_frames_after_peak - curr_frames_after_peak


                # frames_needed_after = maximum_frames_after_peak - curr_frames_after_peak
                vector_after = np.zeros(frames_needed_after)
                new_vector = np.concatenate((current_ch2_amplitudes, vector_after), axis = 0)
                new_aligned_ch2_amplitudes.append(new_vector)
                new_vector = np.concatenate((current_ch1_amplitudes, vector_after), axis = 0)
                new_aligned_ch1_amplitudes.append(new_vector)

            else:

                new_aligned_ch2_amplitudes.append(current_ch2_amplitudes)
                new_aligned_ch1_amplitudes.append(current_ch1_amplitudes)
        

#     print(new_aligned_ch1_amplitudes)
    ch1np = np.asarray(new_aligned_ch1_amplitudes, dtype=np.float32)
    ch2np = np.asarray(new_aligned_ch2_amplitudes, dtype=np.float32)    
    
    ch1mean = np.mean(ch1np, axis = 0, dtype=np.float64)
    ch1std = np.std(ch1np, axis = 0, dtype=np.float64)

    ch2mean = np.mean(ch2np, axis = 0, dtype=np.float64)
    ch2std = np.std(ch2np, axis = 0, dtype=np.float64)
    
    figure(num=None, figsize=(6, 4), dpi=1000, facecolor='w', edgecolor='k')
    
    if display_channels == 'both':
        if padding:

            plt.plot(range(-padding_amount,padding_amount+1),ch1mean,ch1_color)
            plt.plot(range(-padding_amount,padding_amount+1),ch2mean,ch2_color)

            plt.fill_between(range(-padding_amount,padding_amount+1), ch1mean-ch1std, ch1mean+ch1std,color=ch1_color,alpha=0.2)
            plt.fill_between(range(-padding_amount,padding_amount+1), ch2mean-ch2std, ch2mean+ch2std,color=ch2_color,alpha=0.2)

        else:

            plt.plot(ch1mean,ch1_color)
            plt.plot(ch2mean,ch2_color)

            plt.fill_between(np.arange(0,len(ch1mean)), ch1mean-ch1std, ch1mean+ch1std,color=ch1_color,alpha=0.2)
            plt.fill_between(np.arange(0,len(ch2mean)), ch2mean-ch2std, ch2mean+ch2std,color=ch2_color,alpha=0.2)
    elif display_channels == 'ch1':
        if padding:

            plt.plot(range(-padding_amount,padding_amount+1),ch1mean,ch1_color)

            plt.fill_between(range(-padding_amount,padding_amount+1), ch1mean-ch1std, ch1mean+ch1std,color=ch1_color,alpha=0.2)

        else:

            plt.plot(ch1mean,ch1_color)

            plt.fill_between(np.arange(0,len(ch1mean)), ch1mean-ch1std, ch1mean+ch1std,color=ch1_color,alpha=0.2)
    elif display_channels == 'ch2':

        if padding:

            plt.plot(range(-padding_amount,padding_amount+1),ch2mean,ch2_color)

            plt.fill_between(range(-padding_amount,padding_amount+1), ch2mean-ch2std, ch2mean+ch2std,color=ch2_color,alpha=0.2)

        else:

            plt.plot(ch2mean,ch2_color)

            plt.fill_between(np.arange(0,len(ch2mean)), ch2mean-ch2std, ch2mean+ch2std,color=ch2_color,alpha=0.2)
            
    else:
        raise Exception('options for "display_channels": "both" (default), "ch1","ch2"')
    plt.title('aligned with ' +  str(alignment_protocol) + ', mean amplitudes +/- 1 std')
    plt.xlabel('frames')
    plt.ylabel('au intensity')
    plt.ylim(ylimits_plot)
    plt.show()

def plot_distance_fluctuations(tracks,ch1_color,ch2_color,selected_tracks,alignment_protocol = 'maxCh2',percent_of_max = 10,padding_amount = 0,display_channels = 'both',ylimits_plot = [-0.1,1.1],display_amplitudes = False,show_individual=True,show_average=True,print_diffusion = False):
    
    # import all necessary Python libraries
    import numpy as np
    import scipy.io as sio
    import os
    import sys
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    # allow lots of figures
    plt.rcParams.update({'figure.max_open_warning': 0})
    # the Python scripts added to the local path
    # sys.path.append(os.path.abspath(os.getcwd()+'/cmeAnalysisPostProcessingPythonScripts'))
    pythonPackagePath = os.path.abspath(os.getcwd())
    sys.path.append(pythonPackagePath)

    # import functions to use for visualization/analysis
    from display_tracks import display_all_tracks, display_selected_tracks, sort_tracks_descending_lifetimes, display_one_track
    from alignment_function import align_with_optional_scaling
    from maximum_intensity_analysis import display_intensity_maxima
    from lifetime_modification import display_tracks_with_lifetimes, moving_average
    from generate_index_dictionary import return_index_dictionary
    from return_amplitudes import return_ch1_ch2_amplitudes
    
#     from display_tracks import return_ch1_ch2_amplitudes_test
    number_of_tracks = len(tracks['tracks'][0])

    index_dictionary = return_index_dictionary()

    tracks = sort_tracks_descending_lifetimes(tracks)

    # unzip the tracks from the indices
    (tracks, indices) = zip(*tracks)

    distance_moved_ch1 = [] 
    distance_moved_std_ch1 = []
    distance_moved_ch2 = [] 
    distance_moved_std_ch2 = []    
    
    ch1_amplitudes_kept = []
    ch2_amplitudes_kept = []
    
    diffusion_constant_ch1 = []
    diffusion_constant_ch2 = []
    figure(num=None, figsize=(6, 4), dpi=500, facecolor='w', edgecolor='k')

    for i in range(0,number_of_tracks):

        if i in selected_tracks:

            ch1_all_amplitudes = tracks[i][index_dictionary['index_amplitude']][0]
            ch2_all_amplitudes = tracks[i][index_dictionary['index_amplitude']][1]
        
            ch1_amplitudes_kept.append(ch1_all_amplitudes)
            ch2_amplitudes_kept.append(ch2_all_amplitudes)
                        
            dist_temp_ch1 = [0]
            dist_std_temp_ch1 = [0]
            
            dist_temp_ch2 = [0]
            dist_std_temp_ch2 = [0]
            
            x_pos_ch1 = tracks[i][index_dictionary['index_x_pos']][0]
            y_pos_ch1 = tracks[i][index_dictionary['index_y_pos']][0]
            x_std_ch1 = tracks[i][index_dictionary['index_x_pos_pstd']][0]
            y_std_ch1 = tracks[i][index_dictionary['index_y_pos_pstd']][0]
            
            x_pos_ch2 = tracks[i][index_dictionary['index_x_pos']][1]
            y_pos_ch2 = tracks[i][index_dictionary['index_y_pos']][1]
            x_std_ch2 = tracks[i][index_dictionary['index_x_pos_pstd']][1]
            y_std_ch2 = tracks[i][index_dictionary['index_y_pos_pstd']][1]
            

            displacements_from_origin_ch1 = []
            displacements_from_origin_ch2 = []

            for i in range(1,len(x_pos_ch1)):

                displacements_from_origin_ch1.append((x_pos_ch1[i]-x_pos_ch1[0])**2 + (y_pos_ch1[i]-y_pos_ch1[0])**2)
                displacements_from_origin_ch2.append((x_pos_ch2[i]-x_pos_ch2[0])**2 + (y_pos_ch2[i]-y_pos_ch2[0])**2)

            ch1_mean_displacement = np.mean(displacements_from_origin_ch1)
            ch2_mean_displacement = np.mean(displacements_from_origin_ch2)

            diffusion_constant_ch1.append(ch1_mean_displacement/4/tracks[i][index_dictionary['index_lifetime_s']][0][0])
            diffusion_constant_ch2.append(ch2_mean_displacement/4/tracks[i][index_dictionary['index_lifetime_s']][0][0])

            for j in range(1,len(x_pos_ch1)):
                
                delta_x_ch1 = x_pos_ch1[j]-x_pos_ch1[j-1]
                delta_y_ch1 = y_pos_ch1[j]-y_pos_ch1[j-1]
                
                dist_temp_ch1.append(np.sqrt(delta_x_ch1**2 + delta_y_ch1**2))
                dist_std_temp_ch1.append(1/dist_temp_ch1[-1]*np.sqrt( delta_x_ch1**2 * (x_std_ch1[j]**2+x_std_ch1[j-1]**2) + delta_y_ch1**2 * (y_std_ch1[j]**2+y_std_ch1[j-1]**2) ))
                      
                    
                delta_x_ch2 = x_pos_ch2[j]-x_pos_ch2[j-1]
                delta_y_ch2 = y_pos_ch2[j]-y_pos_ch2[j-1]
                
                dist_temp_ch2.append(np.sqrt(delta_x_ch2**2 + delta_y_ch2**2))
                dist_std_temp_ch2.append(1/dist_temp_ch2[-1]*np.sqrt( delta_x_ch2**2 * (x_std_ch2[j]**2+x_std_ch2[j-1]**2) + delta_y_ch2**2 * (y_std_ch2[j]**2+y_std_ch2[j-1]**2) ))
            
            if show_individual:
                print('The TrackID of the following track is: ' + str(i))

                if display_channels == 'both':

                    bottom_limit_ch1 = [a_i - b_i for a_i, b_i in zip(dist_temp_ch1, dist_std_temp_ch1)]
                    top_limit_ch1 = [a_i + b_i for a_i, b_i in zip(dist_temp_ch1, dist_std_temp_ch1)]

                    plt.plot(dist_temp_ch1,color=ch1_color)
                    plt.fill_between(range(len(bottom_limit_ch1)),bottom_limit_ch1,top_limit_ch1,alpha=0.2,color=ch1_color)

                    dist_temp_np_ch1 = np.asarray(dist_temp_ch1)
                    dist_temp_std_np_ch1 = np.asarray(dist_std_temp_ch1)

                    dist_temp_np_nan_idx_ch1 = np.argwhere(np.isnan(dist_temp_np_ch1))
                    dist_temp_std_np_nan_idx_ch1 = np.argwhere(np.isnan(dist_temp_std_np_ch1))
          
                    nan_dist_ch1 = []
                    for i in dist_temp_np_nan_idx_ch1:
                        nan_dist_ch1.append(i[0])
                    nan_dist_std_ch1 = []
                    for i in dist_temp_std_np_nan_idx_ch1:
                        nan_dist_std_ch1.append(i[0])

                    plt.plot(nan_dist_std_ch1,[dist_temp_ch1[k] for k in nan_dist_std_ch1],'s',color=ch1_color,label='undefined std')

                    bottom_limit_ch2 = [a_i - b_i for a_i, b_i in zip(dist_temp_ch2, dist_std_temp_ch2)]
                    top_limit_ch2 = [a_i + b_i for a_i, b_i in zip(dist_temp_ch2, dist_std_temp_ch2)]

                    plt.plot(dist_temp_ch2,color=ch2_color)
                    plt.fill_between(range(len(bottom_limit_ch2)),bottom_limit_ch2,top_limit_ch2,alpha=0.2,color=ch2_color)

                    dist_temp_np_ch2 = np.asarray(dist_temp_ch2)
                    dist_temp_std_np_ch2 = np.asarray(dist_std_temp_ch2)

                    dist_temp_np_nan_idx_ch2 = np.argwhere(np.isnan(dist_temp_np_ch2))
                    dist_temp_std_np_nan_idx_ch2 = np.argwhere(np.isnan(dist_temp_std_np_ch2))
          
                    nan_dist_ch2 = []
                    for i in dist_temp_np_nan_idx_ch2:
                        nan_dist_ch2.append(i[0])
                    nan_dist_std_ch2 = []
                    for i in dist_temp_std_np_nan_idx_ch2:
                        nan_dist_std_ch2.append(i[0])

                    plt.plot(nan_dist_std_ch2,[dist_temp_ch2[k] for k in nan_dist_std_ch2],'s',color=ch2_color,label='undefined std')

                    plt.legend()
                    plt.xlabel('frames')
                    plt.ylabel('distance moved')

                    plt.show()


                elif display_channels == 'ch1':

                    bottom_limit_ch1 = [a_i - b_i for a_i, b_i in zip(dist_temp_ch1, dist_std_temp_ch1)]
                    top_limit_ch1 = [a_i + b_i for a_i, b_i in zip(dist_temp_ch1, dist_std_temp_ch1)]

                    plt.plot(dist_temp_ch1,color=ch1_color)
                    plt.fill_between(range(len(bottom_limit_ch1)),bottom_limit_ch1,top_limit_ch1,alpha=0.2,color=ch1_color)

                    dist_temp_np_ch1 = np.asarray(dist_temp_ch1)
                    dist_temp_std_np_ch1 = np.asarray(dist_std_temp_ch1)

                    dist_temp_np_nan_idx_ch1 = np.argwhere(np.isnan(dist_temp_np_ch1))
                    dist_temp_std_np_nan_idx_ch1 = np.argwhere(np.isnan(dist_temp_std_np_ch1))
          
                    nan_dist_ch1 = []
                    for i in dist_temp_np_nan_idx_ch1:
                        nan_dist_ch1.append(i[0])
                    nan_dist_std_ch1 = []
                    for i in dist_temp_std_np_nan_idx_ch1:
                        nan_dist_std_ch1.append(i[0])

                    plt.plot(nan_dist_std_ch1,[dist_temp_ch1[k] for k in nan_dist_std_ch1],'s',color=ch1_color,label='undefined std')



                    plt.legend()
                    plt.xlabel('frames')
                    plt.ylabel('distance moved')

                    plt.show()

                elif display_channels == 'ch2':
                    

                    bottom_limit_ch2 = [a_i - b_i for a_i, b_i in zip(dist_temp_ch2, dist_std_temp_ch2)]
                    top_limit_ch2 = [a_i + b_i for a_i, b_i in zip(dist_temp_ch2, dist_std_temp_ch2)]

                    plt.plot(dist_temp_ch2,color=ch2_color)
                    plt.fill_between(range(len(bottom_limit_ch2)),bottom_limit_ch2,top_limit_ch2,alpha=0.2,color=ch2_color)

                    dist_temp_np_ch2 = np.asarray(dist_temp_ch2)
                    dist_temp_std_np_ch2 = np.asarray(dist_std_temp_ch2)

                    dist_temp_np_nan_idx_ch2 = np.argwhere(np.isnan(dist_temp_np_ch2))
                    dist_temp_std_np_nan_idx_ch2 = np.argwhere(np.isnan(dist_temp_std_np_ch2))
          
                    nan_dist_ch2 = []
                    for i in dist_temp_np_nan_idx_ch2:
                        nan_dist_ch2.append(i[0])
                    nan_dist_std_ch2 = []
                    for i in dist_temp_std_np_nan_idx_ch2:
                        nan_dist_std_ch2.append(i[0])

                    plt.plot(nan_dist_std_ch2,[dist_temp_ch2[k] for k in nan_dist_std_ch2],'s',color=ch2_color,label='undefined std')
                    plt.legend()



                    plt.xlabel('frames')
                    plt.ylabel('distance moved')

                    plt.show()


                
            distance_moved_ch1.append(dist_temp_ch1)
            distance_moved_std_ch1.append(dist_std_temp_ch1)
            
            distance_moved_ch2.append(dist_temp_ch2)
            distance_moved_std_ch2.append(dist_std_temp_ch2)
    
    if print_diffusion==True:

        print('ch1 diffusion constants:')
        print(diffusion_constant_ch1)
        print()
        print('ch2 diffusion constants:')
        print(diffusion_constant_ch2)
        print()

                
    if show_average:

        for i in range(0,len(ch1_amplitudes_kept)):

            ch1_current_max = np.max(ch1_amplitudes_kept[i])
            ch2_current_max = np.max(ch2_amplitudes_kept[i])

            ch1_amplitudes_kept[i] = 1/ch1_current_max*ch1_amplitudes_kept[i]
            ch2_amplitudes_kept[i] = 1/ch2_current_max*ch2_amplitudes_kept[i]
            
            
        all_track_lengths = []
        index_for_alignment_protocol = []
        number_of_tracks_kept = len(ch1_amplitudes_kept)
        frames_before_peak = []
        frames_after_peak = []
        
        print('The number of tracks kept: ' + str(number_of_tracks_kept))

        for i in range(len(ch1_amplitudes_kept)):
            all_track_lengths.append(len(ch1_amplitudes_kept[i]))

            if alignment_protocol == 'maxCh2':
                index_for_alignment_protocol.append(np.argmax(ch2_amplitudes_kept[i]))
                frames_before_peak.append(index_for_alignment_protocol[-1]-1)
                frames_after_peak.append(all_track_lengths[i]-index_for_alignment_protocol[-1])


            if alignment_protocol == 'maxCh1':
                index_for_alignment_protocol.append(np.argmax(ch1_amplitudes_kept[i]))
                frames_before_peak.append(index_for_alignment_protocol[-1]-1)
                frames_after_peak.append(all_track_lengths[i]-index_for_alignment_protocol[-1])

            elif alignment_protocol == 'NpercentCh1PriorToPeak':

                index_of_maximum_ch1 = np.argmax(ch1_amplitudes_kept[i])
                value_fraction = ch1_amplitudes_kept[i][index_of_maximum_ch1]*percent_of_max/100
                array_current = np.asarray(ch1_amplitudes_kept[i])

                if index_of_maximum_ch1 == 0:
                    idx = 0
                else:
                    idx = (np.abs(array_current[0:index_of_maximum_ch1] - value_fraction)).argmin()
                index_for_alignment_protocol.append(idx)
                frames_before_peak.append(index_for_alignment_protocol[-1]-1)
                frames_after_peak.append(all_track_lengths[i]-index_for_alignment_protocol[-1])

            elif alignment_protocol == 'NpercentCh2PriorToPeak':

                index_of_maximum_ch2 = np.argmax(ch2_amplitudes_kept[i])
                value_fraction = ch2_amplitudes_kept[i][index_of_maximum_ch2]*percent_of_max/100
                array_current = np.asarray(ch2_amplitudes_kept[i])
                if index_of_maximum_ch2 == 0:
                    idx = 0
                else:
                    idx = (np.abs(array_current[0:index_of_maximum_ch2] - value_fraction)).argmin()
                index_for_alignment_protocol.append(idx)
                frames_before_peak.append(index_for_alignment_protocol[-1]-1)
                frames_after_peak.append(all_track_lengths[i]-index_for_alignment_protocol[-1])

            elif alignment_protocol == 'NpercentCh1FollowingPeak':

                index_of_maximum_ch1 = np.argmax(ch1_amplitudes_kept[i])
                value_fraction = ch1_amplitudes_kept[i][index_of_maximum_ch1]*percent_of_max/100
                array_current = np.asarray(ch2_amplitudes_kept[i])
                if index_of_maximum_ch1 == 0:
                    idx = 0
                else:
                    idx = (np.abs(array_current[index_of_maximum_ch1:] - value_fraction)).argmin()
                index_for_alignment_protocol.append(idx+index_of_maximum_ch1)
                frames_before_peak.append(index_for_alignment_protocol[-1]-1)
                frames_after_peak.append(all_track_lengths[i]-index_for_alignment_protocol[-1])

            elif alignment_protocol == 'NpercentCh2FollowingPeak':

                index_of_maximum_ch2 = np.argmax(ch2_amplitudes_kept[i])
                value_fraction = ch2_amplitudes_kept[i][index_of_maximum_ch2]*percent_of_max/100
                array_current = np.asarray(ch2_amplitudes_kept[i])
                if index_of_maximum_ch2 == 0:
                    idx = 0
                else:
                    idx = (np.abs(array_current[index_of_maximum_ch2:] - value_fraction)).argmin()
                index_for_alignment_protocol.append(idx+index_of_maximum_ch2)
                frames_before_peak.append(index_for_alignment_protocol[-1]-1)
                frames_after_peak.append(all_track_lengths[i]-index_for_alignment_protocol[-1])


        maximum_frames_before_peak = np.max(frames_before_peak)
        maximum_frames_after_peak = np.max(frames_after_peak)   
            
        new_aligned_ch1_amplitudes = []
        new_aligned_ch2_amplitudes = []
        
        new_aligned_ch1_distances = []
        new_aligned_ch2_distances = []
        
        print('the number of frames padded before aligned point: ' + str(maximum_frames_before_peak))
        print('the number of frames padded after aligned point: ' + str(maximum_frames_after_peak))

        if (maximum_frames_before_peak > padding_amount or maximum_frames_after_peak > padding_amount):

            raise Exception('the number of padded frames to center your aligned feature is less than the amount of default padded frames. increase the padded value.')

            
            
        for i in range(len(ch1_amplitudes_kept)):

            current_ch1_amplitudes = ch1_amplitudes_kept[i]
            current_ch2_amplitudes = ch2_amplitudes_kept[i]
            curr_frames_before_peak = frames_before_peak[i]
            curr_frames_after_peak = frames_after_peak[i]
            
            current_distance_moved_ch1 = distance_moved_ch1[i]
            current_distance_moved_ch2 = distance_moved_ch2[i]
            frames_needed_before = padding_amount - curr_frames_before_peak
            frames_needed_after = padding_amount - curr_frames_after_peak

            vector_before = np.full(frames_needed_before,np.nan)
            vector_after = np.full(frames_needed_after,np.nan)


            new_vector = np.concatenate((vector_before, current_ch2_amplitudes, vector_after), axis = 0)
            new_aligned_ch2_amplitudes.append(new_vector)
            
            new_vector = np.concatenate((vector_before, current_ch1_amplitudes, vector_after), axis = 0)
            new_aligned_ch1_amplitudes.append(new_vector)
                
            new_vector = np.concatenate((vector_before, current_distance_moved_ch1, vector_after), axis = 0)
            new_aligned_ch1_distances.append(new_vector)
            
            new_vector = np.concatenate((vector_before, current_distance_moved_ch2, vector_after), axis = 0)
            new_aligned_ch2_distances.append(new_vector)

            
        
        ch1np = np.asarray(new_aligned_ch1_amplitudes, dtype=np.float64)
        ch2np = np.asarray(new_aligned_ch2_amplitudes, dtype=np.float64)    
        
        dist_ch1np = np.asarray(new_aligned_ch1_distances, dtype=np.float64)
        dist_ch2np = np.asarray(new_aligned_ch2_distances, dtype=np.float64)    


        ch1mean = np.nan_to_num(np.nanmean(ch1np,axis=0,dtype=np.float64))
        ch2mean = np.nan_to_num(np.nanmean(ch2np,axis=0,dtype=np.float64))
        ch1std = np.nan_to_num(np.nanstd(ch1np,axis=0,dtype=np.float64))
        ch2std = np.nan_to_num(np.nanstd(ch2np,axis=0,dtype=np.float64))

        dist_ch1mean = np.nan_to_num(np.nanmean(dist_ch1np,axis=0,dtype=np.float64))
        dist_ch2mean = np.nan_to_num(np.nanmean(dist_ch2np,axis=0,dtype=np.float64))
        dist_ch1std = np.nan_to_num(np.nanmean(dist_ch1np,axis=0,dtype=np.float64))
        dist_ch2std = np.nan_to_num(np.nanmean(dist_ch2np,axis=0,dtype=np.float64))


        
        if display_channels == 'both':
            
            if display_amplitudes:
                print('test')
                figure(num=None, figsize=(6, 4), dpi=1000, facecolor='w', edgecolor='k')
                print('last test')
                figure, ax1 = plt.subplots(figsize=(15,5))
                ax1.set_ylim(ylimits_plot)
                color = 'k'
                ax1.set_xlabel('frames')
                ax1.set_ylabel('au fluorescence',color=color)
                
                p1,=ax1.plot(range(-padding_amount,padding_amount+1), ch1mean, color= ch1_color,label='ch1 amplitude')
                p2,=ax1.plot(range(-padding_amount,padding_amount+1), ch2mean, color= ch2_color,label='ch2 amplitude')
                plt.fill_between(range(-padding_amount,padding_amount+1), ch1mean-0.25*ch1std, ch1mean+0.25*ch1std,color=ch1_color,alpha=0.2)
                plt.fill_between(range(-padding_amount,padding_amount+1), ch2mean-0.25*ch2std, ch2mean+0.25*ch2std,color=ch2_color,alpha=0.2)
                ax1.title.set_text('aligned with ' +  str(alignment_protocol) + ', mean +/- 1/4 std')

                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'k'
                ax2.set_ylabel('distance between frames', color=color)  # we already handled the x-label with ax1
                
                p3,=ax2.plot(range(-padding_amount,padding_amount+1), dist_ch1mean, color= ch1_color,marker = 'o',label='ch1 distance',markersize=3)
                p4,=ax2.plot(range(-padding_amount,padding_amount+1), dist_ch2mean, color= ch2_color,marker = 'o',label='ch2 distance',markersize=3)
                ax2.fill_between(range(-padding_amount,padding_amount+1), dist_ch1mean-0.25*dist_ch1std, dist_ch1mean+0.25*dist_ch1std,color=ch1_color,alpha=0.2)
                ax2.fill_between(range(-padding_amount,padding_amount+1), dist_ch2mean-0.25*dist_ch2std, dist_ch2mean+0.25*dist_ch2std,color=ch2_color,alpha=0.2)
               
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim(ylimits_plot)                

                lines = [p1, p2, p3, p4]
                ax1.legend(lines, [l.get_label() for l in lines])
                figure.savefig('filename.png', dpi=1000)

                figure.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()
                
            else:
                figure(num=None, figsize=(6, 4), dpi=1000, facecolor='w', edgecolor='k')
                plt.plot(range(-padding_amount,padding_amount+1), dist_ch1mean, color= ch1_color)
                plt.plot(range(-padding_amount,padding_amount+1), dist_ch2mean, color= ch2_color)
                plt.fill_between(range(-padding_amount,padding_amount+1), dist_ch1mean-0.25*dist_ch1std, dist_ch1mean+0.25*dist_ch1std,color=ch1_color,alpha=0.2)
                plt.fill_between(range(-padding_amount,padding_amount+1), dist_ch2mean-0.25*dist_ch2std, dist_ch2mean+0.25*dist_ch2std,color=ch2_color,alpha=0.2)
                plt.ylim(ylimits_plot)
                plt.title('aligned with ' +  str(alignment_protocol) + ', mean +/- 1/4 std')
                plt.xlabel('frames')
                plt.ylabel('distance moved')
                plt.ylim(ylimits_plot)
                plt.show()
                
        elif display_channels == 'ch1':
            
            if display_amplitudes:

                figure, ax1 = plt.subplots(figsize=(15,5))
                ax1.set_ylim(ylimits_plot)
                
                color = 'k'
                ax1.set_xlabel('frames')
                ax1.set_ylabel('au fluorescence',color=color)
                
                p1,=ax1.plot(range(-padding_amount,padding_amount+1), ch1mean, color= ch1_color,label='ch1 amplitude')
                plt.fill_between(range(-padding_amount,padding_amount+1), ch1mean-0.25*ch1std, ch1mean+0.25*ch1std,color=ch1_color,alpha=0.2)
    #             ax1.title('aligned with ' +  str(alignment_protocol) + ', mean +/- 1/4 std')
                ax1.title.set_text('aligned with ' +  str(alignment_protocol) + ', mean +/- 1/4 std')
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'k'
                ax2.set_ylabel('distance between frames', color=color)  # we already handled the x-label with ax1
                
                p2,=ax2.plot(range(-padding_amount,padding_amount+1), dist_ch1mean, color= ch1_color,marker = 'o',label='ch1 distance',markersize=3)
                ax2.fill_between(range(-padding_amount,padding_amount+1), dist_ch1mean-0.25*dist_ch1std, dist_ch1mean+0.25*dist_ch1std,color=ch1_color,alpha=0.2)
                lines = [p1, p2]
                ax1.legend(lines, [l.get_label() for l in lines])
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim(ylimits_plot)
                figure.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()
            else:
                figure(num=None, figsize=(6, 4), dpi=1000, facecolor='w', edgecolor='k')
                plt.plot(range(-padding_amount,padding_amount+1), dist_ch1mean, color= ch1_color)
                plt.fill_between(range(-padding_amount,padding_amount+1), dist_ch1mean-0.25*dist_ch1std, dist_ch1mean+0.25*dist_ch1std,color=ch1_color,alpha=0.2)
                plt.ylim(ylimits_plot)           
                plt.title('aligned with ' +  str(alignment_protocol) + ', mean +/- 1/4 std')
                plt.xlabel('frames')
                plt.ylabel('distance moved')
                plt.ylim(ylimits_plot)
                plt.show()
                
        elif display_channels == 'ch2':

            if display_amplitudes:

                figure, ax1 = plt.subplots(figsize=(15,5))
                ax1.set_ylim(ylimits_plot)
                
                color = 'k'
                ax1.set_xlabel('frames')
                ax1.set_ylabel('au fluorescence',color=color)
                
                p1,=ax1.plot(range(-padding_amount,padding_amount+1), ch2mean, color= ch2_color,label='ch2 amplitude')
                plt.fill_between(range(-padding_amount,padding_amount+1), ch2mean-0.25*ch2std, ch2mean+0.25*ch2std,color=ch2_color,alpha=0.2)

                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                ax1.title.set_text('aligned with ' +  str(alignment_protocol) + ', mean +/- 1/4 std')

                color = 'k'
                ax2.set_ylabel('distance between frames', color=color)  # we already handled the x-label with ax1
                
                p2,=ax2.plot(range(-padding_amount,padding_amount+1), dist_ch2mean, color= ch2_color,marker = 'o',label='ch2 distance',markersize=3)
                ax2.fill_between(range(-padding_amount,padding_amount+1), dist_ch2mean-0.25*dist_ch2std, dist_ch2mean+0.25*dist_ch2std,color=ch2_color,alpha=0.2)
                lines = [p1, p2]
                ax1.legend(lines, [l.get_label() for l in lines])
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim(ylimits_plot)
                figure.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()
            
            else:

                figure(num=None, figsize=(6,4), dpi=1000, facecolor='w', edgecolor='k')
                plt.plot(range(-padding_amount,padding_amount+1), dist_ch2mean, color= ch2_color)
                plt.fill_between(range(-padding_amount,padding_amount+1), dist_ch2mean-0.25*dist_ch2std, dist_ch2mean+0.25*dist_ch2std,color=ch2_color,alpha=0.2)
                plt.ylim(ylimits_plot)
                plt.title('aligned with ' +  str(alignment_protocol) + ', mean +/- 1/4 std')
                plt.xlabel('frames')
                plt.ylabel('distance moved')
                plt.ylim(ylimits_plot)
                plt.show()
        else:
            
            raise Exception('options for "display_channels": "both" (default), "ch1","ch2"')
            
