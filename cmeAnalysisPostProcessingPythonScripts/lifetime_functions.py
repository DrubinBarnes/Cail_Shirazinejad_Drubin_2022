def display_tracks_with_lifetimes(tracks,ch1_color,ch2_color,tracks_to_keep,first_selected_channel = 'ch1', second_selected_channel = 'ch1', first_channel_location = 'prior', second_channel_location = 'following', first_percentage_of_max = 10, second_percentage_of_max = 10,print_new_lifetimes = False, first_moving_average = False, first_moving_average_window = 3, second_moving_average=False, second_moving_average_window = 3):

    if first_selected_channel == 'ch1' or first_selected_channel == 'ch2':
        pass
    else:
        raise Exception('first_selected_channel must be "ch1" or "ch2"')
        
    if second_selected_channel == 'ch1' or second_selected_channel == 'ch2':
        pass
    else:
        raise Exception('second_selected_channel must be "ch1" or "ch2"')

    if first_channel_location == 'prior' or first_channel_location == 'following':
        pass
    else:
        raise Exception('first_channel_location must be "prior" or "following"')
        
    if second_channel_location == 'prior' or second_channel_location == 'following':
        pass
    else:
        raise Exception('second_channel_location must be "prior" or "following"')
        
    if first_percentage_of_max >= 0 or first_percentage_of_max <= 100:
        pass
    else:
        raise Exception('first_percentage_of_max must be >=0 and <=100')
        
    if second_percentage_of_max >= 0 or second_percentage_of_max <= 100:
        pass
    else:
        raise Exception('second_percentage_of_max must be >=0 and <=100')
                
        
        
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

    number_of_tracks = len(tracks['tracks'][0])

    index_dictionary = return_index_dictionary()

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

    number_of_tracks = len(tracks['tracks'][0])

    index_dictionary = return_index_dictionary()
    time_intervals = tracks['tracks'][0][0][index_dictionary['index_time_frames']][0][1]-tracks['tracks'][0][0][index_dictionary['index_time_frames']][0][0]

    lifetimes_new = []
    number_of_tracks = len(tracks['tracks'][0])
    
    tracks = sort_tracks_descending_lifetimes(tracks)

    # unzip the tracks from the indices
    (tracks, indices) = zip(*tracks)

    # imaging frame intervals
    
    for i in range(0,number_of_tracks):
        
        index_temp = i

        if index_temp in tracks_to_keep:

            start_buffer_ch1_amplitudes = 0
            end_buffer_ch1_amplitudes = 0

            start_buffer_ch2_amplitudes = 0
            end_buffer_ch2_amplitudes = 0

            # collect ch1/ch2 amplitudes

            ch1_amplitudes = tracks[i][index_dictionary['index_amplitude']][0]
            ch2_amplitudes = tracks[i][index_dictionary['index_amplitude']][1]

            # check if there is a start and end buffer amplitudes 
            start_buffer = tracks[i][index_dictionary['index_startBuffer']]
            start_buffer_amplitudes = len(start_buffer)

            end_buffer = tracks[i][index_dictionary['index_endBuffer']]
            end_buffer_amplitudes = len(end_buffer)

            # if there is start / end buffer, collect them and overwrite empty lists
            if start_buffer_amplitudes != 0:

                start_buffer_ch1_amplitudes = tracks[i][index_dictionary['index_startBuffer']]['A'][0][0][0]
                start_buffer_ch2_amplitudes = tracks[i][index_dictionary['index_startBuffer']]['A'][0][0][1]

                end_buffer_ch1_amplitudes = tracks[i][index_dictionary['index_endBuffer']]['A'][0][0][0]
                end_buffer_ch2_amplitudes = tracks[i][index_dictionary['index_endBuffer']]['A'][0][0][1]

                ch1_all_amplitudes = np.concatenate((start_buffer_ch1_amplitudes,ch1_amplitudes,end_buffer_ch1_amplitudes), axis = None)
                ch2_all_amplitudes = np.concatenate((start_buffer_ch2_amplitudes,ch2_amplitudes,end_buffer_ch2_amplitudes), axis = None)

            else:

                ch1_all_amplitudes = ch1_amplitudes
                ch2_all_amplitudes = ch2_amplitudes
            
            
            
            figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

            print('The TrackID of the following track is: ' + str(i))
            print('The old lifetime of the following track is: ' + str(tracks[i][index_dictionary['index_lifetime_s']][0][0]) + 's')
            
            if first_selected_channel == 'ch1':
                
                array_current_first_channel = np.asarray(ch1_all_amplitudes)
                
            elif first_selected_channel == 'ch2':
                
                array_current_first_channel = np.asarray(ch2_all_amplitudes)
                
            if second_selected_channel == 'ch1':
                
                array_current_second_channel = np.asarray(ch1_all_amplitudes)
            
            elif second_selected_channel == 'ch2':
                
                array_current_second_channel = np.asarray(ch2_all_amplitudes)
                
           
            if first_moving_average:
                
                array_current_first_channel = moving_average(array_current_first_channel,first_moving_average_window)
            
            if second_moving_average:
                
                array_current_second_channel = moving_average(array_current_second_channel,second_moving_average_window)

                    
                
            
            index_of_maximum_first_channel = np.argmax(array_current_first_channel)
            index_of_maximum_second_channel = np.argmax(array_current_second_channel)
            
            value_fraction_first_channel = first_percentage_of_max/100*array_current_first_channel[index_of_maximum_first_channel]
            value_fraction_second_channel = second_percentage_of_max/100*array_current_second_channel[index_of_maximum_second_channel]
            
            
       
            if first_channel_location== 'prior':
                
                idx_first_channel = (np.abs(array_current_first_channel[0:index_of_maximum_first_channel] - value_fraction_first_channel)).argmin()

            elif first_channel_location == 'following':
                    
                idx_first_channel = index_of_maximum_first_channel + (np.abs(array_current_first_channel[index_of_maximum_first_channel:] - value_fraction_first_channel)).argmin()
                
            if second_channel_location == 'prior':
                
                idx_second_channel = (np.abs(array_current_second_channel[0:index_of_maximum_second_channel] - value_fraction_second_channel)).argmin()

            elif second_channel_location == 'following':
                    
                idx_second_channel = index_of_maximum_second_channel + (np.abs(array_current_second_channel[index_of_maximum_second_channel:] - value_fraction_second_channel)).argmin()
                
            
            time_lifetime = np.abs(time_intervals*(idx_first_channel-idx_second_channel))
            
            lifetimes_new.append(time_lifetime)
            
            print('The new lifetime of the following track is: ' + str(time_lifetime) + 's')

            
            
            
            plt.plot(ch1_all_amplitudes,ch1_color,label='ch1')
            plt.plot(ch2_all_amplitudes,ch2_color,label='ch2')
        
            plt.axhline(y=value_fraction_first_channel, color='k', linestyle=':',label='first threshold')
            plt.axhline(y=value_fraction_second_channel, color='tab:brown', linestyle='-.',label='second threshold')
            
            
            plt.axvline(x=idx_first_channel,color='k', linestyle=':',label='first selection')
            plt.axvline(x=idx_second_channel,color='tab:brown', linestyle='-.',label='second selection')
            
            if first_moving_average:
                
                plt.plot(array_current_first_channel,linestyle=(0, (3, 1, 1, 1, 1, 1)),label='smoothed first selected channel')
                
            if second_moving_average:
                
                plt.plot(array_current_second_channel,linestyle=(0, (3, 1, 1, 1, 1, 1)),label='smoothed second selected channel')
        
            plt.xlabel('frames')
            plt.ylabel('au fluorescence intensity')
            plt.legend()
            plt.show()
            
    if print_new_lifetimes:
        print('The new lifetimes are: ' + str(lifetimes_new))


        

def view_lifetimes_before_modifications(tracks,tracks_to_keep):
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

    number_of_tracks = len(tracks['tracks'][0])

    index_dictionary = return_index_dictionary()

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

    number_of_tracks = len(tracks['tracks'][0])

    index_dictionary = return_index_dictionary()
    time_intervals = tracks['tracks'][0][0][index_dictionary['index_time_frames']][0][1]-tracks['tracks'][0][0][index_dictionary['index_time_frames']][0][0]

    lifetimes_new = []
    number_of_tracks = len(tracks['tracks'][0])
    
    tracks = sort_tracks_descending_lifetimes(tracks)

    # unzip the tracks from the indices
    (tracks, indices) = zip(*tracks)

    lifetime_list = []

    for i in range(0,number_of_tracks):
        
        index_temp = i

        if index_temp in tracks_to_keep:

            lifetime_list.append(tracks[i][index_dictionary['index_lifetime_s']][0][0])


    figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')        
    plt.title('histogram of lifetimes (s)')
    plt.hist(lifetime_list, bins = len(lifetime_list))
    plt.xlabel('lifetime (s)')
    plt.ylabel('counts')
    plt.show()

def moving_average(x, w):
    import numpy as np
    import scipy.io as sio
    import os
    import sys
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    return np.convolve(x, np.ones(w), 'valid') / w
