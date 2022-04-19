def display_intensity_maxima(tracks,tracks_to_keep):
    
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

    number_of_tracks = len(tracks['tracks'][0])

    index_dictionary = return_index_dictionary()
    number_of_tracks = len(tracks['tracks'][0])
    
    index_dictionary = return_index_dictionary()

    tracks = sort_tracks_descending_lifetimes(tracks)

    # unzip the tracks from the indices
    (tracks, indices) = zip(*tracks)


    ch1_maxima = []
    ch2_maxima = []
    
    for i in range(0,number_of_tracks):
        
        
        if i in tracks_to_keep:
            
            
            ch1_all_amplitudes, ch2_all_amplitudes = return_ch1_ch2_amplitudes(tracks,i)
        
            # ch1_amplitudes_kept.append(ch1_all_amplitudes)
            # ch2_amplitudes_kept.append(ch2_all_amplitudes)
                
            ch1_maxima.append(np.max(ch1_all_amplitudes))
            ch2_maxima.append(np.max(ch2_all_amplitudes))
    
    figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')        
    plt.title('histogram of ch1 maximum track intensities')
    plt.hist(ch1_maxima, bins = len(ch1_maxima))
    plt.xlabel('au fluorescence intensity')
    plt.ylabel('counts')
    plt.show()
    
    figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')        
    plt.title('histogram of ch2 maximum track intensities')
    plt.hist(ch2_maxima, bins = len(ch2_maxima))
    plt.xlabel('au fluorescence intensity')
    plt.ylabel('counts')
    plt.show()

    print('The ch1 maxima are: ' + str(ch1_maxima))

    print()

    print('The ch2 maxima are: ' + str(ch2_maxima))
