# Cyna Shirazinejad 6/18/20
# functions to view and manipulate track lifetimes
from generate_index_dictionary import return_index_dictionary
from return_track_attributes import (return_track_x_position, 
                                     return_designated_channels_amplitudes, 
                                     return_track_lifetime,
                                     return_track_amplitude_one_channel)
from alignment import (find_alignment_frame_protocol_max, 
                       find_alignment_frame_protocol_before_max, 
                       find_alignment_frame_protocol_after_max)
import matplotlib.pyplot as plt
import numpy as np

index_dictionary = return_index_dictionary()

def view_raw_lifetimes(tracks,
                       lifetime_limits = [0,np.Inf],
                       bins = 'default',
                       dpi=300,
                       fig_size = (6,4)):
                       

    lifetimes = []
    
    for i in range(len(tracks)):
        
        if return_track_lifetime(tracks,i) > lifetime_limits[0] and return_track_lifetime(tracks,i) < lifetime_limits[1]:
            
            lifetimes.append(return_track_lifetime(tracks,i))
    
    
    
    
    print(f'The total number of events in the tracks object are: {len(tracks)}')
    print(f'The number of events plotted within defined lifetime bounds are: {len(lifetimes)}')       
    plt.figure(num=None, figsize=fig_size, dpi=dpi, facecolor='w', edgecolor='k')        
    plt.title('histogram of raw lifetimes (s)')
    
    if bins == 'default':
        plt.hist(lifetimes)
    else:
        plt.hist(lifetimes, bins = bins)
    
    plt.hist(lifetimes)
    plt.xlabel('lifetime (s)')
    plt.ylabel('counts')
    plt.show()

def modify_lifetimes(tracks,
                     channel_colors=['g'],
                     display_channels=[0],
                     first_channel=0,
                     first_channel_location = -1,
                     first_channel_percentage_of_max=10,
                     first_moving_average=True,
                     first_average_display=True,
                     first_moving_average_window=3,
                     second_channel=1,
                     second_channel_location=0,
                     second_channel_percentage_of_max=10,
                     second_moving_average=True,
                     second_average_display=True,
                     second_moving_average_window=3,
                     print_old_lifetimes=True,
                     print_new_lifetimes=True,
                     display_final_histogram = True,
                     display_individual_modifications = True,
                     histogram_bins = 'default',
                     histogram_fig_size=(6,4),
                     histogram_dpi=300,
                     individual_fig_size=(6,4),
                     individual_dpi=100):
     
    
    number_of_channels = len(return_track_x_position(tracks,0))
    
    all_channel_amplitudes = return_designated_channels_amplitudes(tracks, [*range(number_of_channels)], False)                     
    
    first_channel_amplitudes = return_designated_channels_amplitudes(tracks, [first_channel], False)   
    
    second_channel_amplitudes = return_designated_channels_amplitudes(tracks, [second_channel], False)
    
    first_channel_pertinent_frames, first_channel_thresholds = retrieve_frames_and_thresholds_from_conditions(all_channel_amplitudes,
                                                                                                              first_channel_percentage_of_max,
                                                                                                              first_channel,
                                                                                                              first_channel_location,
                                                                                                              first_channel_percentage_of_max,
                                                                                                              first_moving_average,
                                                                                                              first_moving_average_window)

    second_channel_pertinent_frames, second_channel_thresholds = retrieve_frames_and_thresholds_from_conditions(all_channel_amplitudes,
                                                                                                               second_channel_percentage_of_max,
                                                                                                               second_channel,
                                                                                                               second_channel_location,
                                                                                                               second_channel_percentage_of_max, 
                                                                                                               second_moving_average,
                                                                                                               second_moving_average_window)
#     print(first_channel_pertinent_frames)
#     print(second_channel_pertinent_frames)
    
    interval = tracks[0][index_dictionary['index_time_frames']][0][1] - tracks[0][index_dictionary['index_time_frames']][0][0]                                               
    
    new_lifetimes = calculate_new_lifetimes(first_channel_pertinent_frames,
                           second_channel_pertinent_frames,
                           interval)
    
    if display_final_histogram:
        
        display_histogram(histogram_fig_size, 
                          histogram_dpi,
                          new_lifetimes,
                          histogram_bins)
    
    if display_individual_modifications:
        print()
        display_individual_lifetimes(tracks,
                                     new_lifetimes,
                                     channel_colors,
                                     display_channels,
                                     all_channel_amplitudes,
                                     first_channel,
                                     second_channel,
                                     first_channel_amplitudes,
                                     second_channel_amplitudes,
                                     first_channel_pertinent_frames,
                                     second_channel_pertinent_frames,
                                     first_channel_thresholds,
                                     second_channel_thresholds,
                                     first_moving_average,
                                     first_average_display,
                                     first_moving_average_window,
                                     second_moving_average,
                                     second_average_display,
                                     second_moving_average_window,
                                     print_old_lifetimes,
                                     print_new_lifetimes,
                                     individual_fig_size,
                                     individual_dpi)
            

            
    return new_lifetimes

def display_individual_lifetimes(tracks,
                                 new_lifetimes,
                                 channel_colors,
                                 display_channels,
                                 all_channel_amplitudes,
                                 first_channel,
                                 second_channel,
                                 first_channel_amplitudes,
                                 second_channel_amplitudes,
                                 first_channel_pertinent_frames,
                                 second_channel_pertinent_frames,
                                 first_channel_thresholds,
                                 second_channel_thresholds,
                                 first_moving_average,
                                 first_moving_average_display,
                                 first_moving_average_window,
                                 second_moving_average,
                                 second_moving_average_display,
                                 second_moving_average_window,
                                 print_old_lifetimes,
                                 print_new_lifetimes,
                                 individual_fig_size,
                                 individual_dpi):
    number_of_channels = len(return_track_x_position(tracks,0))
    for i in range(len(tracks)):
        
        plt.figure(num=None, figsize=individual_fig_size, dpi=individual_dpi, facecolor='w', edgecolor='k')
        print()
        print(f'The trackID of the following track is {i}')
        legend_label = []
        
#         track_lifetime = return_track_lifetime(tracks,i)
        
        if print_old_lifetimes:
            
            print(f'The old lifetime of the track is: {len(return_track_amplitude_one_channel(tracks,i,0))} s')
        
        if print_new_lifetimes:
        
            print(f'The new lifetime of the track is: {new_lifetimes[i]} s')
            
        for j in range(number_of_channels):
            
            if j in display_channels:
#                 print(len(all_channel_amplitudes[i][j]))
                plt.plot(all_channel_amplitudes[i][j], color=channel_colors[j])
                
                legend_label.append('ch' + str(j))
        
        if first_moving_average_display:
#             print(first_channel)
            plt.plot(moving_average(all_channel_amplitudes[i][first_channel],first_moving_average_window),linestyle=(0, (3, 1, 1, 1, 1, 1)))
            legend_label.append('first selected channel smoothed')

        if second_moving_average_display:
#             print('test')
            plt.plot(moving_average(all_channel_amplitudes[i][second_channel],second_moving_average_window),linestyle=(0, (3, 1, 1, 1, 1, 1)))
            legend_label.append('second selected channel smoothed')

        plt.axhline(y=first_channel_thresholds[i], color='k', linestyle=':')
        legend_label.append('first threshold')
        
        plt.axhline(y=second_channel_thresholds[i], color='tab:brown', linestyle='-.')
        legend_label.append('second_threshold')    
            
        plt.axvline(x=first_channel_pertinent_frames[i],color='k', linestyle=':')
        legend_label.append('first selection')
        plt.axvline(x=second_channel_pertinent_frames[i],color='tab:brown', linestyle='-.')
        legend_label.append('second selection')
        
        plt.plot(first_channel_pertinent_frames[i],first_channel_thresholds[i],marker='*',markerSize=20,color='r')
        plt.legend(legend_label)
        
        plt.show()
    
def display_histogram(histogram_fig_size, 
                      histogram_dpi,
                      new_lifetimes,
                      histogram_bins):
    
    
        
    plt.figure(num=None, figsize=histogram_fig_size, dpi=histogram_dpi, facecolor='w', edgecolor='k')
    if histogram_bins=='default':
        plt.hist(new_lifetimes)
    else:
        plt.hist(new_lifetimes,bins=histogram_bins)
    plt.xlabel('lifetime (s)')
    plt.ylabel('frequency')
    plt.title('modified lifetime histogram')
    plt.show()
        
def calculate_new_lifetimes(first_frame_locations,
                           second_frame_locations,
                           interval):
#     print(interval * np.abs((np.asarray(second_frame_locations) - np.asarray(first_frame_locations))))
    return interval * np.abs((np.asarray(second_frame_locations) - np.asarray(first_frame_locations)))

    
            

        
        
        
def retrieve_frames_and_thresholds_from_conditions(amplitudes,
                                                   alignment_percentage,
                                                   channel,
                                                   channel_location,
                                                   channel_percentage_of_max,
                                                   moving_average_convolution,
                                                   window_size):
                                    
    
    
    frames_return = []
    channel_thresholds = []
    
    for i in range(len(amplitudes)):
        
        current_amplitudes = amplitudes[i][channel]
#         print(len(current_amplitudes))  
            
        if channel_location==0:
            
            channel_thresholds.append(np.max(current_amplitudes))
#             print(np.max(current_amplitudes))
        elif channel_location==-1:
            
            channel_thresholds.append(np.max(current_amplitudes)*alignment_percentage/100)
            
        elif channel_location==1:
            
            channel_thresholds.append(np.max(current_amplitudes)*alignment_percentage/100)
#         print(current_amplitudes)   
            
        if moving_average_convolution:
#             print(current_amplitudes)
#             print(window_size)
#             print((np.ndarray.flatten(current_amplitudes)))
#             print('test')
            current_amplitudes = moving_average(np.ndarray.flatten(np.asarray(current_amplitudes)),window_size)

#         print(current_amplitudes)   
        if channel_location==0:
            
            frames_return.append(find_alignment_frame_protocol_max(current_amplitudes)[0])
        
        elif channel_location==-1:
            
            frames_return.append(find_alignment_frame_protocol_before_max(current_amplitudes,channel_percentage_of_max)[0])
            
        elif channel_location==1:
            
            frames_return.append(find_alignment_frame_protocol_after_max(current_amplitudes,channel_percentage_of_max)[0])
            
    return frames_return, channel_thresholds


def moving_average(x, w):
    import numpy as np
    return np.convolve(x, np.ones(w), 'valid') / w