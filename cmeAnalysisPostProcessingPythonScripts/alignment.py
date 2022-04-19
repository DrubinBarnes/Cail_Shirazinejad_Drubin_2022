# Cyna Shirazinejad 6/16/20
# modified alignment functions for tracking data of arbitrary number of channels
# 
from return_track_attributes import return_designated_channels_amplitudes
import numpy as np
import matplotlib.pyplot as plt
from return_track_attributes import return_track_x_position


def align_tracks(tracks, 
                 channel_colors, 
                 alignment_channel = 0,
                 alignment_protocol = 0, 
                 alignment_percentage = 100,
                 display_channels=[0],
                 display_movement_fluctuations = False,
                 scale_to_one_aligned=False, 
                 scale_to_one_all=False,
                 padding=False, 
                 padding_amount=0, 
                 pad_with_zero = False,
                 stds = 0.25,
                 ylimits=[-0.1, 1.1],
                 fig_size = (3,2),
                 dpi = 300):

    
        
    number_of_channels = len(return_track_x_position(tracks,0))
    
    channel_amplitudes = return_designated_channels_amplitudes(tracks, [*range(number_of_channels)], scale_to_one_aligned)
    
    frames_before_alignment, frames_after_alignment = find_frames_around_alignment_point(channel_amplitudes,
                                                                               alignment_channel, 
                                                                               alignment_protocol,
                                                                               alignment_percentage)

    
    max_frames_around_alignment = np.max([np.max(frames_before_alignment), np.max(frames_after_alignment)])
    
    if padding and (max_frames_around_alignment > padding_amount or max_frames_around_alignment > padding_amount):

        raise Exception('the number of padded frames to center your aligned feature is less than the amount of default padded frames. increase the padded value to be greater than or equal to ' + str(max_frames_around_alignment))
   
    if not padding:
                        
        padding_amount = max_frames_around_alignment    
    
    shifted_amplitudes = return_shifted_amplitudes(channel_amplitudes, 
                                                   frames_before_alignment, 
                                                   frames_after_alignment, 
                                                   padding_amount, 
                                                   pad_with_zero)


    shifted_amplitudes_averaged, shifted_amplitudes_std = return_averaged_amplitudes(shifted_amplitudes)

    if scale_to_one_all:
        
        shifted_amplitudes_averaged, shifted_amplitudes_std = scale_all_to_one(shifted_amplitudes_averaged, shifted_amplitudes_std)
    
    plot_aligned_features(shifted_amplitudes_averaged, 
                          shifted_amplitudes_std, 
                          padding_amount, 
                          channel_colors,
                          display_channels,
                          stds,
                          ylimits,
                          dpi,
                          fig_size)

def scale_all_to_one(mean_amplitudes, std_amplitudes):
    
    scaled_averaged_amplitudes = []
    scaled_std_amplitudes = []
    
    for i in range(len(mean_amplitudes)):
        
        max_signal = np.max(mean_amplitudes[i])
        
        scaled_averaged_amplitudes.append(1/max_signal * mean_amplitudes[i])
        scaled_std_amplitudes.append(1/max_signal * std_amplitudes[i])
        
    return scaled_averaged_amplitudes, scaled_std_amplitudes
        

    
def return_shifted_amplitudes(amplitudes, 
                              frames_before_alignment, 
                              frames_after_alignment, 
                              padding_amount,
                              pad_with_zero):
    
    shifted_amplitudes = []

    for i in range(len(amplitudes)):

        temp_track_amplitudes = []

        for j in range(len(amplitudes[i])):


            temp_track_amplitudes.append(pad_amplitudes(amplitudes[i][j],
                                                        frames_before_alignment[i],
                                                        frames_after_alignment[i],
                                                        padding_amount,pad_with_zero))
        
        shifted_amplitudes.append(np.asarray(temp_track_amplitudes))
        
    return shifted_amplitudes
    
                                                   
def return_averaged_amplitudes(amplitudes):
    
    amplitudes = np.asarray(amplitudes)
    
    averaged_amplitudes = []
    
    amplitudes_averaged = np.nan_to_num(np.nanmean(amplitudes,axis=0,dtype=np.float64))
    amplitudes_std = np.nan_to_num(np.nanstd(amplitudes,axis=0,dtype=np.float64))
    
    return amplitudes_averaged, amplitudes_std
    
    
def plot_aligned_features(shifted_amplitudes_averaged, 
                          shifted_amplitudes_std, 
                          padding_amount, 
                          channel_colors,
                          display_channels,
                          stds,
                          ylimits,
                          dpi,
                          fig_size):
    
    plt.figure(num=None, figsize=fig_size, dpi=dpi, facecolor='w', edgecolor='k')
    legend_label = []
    for i in range(len(shifted_amplitudes_averaged)):
        
        if i in display_channels:
            
            plt.plot(range(-padding_amount,padding_amount + 1),shifted_amplitudes_averaged[i],channel_colors[i])
            plt.fill_between(range(-padding_amount,padding_amount + 1),
                             shifted_amplitudes_averaged[i] - stds * shifted_amplitudes_std[i],
                             shifted_amplitudes_averaged[i] + stds * shifted_amplitudes_std[i],
                             color = channel_colors[i],
                             alpha = 0.2)
            legend_label.append('ch' + str(i))
    plt.legend(legend_label)        
    plt.ylim(ylimits) 
    plt.show()
    
def pad_amplitudes(amplitudes, 
                   frames_before, 
                   frames_after, 
                   padding_amount,
                   pad_with_zero):
    
    frames_before = padding_amount - frames_before
    frames_after = padding_amount - frames_after
    
    if pad_with_zero:
        pad_val = 0
    else:
        pad_val = np.nan
    vector_before = np.full(frames_before, pad_val)
    vector_after = np.full(frames_after, pad_val)
    
    return np.concatenate((vector_before, amplitudes, vector_after), axis = 0)

                
    
def find_frames_around_alignment_point(channel_amplitudes,
                                       alignment_channel, 
                                       alignment_protocol,
                                       alignment_percentage):
    
    frames_before_alignment = []
    frames_after_alignment = []
    
    for i in range(len(channel_amplitudes)):
        
        if alignment_protocol == 0:
            
            frames_temp = find_alignment_frame_protocol_max(channel_amplitudes[i][alignment_channel])

            frames_before_alignment.append(frames_temp[0])
            frames_after_alignment.append(frames_temp[1])
            
        elif alignment_protocol == -1:
            
            frames_temp = find_alignment_frame_protocol_before_max(channel_amplitudes[i][alignment_channel],
                                                                   alignment_percentage)
            
            frames_before_alignment.append(frames_temp[0])
            frames_after_alignment.append(frames_temp[1])

        elif alignment_protocol == 1:
            
            frames_temp = find_alignment_frame_protocol_after_max(channel_amplitudes[i][alignment_channel],
                                                                   alignment_percentage)
            
            frames_before_alignment.append(frames_temp[0])
            frames_after_alignment.append(frames_temp[1])
            
    return frames_before_alignment, frames_after_alignment


def find_alignment_frame_protocol_max(amplitudes):
    
    return (np.argmax(amplitudes), len(amplitudes) - np.argmax(amplitudes) - 1)


def find_alignment_frame_protocol_before_max(amplitudes, 
                                             alignment_percentage):
    
#     print(np.argmax(amplitudes))
#     print(len(amplitudes))
    value_fraction = amplitudes[np.argmax(amplitudes)]*alignment_percentage/100
    
    if np.argmax(amplitudes) == 0:
        
        return (0, len(amplitudes) - 1)
    
    else:
        
        idx = (np.abs(amplitudes[0:np.argmax(amplitudes)] - value_fraction)).argmin()
        return (idx,len(amplitudes) - idx - 1)

    
def find_alignment_frame_protocol_after_max(amplitudes, alignment_percentage):

    
    value_fraction = amplitudes[np.argmax(amplitudes)]*alignment_percentage/100
    
    if np.argmax(amplitudes) == (len(amplitudes) - 1):
        
        return (len(amplitudes) - 1, 0)
    
    else:
        
        idx = (np.abs(amplitudes[np.argmax(amplitudes):] - value_fraction)).argmin()
        return (np.argmax(amplitudes) + idx, len(amplitudes) - idx - np.argmax(amplitudes) - 1)