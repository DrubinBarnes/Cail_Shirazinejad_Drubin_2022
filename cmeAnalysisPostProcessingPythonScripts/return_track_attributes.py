# Cyna Shirazinejad, Drubin/Barnes Lab, modified 6/12/20

from generate_index_dictionary import return_index_dictionary
import numpy as np
index_dictionary = return_index_dictionary()


def return_puncta_x_position(tracks,
                             track_number,
                             channel,
                             frame):

    return tracks[track_number][index_dictionary['index_x_pos']][channel][frame]


def return_puncta_y_position(tracks,
                             track_number,
                             channel,
                             frame):

    return tracks[track_number][index_dictionary['index_y_pos']][channel][frame]

def return_puncta_x_position_whole_track(tracks,
                             track_number,
                             channel):

    
    start_buffer_test = tracks[track_number][index_dictionary['index_startBuffer']]
    positions = tracks[track_number][index_dictionary['index_x_pos']][channel]

    if len(start_buffer_test) == 0:

        return tracks[track_number][index_dictionary['index_x_pos']][channel]

    else:
        start_buffer = tracks[track_number][index_dictionary['index_startBuffer']]['x'][0][0][channel]
        end_buffer = tracks[track_number][index_dictionary['index_endBuffer']]['x'][0][0][channel]
        return np.concatenate((start_buffer, positions, end_buffer))

#     return tracks[track_number][index_dictionary['index_x_pos']][channel]


def return_puncta_y_position_whole_track(tracks,
                             track_number,
                             channel):

    
    start_buffer_test = tracks[track_number][index_dictionary['index_startBuffer']]
    positions = tracks[track_number][index_dictionary['index_y_pos']][channel]

    if len(start_buffer_test) == 0:

        return tracks[track_number][index_dictionary['index_y_pos']][channel]

    else:
        start_buffer = tracks[track_number][index_dictionary['index_startBuffer']]['y'][0][0][channel]
        end_buffer = tracks[track_number][index_dictionary['index_endBuffer']]['y'][0][0][channel]
        return np.concatenate((start_buffer, positions, end_buffer))

def return_track_category(tracks,
                          track_number):

    return tracks[track_number][index_dictionary['index_catIdx']][0][0]


def return_track_lifetime(tracks,
                          track_number):

    return tracks[track_number][index_dictionary['index_lifetime_s']][0][0]


def return_track_amplitude(tracks,
                           track_number):

    return tracks[track_number][index_dictionary['index_amplitude']]


def return_track_x_position(tracks,
                            track_number):

    return tracks[track_number][index_dictionary['index_x_pos']]

def return_track_y_position(tracks,
                            track_number):

    return tracks[track_number][index_dictionary['index_y_pos']]

def return_is_CCP(tracks,
                  track_number):

    if tracks[track_number][index_dictionary['index_isCCP']][0][0] == 1:
        return True
    else:
        return False

def return_frames_in_track(tracks,
                           track_number):
    
#     return tracks[track_number][index_dictionary['index_frames']][0]

    start_buffer_test = tracks[track_number][index_dictionary['index_startBuffer']]
    frames = tracks[track_number][index_dictionary['index_frames']][0]

    if len(start_buffer_test) == 0:

        return frames

    else:
        start_buffer = tracks[track_number][index_dictionary['index_startBuffer']]['f'][0][0][0]
        end_buffer = tracks[track_number][index_dictionary['index_endBuffer']]['f'][0][0][0]
#         print(start_buffer)
#         print(end_buffer)
#         print(frames)
        return np.concatenate((start_buffer, frames, end_buffer))
    
    
    
def return_track_amplitude_no_buffer_channel(tracks,
                                             track_number,
                                             channel):

    return tracks[track_number][index_dictionary['index_amplitude']][channel]


def return_track_amplitude_start_buffer_chanel(tracks,
                                               track_number,
                                               channel):

    return tracks[track_number][index_dictionary['index_startBuffer']]['A'][0][0][channel]


def return_track_amplitude_end_buffer_channel(tracks,
                                              track_number,
                                              channel):

    return tracks[track_number][index_dictionary['index_endBuffer']]['A'][0][0][channel]


def return_track_amplitude_one_channel(tracks,
                                       track_number,
                                       channel):
#     print(tracks[track_number][index_dictionary['index_startBuffer']])
#     print(tracks[track_number])
    start_buffer_test = tracks[track_number][index_dictionary['index_startBuffer']]
    amplitudes = return_track_amplitude_no_buffer_channel(tracks, track_number, channel)

    if len(start_buffer_test) == 0:

        return amplitudes

    else:
#         print('test')
        start_buffer = return_track_amplitude_start_buffer_chanel(tracks, track_number, channel)
        end_buffer = return_track_amplitude_end_buffer_channel(tracks, track_number, channel)
        return np.concatenate((start_buffer, amplitudes, end_buffer))


def return_designated_channels_amplitudes(tracks,
                                         channels,
                                         scale_to_one):
    
    return_amplitudes = []
    
    for i in range(len(tracks)):
        
        track_amplitudes = []
        
        for j in range(len(channels)):
            
            
            channel_amplitudes = return_track_amplitude_one_channel(tracks, i, j)
            
            if scale_to_one:
                
                channel_amplitudes = scale_amplitudes(channel_amplitudes)
                
            track_amplitudes.append(channel_amplitudes)
        
        return_amplitudes.append(track_amplitudes)
        
    return return_amplitudes
        
        
def scale_amplitudes(amplitudes):

    max_amplitude = max(amplitudes)

    return (1 / max_amplitude) * amplitudes

def return_distance_traveled_from_origin(tracks,
                                         track_number,
                                         channel):
    
    x_positions = return_puncta_x_position_whole_track(tracks,
                                                       track_number,
                                                       channel)
    
    y_positions = return_puncta_y_position_whole_track(tracks,
                                                       track_number,
                                                       channel)
    
    distance_traveled = []
    
    for i in range(len(x_positions)):
        
        distance_traveled.append(np.sqrt( (x_positions[i] - x_positions[0])**2 + (y_positions[i] - y_positions[0])**2) )
    
    return distance_traveled



def return_distance_between_two_channel(tracks,
                                        track_number,
                                        first_channel,
                                        second_channel):
    
    first_channel_x_position = return_puncta_x_position_whole_track(tracks,
                                                                    track_number,
                                                                    first_channel)    
    
    second_channel_x_position = return_puncta_x_position_whole_track(tracks,
                                                                    track_number,
                                                                    second_channel)
    
    first_channel_y_position = return_puncta_y_position_whole_track(tracks,
                                                                    track_number,
                                                                    first_channel)    
    
    second_channel_y_position = return_puncta_y_position_whole_track(tracks,
                                                                    track_number,
                                                                    second_channel)
    
    distance_separation = []
    
    for i in range(len(first_channel_x_position)):
        
        distance_separation.append(np.sqrt( (first_channel_x_position[i] - second_channel_x_position[i])**2 + (first_channel_y_position[i] - second_channel_y_position[i])**2 ))
        
    return distance_separation


def return_pvals_detection_no_buffer(tracks,
                                     track_number,
                                     channel):
    """Extract the the p-values of detections in a track (not including possible start/end buffers)."""
    
    
    return tracks[track_number][index_dictionary['index_pval_Ar']][channel]


def return_puncta_x_position_no_buffer_one_channel(tracks,
                             track_number,
                             channel):
    """Extract the fitted x-axis coordinats of one channel of a track (not including possible start/end buffers)."""
    return tracks[track_number][index_dictionary['index_x_pos']][channel]
def return_puncta_y_position_no_buffer_one_channel(tracks,
                             track_number,
                             channel):
    """Extract the fitted y-axis coordinats of one channel of a track (not including possible start/end buffers)."""
    return tracks[track_number][index_dictionary['index_y_pos']][channel]