def merge_experiments(merged_tracks_input_list, merged_selected_IDs_input_list):

    if len(merged_tracks_input_list) == len(merged_selected_IDs_input_list):
        pass
    else:
        raise Exception('the length of "merged_tracks_input_list" must be equal to the length of "merged_selected_IDs_input_list"')

    output_tuple_tracks = ()

    for i in range(len(merged_tracks_input_list)):

        tracks_kept_from_exp = tuple(merged_tracks_input_list[i][j] for j in range(len(merged_tracks_input_list[i])) 
                                     if j in merged_selected_IDs_input_list[i])

        output_tuple_tracks += tracks_kept_from_exp

    return output_tuple_tracks

