
def get_dataset_info(dataset, window_len):
    
    if dataset == 'BASELINE':
        if window_len == 30:
            mwl_levels = {'eyesclosed':1706, 'eyesopen':1648}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'eyesopen_v_eyesclosed'
            print('labels:', label_dict)
        elif window_len == 10:
            mwl_levels = {'eyesclosed': 7840, 'eyesopen': 7578}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'eyesopen_v_eyesclosed'
            print('labels:', label_dict)
        elif window_len == 5:
            mwl_levels = {'eyesclosed': 15832, 'eyesopen': 15321}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'eyesopen_v_eyesclosed'
            print('labels:', label_dict)
        elif window_len == 2:
            mwl_levels = {'eyesclosed': 39784, 'eyesopen': 38526}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'eyesopen_v_eyesclosed'
            print('labels:', label_dict)
        else:
            raise ValueError(f'Window length of size {window_len} was not yet created for this experiment')

    
    elif dataset == 'EYESOPEN_V_NBACK':
        if window_len == 30:
            mwl_levels = {'eyesopen':1653, 'nback':3076}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'eyesopen_v_nback'
            print('labels:', label_dict)
        elif window_len == 30:
            mwl_levels = {'eyesopen':1653, 'nback':3076}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'eyesopen_v_nback'
            print('labels:', label_dict)
        elif window_len == 30:
            mwl_levels = {'eyesopen':1653, 'nback':3076}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'eyesopen_v_nback'
            print('labels:', label_dict)
        elif window_len == 30:
            mwl_levels = {'eyesopen':1653, 'nback':3076}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'eyesopen_v_nback'
            print('labels:', label_dict)
        else:
            raise ValueError(f'Window length of size {window_len} was not yet created for this experiment')

        
    elif dataset == 'EASY_V_HARD':
        if window_len == 30:
            mwl_levels = {'easy':1049, 'hard':999}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'nback_easy_v_hard'
            print('labels:', label_dict)
        else:
            raise ValueError(f'Window length of size {window_len} was not yet created for this experiment')
        
    elif dataset == 'EASY_V_MEDIUM':
        if window_len == 30:
            mwl_levels = {'easy':1049, 'medium':1048}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'nback_easy_v_medium'
            print('labels:', label_dict)
        else:
            raise ValueError(f'Window length of size {window_len} was not yet created for this experiment')
        
    elif dataset == 'MEDIUM_V_HARD':
        if window_len == 30:
            mwl_levels = {'medium':1048, 'hard':999}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'nback_medium_v_hard'
            print('labels:', label_dict)
        else:
            raise ValueError(f'Window length of size {window_len} was not yet created for this experiment')
        
    elif dataset == 'EASY_V_MEDIUM_V_HARD':
        if window_len == 30:
            mwl_levels = {'easy':1049, 'medium':1048, 'hard':999}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'nback_easy_v_med_v_hard'
            print('labels:', label_dict)
        elif window_len == 10:
            mwl_levels = {'easy': 5671, 'medium': 5682, 'hard': 5267}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'nback_easy_v_med_v_hard'
            print('labels:', label_dict)
        elif window_len == 5:
            mwl_levels = {'easy': 11650, 'medium': 11590, 'hard': 10767}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'nback_easy_v_med_v_hard'
            print('labels:', label_dict)
        elif window_len == 2:
            mwl_levels = {'easy': 29543, 'medium': 29332, 'hard': 27326}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'nback_easy_v_med_v_hard'
            print('labels:', label_dict)
        else:
            raise ValueError(f'Window length of size {window_len} was not yet created for this experiment')
        
    elif dataset == '0_v_2':
        if window_len == 30:
            mwl_levels = {'0':612, '2':621}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'nback_0_v_2'
            print('labels:', label_dict)
        else:
            raise ValueError(f'Window length of size {window_len} was not yet created for this experiment')
        
    elif dataset == '0_v_4':
        if window_len == 30:
            mwl_levels = {'0':612, '4':581}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'nback_0_v_4'
            print('labels:', label_dict)
        else:
            raise ValueError(f'Window length of size {window_len} was not yet created for this experiment')
        
    elif dataset == '0_v_2_v_4':
        if window_len == 30:
            mwl_levels = {'0':612, '2': 621, '4':581}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'nback_0_v_2_v_4'
            print('labels:', label_dict)
        else:
            raise ValueError(f'Window length of size {window_len} was not yet created for this experiment')
        
    elif dataset == '1_v_3_v_5':
        if window_len == 30:
            mwl_levels = {'1':612, '3': 621, '5':581}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'nback_1_v_3_v_5'
            print('labels:', label_dict)
        else:
            raise ValueError(f'Window length of size {window_len} was not yet created for this experiment')
        
    elif dataset == 'STEW':
        if window_len == 5:
            mwl_levels = {'low': 4170, 'high': 3925}
            label_dict = {}
            number_label_dict = {}
            for i, k in enumerate(mwl_levels.keys()):
                label_dict[k] = i
                number_label_dict[i] = k
            feature_name = 'stew'
            print('labels:', label_dict)
        else:
            raise ValueError(f'Window length of size {window_len} was not yet created for this experiment')
    
    return mwl_levels, label_dict, number_label_dict, feature_name
        