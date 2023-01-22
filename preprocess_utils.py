import numpy as np

def get_quantile(df, colname, interval=0.1):
    thres = [0] + list(np.arange(interval, 1, interval)) + [1]
    return df[colname].quantile(thres)

def get_fifth_str(x):
    try :
        return x[5]
    except:
        return 'etc'

def get_count_percentages(df, col, n, perc):
    temp = df[col].value_counts()/n
    #above = (temp > perc)
    below = (temp <= perc)
    return temp[below].index

def clip_with_quantile(df, col, q_low, q_high):
    q_low = np.round(q_low, 2)
    q_high = np.round(q_high, 2)
    Q = get_quantile(df, col, 0.05)
    Q.index = np.round(Q.index, 2)
    v_low = Q[q_low]
    v_high = Q[q_high]
    return df[col].clip(v_low, v_high, axis=0)

def get_oper_part_word(oper_part_token, text_cnt_freq_text, top_K):
    text_cnt_freq_text_filtered = text_cnt_freq_text[:top_K]
    word_dict = dict(zip(text_cnt_freq_text_filtered,
                         [f'word{i}' for i in range(1, top_K + 1)]))
    oper_part_word = []
    for i in oper_part_token:
        temp = []
        for j in i:
            try :
                temp.append(word_dict[j])
            except:
                temp.append('word_etc')
        oper_part_word.append(temp)
    return oper_part_word
                
def get_dummy_df(oper_part_word, top_K):
    word_df_cols = [f'word{i}' for i in range(1, top_K+1)] + ['word_etc']
    TEMP = []
    for col in word_df_cols:
        temp = []
        for word in oper_part_word:
            for col in word:
                temp.append(1)
            else:
                temp.append(0)
        TEMP.append(temp)
    word_df = pd.DataFrame(np.array(TEMP).T)
    word_df.columns = word_df_cols
    return word_df
    
