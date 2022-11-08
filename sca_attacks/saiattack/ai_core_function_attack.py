from .attack_functions import full_ranks_dict
from .attack_functions import full_ranks_dict_cubic
from .attack_functions import load_sca_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sutils import load_ascad_attack_groups
from sutils import check_file_exists
from sutils import JSONManager

import os

from tqdm import trange


def supervised_attack(train_dir, ascad_database, bindex=0, num_traces=2000, apply_scaler=False):
    jsonManager = JSONManager(os.path.join(traindirectory, 'training_info.json'))
    #jsonManager.get_value('training_model_name')
    attack_list = jsonManager.get_value('attack_list') if jsonManager.has_key('attack_list') else []
    model_file = os.path.join(train_dir, jsonManager.get_value('training_model_name'))

    check_file_exists(model_file)
    check_file_exists(ascad_database)
    # Load profiling and attack data and metadata from the ASCAD database
    (X_attack, Y_attack, Metadata_attack) = load_ascad_attack_groups(ascad_database, trace_type=np.float32, load_metadata=True)

    print ('[INFO]: Training model name:', jsonManager.get_value('training_model_name'))
    print ('[INFO]: Attack traces shape:', X_attack.shape)
    print ('[INFO]: Attack label shape:', Y_attack.nrows)
    print ('[INFO]: Few samples:', X_attack[0][0:10])
    print ('[INFO]: Few labels:', Y_attack[0:10])
    print ('[INFO]: Few metadata:', Metadata_attack[0])

    # Prepare scaler list
    scaler_list = []
    if apply_scaler and not applynewscaler:
        import pickle
        scalers_list_path = jsonManager.get_value('applied_scalers')
        for scaler in scalers_list_path:
            with open(scaler, 'rb') as f:
                scaler_list.append(pickle.load(f))
    else:
        print ('[INFO]: Apply scaler purely based on attack traces')
        ss_new = StandardScaler()
        mm_new = MinMaxScaler()
        # Training new scalers
        for i in trange(X_attack.shape[0], desc='[INFO]: Apply new standard scalers'):
            ss_new.partial_fit([X_attack[i]])
        for i in trange(X_attack.shape[0], desc='[INFO]: Apply new mimmax scalers'):
            mm_new.partial_fit(ss_new.transform([X_attack[i]]))
        scaler_list.append(ss_new)
        scaler_list.append(mm_new)
        
    # Computing indexes array
    all_indexes = []
    print ('[INFO]: Computing indexes')
    for j in trange(batteryattack):
        if not randomindex:
            sub_X_attack = X_attack[bindex:]
            sub_Metadata_attack = Metadata_attack[bindex:]
        else:
            indexes = np.random.choice(np.array(range(0, X_attack.shape[0]), dtype=np.uint32), num_traces, replace=False)
            all_indexes.append(indexes)
    
    X_attack = X_attack[:]
    # Load model
    model = load_sca_model(model_file)
    x = None
    cumulative = None
    print ('[INFO]: Performing battery of attacks')
    for j in trange(batteryattack):
        sub_X_attack = None
        sub_X2_attack = None
        sub_Metadata_attack = None
        
        indexes = all_indexes[j]
        sub_X_attack = X_attack[indexes]
        sub_Metadata_attack = Metadata_attack.read_coordinates(indexes)
                    
        if samples > 0:
            sub_X_attack = sub_X_attack[:, :samples]
        
        # Apply scalers
        for storaged_scaler in scaler_list:
            sub_X_attack = storaged_scaler.transform(sub_X_attack)
            
        train_identifier = train_dir.split('/')[-1]       
        input_data = sub_X_attack[:num_traces, :]
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    
        predict_dict = {'input_1': input_data}
        ranks = full_ranks_dict(model, predict_dict, sub_Metadata_attack, 0, num_traces, 10, attack_byte)
        if x is None:
            #x = np.empty(shape=ranks.shape[0], dtype=np.uint16)
            x = [ranks[i][0] for i in range(0, ranks.shape[0])]#ranks[:, 0]
            cumulative = np.zeros(shape=(len(x)), dtype=np.uint32)
        
        y = [ranks[i][1] for i in range(0, ranks.shape[0])]
        cumulative = np.add(cumulative, ranks[:, 1])
        bindex = bindex + offset


    print ('[INFO]: Cumulative', cumulative)
    
    matplotlib.use('pdf')
    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=False)
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    plt.rc('axes', labelsize='medium')
    result = np.true_divide(cumulative, batteryattack, dtype=np.float32)
    file_name = os.path.join(train_dir, 'average_ge_{}_{}_bt{}_mode-{}_i{}{}_{}{}'.format('b'+str(attack_byte), 'ans' if applynewscaler else 'as', batteryattack, 'ri' if randomindex else 'sq', '_bi' + str(bindex) if not randomindex else '', len(attack_list), '_ptb' if portable_exp else '', train_identifier))
    np.save(file_name, np.array([x, result]))
    plt.xlabel('Number of Traces')
    plt.ylabel('Guessing Entropy')
    plt.grid(True)
    plt.plot(x, result)
    plt.savefig(file_name)
    plt.close()
    attackJSON = JSONManager()
    attackJSON.add_string('npy_file', file_name + '.npy')
    attackJSON.add_string('pdf_file', file_name + '.pdf')
    attack_list.append(attackJSON.JSONObject)
    jsonManager.add_array('attack_list', attack_list)
    jsonManager.save()

train_dir = traindirectory
num_traces = numtraces
bindex = bindex
ascad_database=traces_file=dataset
print ('[INFO]: Using', num_traces, 'to attack')
if not randomindex:
    print ('[INFO]: Initial index:', bindex, 'final index:', bindex+num_traces)
else: 
    print ('[INFO]: Set for random index')
print ('[INFO]: Apply scaler:', applyscaler, 'but scaler based on traces' if applynewscaler else '')
print ('[INFO]: Train dir:', train_dir)
print ('[INFO]: Setting for baterry of attacks')
print ('[INFO]: Traces dffset:', offset) 
print ('[INFO]: Attack byte:', attack_byte)
print ('[INFO]: Portable experiment:', portable_exp)