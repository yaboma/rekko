import pandas as pd 
import numpy as np 
import scipy.sparse as sp 
import pickle
from scipy.sparse import coo_matrix,vstack,hstack
from sklearn.feature_extraction.text import CountVectorizer
import tqdm
import average_precision

from sklearn.base import TransformerMixin

DATA_PATH = '../okko/orig_data'
PREPARED_PATH = './prepared_data/'

# Итак, нам надо вопроизвести максимально похоже условия использования системы. т.е. на момент времени t_train_end
# мы имеем только фильмы из трейна. и атрибуты от фильмов из трейна.
# теперь в момент t_test_end  мы будем иметь N  новых фильмов и M  новых пользователей - это задачи холодного старта.
# Разобьем нашу задачу на 4 и правильно сформируем тест.
# 1- старые пользователи - старые фильмы
# 2 - новые пользователи - старые фильмы
# 3 - старые пользователи - новые фильмы
# 4 - новые пользователи - новые фильмы

def get_target(actions):
    '''
    Функция, которая вернет число просмотреннх серий каждым пользователем каждого сериала, потом вернет то,что недопотребил
    А потом то, что точно потребил согласно правилам соревнования - например, так можно вычислить примерную длительность сериала 
    и его же рекомендовать в потребленные после.
    '''
    watch_actions = actions[actions.action == 'watch']
    # Блок нахождения всяких статистик по сериалам
    serials = watch_actions[watch_actions['type'] != 1]
    # Заменим длиетльность на 0, там где длительности нет.. или это очень короткие, надо подумоть.
    serials['num_of_series'] = (serials['watched_time']/serials['duration']).fillna(0).replace(np.inf,0).astype(int)
    serials['time_being'] = serials.index.get_level_values(2)
    dur_being = serials.groupby(level = 1).agg({'time_being':[min,len],'num_of_series':[lambda x:x.mode()[0],max]})
    dur_being.columns = ['time_being','count_of_watch','num_of_series_mode','num_of_series_max']
    # Модифицируем длитеьность сериала - как произвелдение числа серий на продолжиттельность одной
    dur = watch_actions.join(dur_being['num_of_series_max'])['num_of_series_max']*watch_actions['duration']
    watch_actions.loc[~dur.isnull(),'duration'] = dur[~dur.isnull()]
    
    
    # Блок нахождения статистик по фильмам для пользователя
    films = watch_actions[watch_actions['type'] == 1]
    # Здесь важно видимо, как долго смотрел
    films['time_being'] = films.index.get_level_values(2)
    dur_films = films.groupby(level = 1).agg({'time_being':[min,len]})
    dur_films.columns = ['time_being','count_of_watch']
    
    # Блок нахождения статистик по фильмам и пользователям
    watch_actions['rel_dur'] = (watch_actions['watched_time']/watch_actions['duration'])
    target = 1*(watch_actions['rel_dur'] >= 1/3) | watch_actions['consumption_mode'].isin(['R','P']) 
    target = target.groupby(level = [0,1]).mean()
    watch_actions = watch_actions.groupby(level = [0,1]).mean()
    watch_actions['rel_dur'] = watch_actions['rel_dur'].replace(np.inf,1)# Заглушка для фильмов с 0 длительностью
    
    
    
    return dur_being,dur_films,watch_actions,target


def get_train_test(actions,mode = 'by_time',perc = (0.6,0.2,0.2)):
    '''
    здесь не очень аккуратно обращаемся с временем просмотра, потому что фильмы на границе должны быть 
    с обрезанной длительностью - но насрать
    '''
    X = actions.copy()
    X['ones'] = 1
    X['increment'] = np.arange(len(X))
    if mode == 'by_time':
        
        by_time = X.groupby(level = 2)['ones'].sum()
        by_time.sort_index(inplace = True)
        #проверили, что вроде как все ок и равномерно во времени
        cur = 0
        idx = []
        for i in range(len(perc)):
#             print(np.round((cur)*len(by_time)),np.round((cur+perc[i])*len(by_time)))
            by_time_temp = by_time.iloc[int(np.round((cur)*len(by_time))):int(np.round((cur+perc[i])*len(by_time)))].index.values
            print(len(by_time_temp))
            mn = by_time_temp.min()
            mx = by_time_temp.max()
            cur+=perc[i]
            idx.append(X.loc[(slice(None),slice(None),slice(mn,mx)),'increment'].values)

    elif mode == 'by_time_wm':
        idx = []
        cur = 0
        by_time = np.sort(X['first_ts'])
        for i in range(len(perc)):
#             print(np.round((cur)*len(by_time)),np.round((cur+perc[i])*len(by_time)))
            by_time_temp = by_time[int(np.round((cur)*len(by_time))):int(np.round((cur+perc[i])*len(by_time)))]
            mn = np.min(by_time_temp)
            mx = np.max(by_time_temp)
            cur+=perc[i]
            idx.append(X.loc[(X['first_ts']<mx) & (X['first_ts']>=mn),'increment'].values)
   
    return idx


def get_users_features(actions,bag_of_attr):
    '''
    Получаем трейн
    bag_of_attr - словарь, где просто каждому id  фильма сопоставлена строка атрибутов через запятую.
    строго  говоря в просмотренных фильмах атрибутов может оказаться меньше, чем во всем пуле фильмов, но я 
    пока не знаю проблема ли это ToDo
    Если history_movie определен из теста, например, то мы должны убирать новинки из формирования матрицы для простого обучения.
    Без холодного старта.
    '''
    # Приделаем каждому чуваку атрибуты просмотренных фильмов. ну или вообще по всем действиям - они все позитивные
    ind_user = []
    buf = []
    for i in tqdm.tqdm(np.unique(actions.index.get_level_values(0))):
        
        temp = np.unique(actions.loc[i].index.get_level_values(0))
        ind_user.append(i)

        s = ''
        for ii in temp:
#             if (history_movie is None) or (ii in list(history_movie.keys())):
                s+=bag_of_attr[ii]

                s+=','
        #assert X.shape[1] == len(a)
        buf.append(s)

    cv1 = CountVectorizer(token_pattern='\d+',)
    X_user = cv1.fit_transform(buf)
    
    match_user_row = {i:ii for ii,i in enumerate(ind_user)}
    match_row_user = {ii:i for ii,i in enumerate(ind_user)}
    match_feature_columns = {i:ii for ii,i in enumerate(list(cv1.get_feature_names()))}
    match_columns_feature = {ii:i for ii,i in enumerate(list(cv1.get_feature_names()))}
    print(X_user.shape,len(match_user_row),len(match_feature_columns))
    return match_user_row,match_row_user,match_feature_columns,match_columns_feature,X_user
def shape_corrector(X,num_col,num_row):
    if X.shape[0]<num_row:
        X = vstack((X,coo_matrix((int(num_row - X.shape[0]),X.shape[1]))))
    if X.shape[1]<num_col:
        
        X = hstack((X,coo_matrix((X.shape[0],int(num_col - X.shape[1])))))
    return X
def get_cold_start_matrix(actions,match_user_row,match_feature_columns,match_movie_columns,bag_of_attr):
    '''
    Нужно переписать через coo_matrix, чтоб все атрибуты совпадали
    '''
    # Наполнение по тесту для старых пользователей и старых фильмов
    row_ = []
    col_ = []
    ones = []
    # Наполнение матрицы по старым атрибутам для новых пользователей
    row_user = []
    col_user = []
    ones_user = []
    # Здесь id  фильмов, которые не смотрели в трейне
    new_movie_buf = []
    
    buf = []
    
    ind_user = []
    for i in tqdm.tqdm(np.unique(actions.index.get_level_values(0))):
        if i in match_user_row:
            temp = np.unique(actions.loc[i].index.get_level_values(0))


            s = ''
            for ii in temp:
                for k in bag_of_attr[ii].split(','):
                    if k in match_feature_columns:
                        row_.append(match_user_row[i])
                        col_.append(match_feature_columns[k])
                        ones.append(1)
                if ii not in match_movie_columns:
                    # Фильма нет в трейне
                    # Значит нужно просто сохранить его id и забрать из большой таблицы с фичами и атрибутами
                    new_movie_buf.append(ii)
                    
        else:
            # Пользователя не было в трейне
            # По сути надо создать еще несколько массивов и мапов
            temp = np.unique(actions.loc[i].index.get_level_values(0))
            ind_user.append(i)

            for ii in temp:
                for k in bag_of_attr[ii].split(','):
                    if k in match_feature_columns:
                        row_user.append(len(ind_user)-1)
                        col_user.append(match_feature_columns[k])
                        ones_user.append(1)
                if ii not in match_movie_columns:
                    # Фильма нет в трейне и еще нет пользователя
                    pass
                    # ХЗ че с этим делать
            

    # По построению test matrix должна иметь те же размеры, что и трейн матрикс, но тут надо быть аккуратнее
    # Вроде как если не попадется максимальный номер строки или столбца, то он его не нарастит - надо проверку бы
    test_matrix = coo_matrix((ones,(row_,col_)))# Старые юзеры, старые фильмы, но новое распределение атрибутов
    test_matrix = shape_corrector(test_matrix,max(match_feature_columns.values())+1,max(match_user_row.values())+1)
    # ToDo - мб нужно будет как-то сложить матрицу атрибутов, но вроде не надо
    
    if len(ones_user) != 0:
        new_user_matrix = coo_matrix((ones_user,(row_user,col_user)))
        new_match_user_row = {i:ii for ii,i in enumerate(ind_user)}
        new_match_row_user = {ii:i for ii,i in enumerate(ind_user)}
        new_user_matrix = shape_corrector(new_user_matrix,max(match_feature_columns.values())+1,max(new_match_user_row.values())+1)
    else:
        new_user_matrix = coo_matrix((len(row_),len(col_)))
        new_match_user_row = {}
        new_match_row_user = {}
        new_user_matrix = {}
    if len(new_movie_buf)!=0:
        new_match_row_movie = {ii:i for ii,i in enumerate(new_movie_buf)}
        new_match_movie_row = {i:ii for ii,i in enumerate(new_movie_buf)}   
    else:
        new_match_row_movie = {}
        new_match_movie_row = {}
    return test_matrix,new_match_user_row,new_match_row_user,new_user_matrix,new_match_row_movie,new_match_movie_row

def df_to_matrix(X,match_user_row,match_element_row, is_censor = True, delimiter = 4):
        '''
        На вход подается датафрейм с мультииндексом <user_id, element_id> и некоторой оценкой пары, затем он переупорядочивается и дополняется 
        по шаблонам из строк всяких спарс матричек для фильмов и юзеров
        match_user_row - отображение из айди в номер строки в матрице, match_element_row - аналогично
        '''
        Y = X.copy()
        if is_censor:
            Y[(Y<delimiter)] = -1
            Y[(Y>=delimiter)] = 1
        else:
            pass#Y = Y+1 #  на всякий случай, а не то поделим что-нибудь на 0
        Y['users'] = Y.index.get_level_values(0).map(match_user_row)
        Y['items'] = Y.index.get_level_values(1).map(match_element_row)
        Y.dropna(subset = ['users','items'],inplace = True)
        Y['users'] = Y['users'].astype(int)
        Y['items'] = Y['items'].astype(int)
        Z = coo_matrix((Y[X.columns].values.squeeze(),(Y['users'].values,Y['items'].values)))
        print(max(match_element_row.values())+1,max(match_user_row.values())+1)
        Z = shape_corrector(Z,max(match_element_row.values())+1,max(match_user_row.values()) +1)
        print(X.shape,Y.shape)
        print(Z.shape)
        return Z

class FeatureExtractor(TransformerMixin):
    def __init__(self,all_about_movie,bag_of_attr,is_censor = True,delimiter = 4, mode = 'rating',target_col_name = 'rating',is_filtered = True,is_filtered_action = True):
        self.all_about_movie = all_about_movie
        self.movie_attr_matrix = all_about_movie['movie_attr_matrix']
        self.movie_match_columns_attr = all_about_movie['movie_columns_match']
        self.movie_match_attr_columns = all_about_movie['movie_match_columns']
        self.movie_match_row_movie = all_about_movie['movie_match_row_movie']
        self.movie_match_movie_row = all_about_movie['movie_match_movie_row']

        self.is_censor = is_censor
        self.delimiter = delimiter
        self.mode = mode
        self.target_col_name = target_col_name

        self.is_filtered = is_filtered
        self.is_filtered_action = is_filtered_action
        self.bag_of_attr = bag_of_attr
    def fit(self,X):
        if self.is_filtered_action:
            watch_actions_train = X[X['action'] == 'watch']
        else:
            watch_actions_train = X.copy()
        # Сначала нам нужна матрица из всех просмотренных фильмов в трейне и меппинги оттуда 
        # - это исчерпывающая информация известная на конец трейна
        res = get_users_features(watch_actions_train,self.bag_of_attr)
        
        self.match_user_row = res[0] 
        self.match_row_user = res[1]
        self.match_feature_columns = res[2]
        self.match_columns_feature = res[3]
        self.train_user = res[4]
        # Вообще фильмов здесь намного больше дб, наверное стоит как-то смеппить признак 1
        # но для обычных рекомендаций это не так уж и важно.
        self.train_movie_match_movie_row = {i:ii for ii,i in enumerate(np.unique(watch_actions_train.index.get_level_values(1)))}
        self.train_movie_match_row_movie = {ii:i for ii,i in enumerate(np.unique(watch_actions_train.index.get_level_values(1)))}
        
        
        return self
    def transform(self,X,y = None,):
        if self.is_filtered:
            if self.mode == 'rating':
                part_of_train = X.loc[X.action =='rate',self.target_col_name].groupby(level = [0,1]).mean().to_frame() 
            elif self.mode == 'duration':
                part_of_train = X.loc[X.action =='watch',self.target_col_name].groupby(level = [0,1]).mean().to_frame()
            elif self.mode == 'not_null':
                part_of_train = X.loc[~X[self.target_col_name].isnull(),self.target_col_name].groupby(level = [0,1]).mean().to_frame()
        else: 
            part_of_train = X[self.target_col_name].to_frame()
        res = df_to_matrix(part_of_train,self.match_user_row,self.train_movie_match_movie_row,self.is_censor,self.delimiter)
        return res
class ColdFeatureExtractor(TransformerMixin):
    def __init__(self,fitted_FE):
        self.fitted_FE = fitted_FE
        self.im_columns = ['is_purchase',
             'is_rent',
             'is_subscription',
             'duration',
             'feature_1',
             'feature_2',
             'feature_3',
             'feature_4',
             'feature_5',
             'type_movie',
             'type_serial',]
        
    def fit(self,X):
        # Задача вычленить фильмы из трейна из большой матрицы фильмов и перенумеровать id
        # Здесь же когда-нибудь появтся новинки
        if self.fitted_FE.is_filtered_action:
            self.movie_train = np.unique(X[X['action'] == 'watch'].index.get_level_values(1))
        else:
            self.movie_train = np.unique(X.index.get_level_values(1))
        
        # Теперь нужна матрица атрибутов фильмов для юзера
        self.attr_train_map = list(self.fitted_FE.match_feature_columns.keys())
        self.attr_train_map = [*self.im_columns,*self.attr_train_map]
        
        self.train_movie_rows = [self.fitted_FE.movie_match_movie_row[i] for i in self.movie_train]
        self.train_movie_cols = [self.fitted_FE.movie_match_attr_columns[i] for i in  self.attr_train_map]
        
        
        
        return self
    def transform(self,X,y = None,):
        # Сначала надо получить список не новинок, доступных на конец трейна
        #test_user_matrix,test_match_user_row,new_match_row_user,new_user_matrix,new_match_row_movie,new_match_movie_row=
        if self.fitted_FE.is_filtered: 
            if self.fitted_FE.mode == 'rating':
                part_of_train = X.loc[X.action =='rate',self.fitted_FE.target_col_name].groupby(level = [0,1]).mean().to_frame() 
            elif self.fitted_FE.mode == 'duration':
                part_of_train = X.loc[X.action =='watch',self.fitted_FE.target_col_name].groupby(level = [0,1]).mean().to_frame()
            elif self.fitted_FE.mode == 'not_null':
                part_of_train = X.loc[~X[self.fitted_FE.target_col_name].isnull(),self.fitted_FE.target_col_name].groupby(level = [0,1]).mean().to_frame()
        else:
            part_of_train = X[self.fitted_FE.target_col_name].to_frame()
        
        res = get_cold_start_matrix(part_of_train,self.fitted_FE.match_user_row,
                             self.fitted_FE.match_feature_columns,
                              self.movie_train,self.fitted_FE.bag_of_attr)
        
        train_movie = self.fitted_FE.movie_attr_matrix.tocsc()[:,self.train_movie_cols]
        train_movie = train_movie.tocsr()[self.train_movie_rows,:]
        test_rows = [self.fitted_FE.movie_match_movie_row[i] for i in res[5]]
        test_movie = self.fitted_FE.movie_attr_matrix.tocsc()[:,self.train_movie_cols]
        test_movie = test_movie.tocsr()[test_rows,:]
        # Мапы для фильмов
        map_movie_row = {i:ii for ii,i in enumerate(self.train_movie_rows)}
        map_row_movie = {ii:i for ii,i in enumerate(self.train_movie_rows)}
        
        map_movie_row_new = {i:ii for ii,i in enumerate(res[5])}
        map_row_movie_new = {ii:i for ii,i in enumerate(res[5])}
        
        map_feature_column = {i:ii for ii,i in enumerate(self.train_movie_cols)}
        map_column_feature = {ii:i for ii,i in enumerate(self.train_movie_cols)}
        
        res_dict = {
            # Сначала матрицы, потом меппинги
            'train_user':self.fitted_FE.train_user,
            'test_user':res[0],
            'new_test_user':res[3],
            # Все меппинги для столбцов из трейна, для трейна и теста по строкам одинаковые для старых пользователей
            'map_user_row':self.fitted_FE.match_user_row,
            'map_row_user':self.fitted_FE.match_row_user,
            'map_user_row_new':res[1],
            'map_row_user_new':res[2],
            'map_attr_column':self.fitted_FE.match_feature_columns,
            'map_column_attr':self.fitted_FE.match_columns_feature,
            
            # Для фильмов не так - уних могут появиться толок новые фильмы, а атрибуты не могут измениться 
            # (разве что новинковость)
            # 
            'train_movie':train_movie,
            'test_movie':test_movie,
            'map_movie_row':map_movie_row,
            'map_row_movie':map_row_movie,
            'map_movie_row_new':map_movie_row_new,
            'map_row_movie_new':map_row_movie_new,
            'map_feature_column':map_feature_column,
            'map_column_feature':map_column_feature,
        }
        
        # Теперь набор атрибутов и 
        return res_dict

def metric(true_data, predicted_data, k=20):
    true_data_set = {k: set(v) for k, v in true_data.items()}

    return average_precision.average_precision(true_data_set, predicted_data, k=k)

if __name__ == '__main__':
    # Получили фичи для фильмов
    actions = pd.read_pickle(PREPARED_PATH+'actions_one_table.pkl')

    actions.sort_index(inplace = True)

    idx = get_train_test(actions)

    train,test,valid = actions.iloc[idx[0]],actions.iloc[idx[1]],actions.iloc[idx[2]]

    with open(PREPARED_PATH+'bag_of_attr_movie.pkl','rb') as f:
        bag_of_attr = pickle.load(f)

    with open(PREPARED_PATH+'catalogue_features.pkl','rb') as f:
        match_element_row,match_row_element,match_columns,element_matrix = pickle.load(f)
    movie_match_columns = {i:ii for ii,i in enumerate(match_columns)}
    movie_columns_match = {ii:i for ii,i in enumerate(match_columns)}

    fe = FeatureExtractor({'movie_attr_matrix':element_matrix,'movie_match_columns':movie_match_columns,
                       'movie_columns_match':movie_columns_match,'movie_match_row_movie':match_row_element,
                      'movie_match_movie_row':match_element_row,},bag_of_attr,)
    fe.fit(train)
    train_ = fe.transform(train)
    test_ = fe.transform(test)

    train_.shape,test_.shape 

    cfe = ColdFeatureExtractor(fe)

    cfe.fit(train)

    print(len(cfe.train_movie_rows),len(cfe.train_movie_cols))

    train_res = cfe.transform(test)

    train_res.keys()

    print(train_res['train_user'].shape,train_res['test_user'].shape,train_res['new_test_user'].shape)

    print(train_res['train_movie'].shape,train_res['test_movie'].shape)