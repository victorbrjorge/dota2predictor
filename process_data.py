#!/usr/bin/python
#-*- coding: utf-8 -*-
# encoding: utf-8

import pandas as pd
import numpy as np


def join_tables():

    matches = pd.read_csv('data/match.csv')
    players = pd.read_csv('data/players.csv')
    hero_names = pd.read_csv('data/hero_names.csv')

    matches = matches.loc[matches['game_mode'] == 22]
    
    needed_columns = ['match_id', 'radiant_win']
    matches = matches[needed_columns]

    needed_columns = ['match_id', 'hero_id', 'player_slot']
    players = players[needed_columns]

    #hero_names = hero_names[needed_columns]

    hero_df = pd.merge(players, hero_names, on='hero_id', how='left')
    hero_df.set_index(['match_id','player_slot'], inplace=True)

    id_hero_df = hero_df['hero_id'].unstack()
    name_hero_df = hero_df['localized_name'].unstack()

    dataset = pd.merge(matches, id_hero_df, left_index=True, right_index=True)
    dataset = pd.merge(dataset,name_hero_df,left_index = True,right_index = True)

    cols = dataset.columns.tolist()
    cols = cols[:1] + cols[2:] + cols[1:2]
    dataset = dataset[cols]

    return dataset

def apply_one_hot_enc(dataset):
    
    dataset['radiant_id'] = [np.zeros(112) for _ in range(len(dataset))]
    dataset['dire_id'] = [np.zeros(112) for _ in range(len(dataset))]
    dataset['radiant_name'] = [[] for _ in range(len(dataset))]
    dataset['dire_name'] = [[] for _ in range(len(dataset))]


    for ind in dataset.index:
        for num in range(0,5):
            d = dataset.loc[ind,str(num)+'_x']
            name = dataset.loc[ind,str(num)+'_y']
            dataset.loc[ind,'radiant_id'][d-1] = 1
            dataset.loc[ind,'radiant_name'].append(name)
        for num in range(128,133):
            d = dataset.loc[ind,str(num)+'_x']
            name = dataset.loc[ind,str(num)+'_y']
            dataset.loc[ind,'dire_id'][d-1] = 1
            dataset.loc[ind,'dire_name'].append(name)

    df = dataset[['radiant_id','dire_id']]
    def f(x):
        return list(x.radiant_id)+list(x.dire_id)
    dataset['combine_id'] = df.apply(f, axis = 1)
    
    dataset.radiant_win = dataset['radiant_win'].astype(int)
    return dataset

if __name__ == '__main__':
    
    dataset = join_tables()
    dataset = apply_one_hot_enc(dataset)
    dataset.to_csv('dota_matches.csv', index=False)