#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import datetime
now = datetime.datetime.now
import click, re, sys, os, time, subprocess
import glob
from joblib import dump, load
import xgboost
from icecube import NewNuFlux, AtmosphericSelfVeto, astro
import pandas as pd
import matplotlib.pyplot as plt
import histlite as hl
from csky.utils import ensure_dir
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, cross_validate
flush = sys.stdout.flush

@click.group (invoke_without_command=True, chain=True)
@click.option ('--save/--nosave', default=False)
@click.pass_context
def cli (ctx, save):
    pass


@cli.command()
@click.option( 'train','--train_size', default = 0.5)
@click.option('--seed', default = 3, type=int)
@click.option('--feat', default = None)
@click.option('--use_corsika', default=True)
def load_and_split(train, seed, feat,use_corsika):
    def load_and_weight(use_corsika):
        nu = pd.read_hdf('/data/user/ssclafani/data/cscd/IC86_2013/IC86_2013_egen.hdf5')
        #nu = pd.read_hdf('/data/user/ssclafani/data/cscd/IC86_2013/IC86_2013_mc_estest_nopull.hdf5')
        mu = pd.read_hdf('/data/user/ssclafani/data/cscd/IC86_2013/IC86_2013_muongun_testes_nopull.hdf5')
        if use_corsika:
            corsika = (pd.read_hdf('/data/user/ssclafani/data/cscd/IC86_2013/IC86_2013_corsika_testes_nopull.hdf5'))
            print('Corsika Raw Events: {}'.format(len(corsika)))
        print('Nugen Raw Events: {}'.format(len(nu)))
        print('MuonGun Raw Events: {}'.format(len(mu)))
        nu.cgen_azi = nu.cgen_azi%(2*np.pi)
        nu.ra= nu.cgen_azi
        nu.cgen_zen = np.arccos(np.cos(nu.cgen_zen))
        nu.dec = nu.cgen_zen - np.pi/2
        
        mu.cgen_azi = mu.cgen_azi%(2*np.pi)
        mu.ra= mu.cgen_azi
        mu.cgen_zen = np.arccos(np.cos(mu.cgen_zen))
        mu.dec = mu.cgen_zen - np.pi/2
        
        if use_corsika:
            corsika.cgen_azi = corsika.cgen_azi%(2*np.pi)
            corsika.ra= corsika.cgen_azi
            corsika.cgen_zen = np.arccos(np.cos(corsika.cgen_zen))
            corsika.dec = corsika.cgen_zen - np.pi/2
        if use_corsika:
            print('Weighted Corsika Events: {}'.format(sum(corsika.weight)*86400*364))
        print('Weighted MuonGun Events: {}'.format((sum(np.nan_to_num(mu.weight)*86400*364))))

        nu['true_ra'] = nu.trueRa
        nu['true_dec'] = nu.trueDec
        mu['true_ra'] = mu.trueRa
        mu['true_dec'] = mu.trueDec
        nu['true_energy'] = nu.trueE
        mu['true_energy'] = mu.trueE
        nu['oneweight'] = nu.ow
        mu['oneweight'] = mu.weight

        nu['dpsi'] = astro.angular_distance (nu.true_ra, nu.true_dec, nu.ra, nu.dec)
        nu['mese_dpsi'] = astro.angular_distance (nu.true_ra, nu.true_dec, nu.DNNMESE_azi, nu.DNNMESE_zen - np.pi/2)

        mu['corsika'] = np.zeros(len(mu))


        if use_corsika == True:
            corsika['true_ra'] = corsika.trueRa
            corsika['true_dec'] = corsika.trueDec
            corsika['true_energy'] = corsika.trueE
            corsika['oneweight'] = corsika.weight
            corsika['corsika'] = np.ones(len(corsika))
        for a in (nu, mu):
            a['sigma'] = np.where (np.isfinite (a.sigma), a.sigma, np.radians (180))
        if use_corsika:
                corsika['sigma'] = np.where (np.isfinite (corsika.sigma), corsika.sigma, np.radians (180))
        inds = mu.loc[np.isnan(mu['oneweight']) == True].index
        mu = mu.drop(index = inds)
        if use_corsika:
            inds = corsika.loc[np.isnan(corsika['oneweight']) == True].index
            corsika = corsika.drop(index = inds)
        honda = NewNuFlux.makeFlux ('honda2006')
        honda.knee_reweighting_model = 'gaisserH3a_elbert'
        honda.relative_kaon_contribution = .91
        af = AtmosphericSelfVeto.AnalyticPassingFraction
        honda_veto_mese = af('conventional', veto_threshold=1e2)
        veto_args = nu.trueptype, nu.true_energy, -np.sin (nu.true_dec), 1950. - nu.DNNlabels_true_VertexZ
        phi_atm = np.vectorize (honda.getFlux) (nu.trueptype, nu.true_energy, -np.sin (nu.true_dec))
        veto_atm = honda_veto_mese (*veto_args)

        nu['weight_atm'] = 2* nu.oneweight * phi_atm * veto_atm
        nu['weight_E250'] = 2.23e-18 * nu.oneweight * (nu.true_energy/1e5)**-2.5
        nu['weight_E200'] = 1e-18 * nu.oneweight * (nu.true_energy/1e5)**-2
        nu['weight_E225'] = 1.5e-18 * nu.oneweight * (nu.true_energy/1e5)**-2.25
        nu['weight_total'] = nu.weight_atm + nu.weight_E200
        return nu, mu, corsika
    def prep_data(nu, mu, corsika, seed, train):
        mask_CC_numu = (np.abs (nu.trueptype) == 14) & (nu.trueinttype == 1)
        tnu = nu[mask_CC_numu]
        cnu = nu[~mask_CC_numu]

        livetime = .95 * 365.25 * 86400
        for (a, label) in zip ((tnu, cnu), 'tracks cascades'.split()):
            print ('{:10s} {:4} atm + {:3} astro (per year)'.format (
                label+':', *(livetime * np.array ([a.weight_atm.sum(), a.weight_E250.sum()])).astype(int)))
        print ('muons: {:2} atm'.format ((livetime * mu.weight.sum())))
        if use_corsika:
            print ('Corsika muons: {:2} atm'.format ((livetime * corsika.weight.sum())))
        if feat == 'slim':
            features = [u'DNNMESE_E', u'DNNMESE_azi', u'DNNMESE_azi_unc', 
                        #u'DNNMESE_x',u'DNNMESE_y', u'DNNMESE_z', 
                        u'DNNMESE_zen', u'DNNMESE_zen_unc',
                        u'DNN_es_length_02', u'DNN_es_length_log_unc_02',
                        u'DNN_es_p_starting_300m_mp_big_02', u'DNN_es_p_track',
                        u'cgen_E', u'cgen_azi',
                        u'cgen_time', u'cgen_x', u'cgen_y', u'cgen_z', u'cgen_zen',
                        u'cscdSBU_DepthFirstHit', u'cscdSBU_I3XYScale',
                        u'cscdSBU_MaxQtotRatio_HLC', u'cscdSBU_Qtot_HLC',
                        u'cscdSBU_VetoEarliestLayer', u'cscdSBU_VetoMaxDomChargeString',
                        u'cscdSBU_distance_deepcore', u'cscdSBU_distance_icecube',
                        u'cscdl3_cont_tag',# u'dec', 
                        u'es_03_starting',
                        u'es_cscdl2_big_starting', u'es_cscdl3_02_starting'] #u'monopod_E',
                           #u'monopod_azi', #u'monopod_x', u'monopod_y', u'monopod_z',
                           #u'monopod_zen', #u'ow', 
                          # u'ra', u'sigma']
        elif feat == 'min':
            features = [#u'DNNMESE_E', u'DNNMESE_azi', u'DNNMESE_azi_unc', 
                        #u'DNNMESE_x',u'DNNMESE_y', u'DNNMESE_z', 
                        #u'DNNMESE_zen', u'DNNMESE_zen_unc',
                        u'DNN_es_length_02', #u'DNN_es_length_log_unc_02',
                        u'DNN_es_p_starting_300m_mp_big_02', u'DNN_es_p_track',
                        u'cgen_E', u'cgen_azi',
                        u'cgen_time', u'cgen_x', u'cgen_y', u'cgen_z', u'cgen_zen',
                        u'cscdSBU_DepthFirstHit', u'cscdSBU_I3XYScale',
                        u'cscdSBU_MaxQtotRatio_HLC', u'cscdSBU_Qtot_HLC',
                        u'cscdSBU_VetoEarliestLayer', u'cscdSBU_VetoMaxDomChargeString',
                        u'cscdSBU_distance_deepcore', u'cscdSBU_distance_icecube',
                        u'cscdl3_cont_tag']# u'dec', 
                        #u'es_03_starting',
                        #u'es_cscdl2_big_starting', u'es_cscdl3_02_starting'] #u'monopod_E',
                           #u'monopod_azi', #u'monopod_x', u'monopod_y', u'monopod_z',
                           #u'monopod_zen', #u'ow', 
                          # u'ra', u'sigma']
        else:
            print('using full feature set')
            features = [u'DNNMESE_E', u'DNNMESE_azi', u'DNNMESE_azi_unc', 
                        #u'DNNMESE_x',u'DNNMESE_y', u'DNNMESE_z', 
                        u'DNNMESE_zen', u'DNNMESE_zen_unc',
                        u'DNN_es_length_02', u'DNN_es_length_log_unc_02',
                        u'DNN_es_p_starting_300m_mp_big_02', u'DNN_es_p_track',
                        u'cgen_E', u'cgen_azi',
                        u'cgen_time', u'cgen_x', u'cgen_y', u'cgen_z', u'cgen_zen',
                        u'cscdSBU_DepthFirstHit', u'cscdSBU_I3XYScale',
                        u'cscdSBU_MaxQtotRatio_HLC', u'cscdSBU_Qtot_HLC',
                        u'cscdSBU_VetoEarliestLayer', u'cscdSBU_VetoMaxDomChargeString',
                        u'cscdSBU_distance_deepcore', u'cscdSBU_distance_icecube',
                        u'cscdl3_cont_tag', u'dec', u'es_03_starting',
                        u'es_cscdl2_big_starting', u'es_cscdl3_02_starting'] #u'monopod_E',
                           #u'monopod_azi', #u'monopod_x', u'monopod_y', u'monopod_z',
                           #u'monopod_zen', #u'ow', 
                           #u'ra', u'sigma']
        def split_test_train(a, weight_keys, percent_train=.5, random_state=seed):
            atrain, atest = train_test_split(a, train_size = percent_train, random_state = seed, shuffle=True)
            Ftrain = percent_train
            Ftest = 1 - percent_train
            atrain.loc[:, weight_keys] /=Ftrain
            atest.loc[:, weight_keys] /=Ftest
            return atrain, atest

        nu_weight_keys = ['oneweight', 'weight_E250', 'weight_atm', 'weight_total', 'weight_E200', 'weight_E225']
        mu_weight_keys =['oneweight', 'weight', 'weight_sibyll', 'weight_dpmjet']
        corsika_weight_keys =['oneweight', 'weight']
        print('splitting nu')
        nu_train, nu_test = split_test_train (nu, nu_weight_keys, train, seed)
        print('Nu Train: {} Nu Test: {}'.format(len(nu_train), len(nu_test)))
        print('splitting muons')
        mu_train,  mu_test  = split_test_train (mu, mu_weight_keys, train, seed)
        print('Mu Train: {} Mu Test: {}'.format(len(mu_train), len(mu_test)))
        if use_corsika:
            print('Splitting Corsika')
            corsika_train,  corsika_test  = split_test_train (corsika, corsika_weight_keys, train, seed)
            print('Corsika Train: {} Corsika Test: {}'.format(len(corsika_train), len(corsika_test)))
            comb_train = pd.concat([mu_train, corsika_train])
            comb_test = pd.concat([mu_test, corsika_test])
            weight_train = np.concatenate ((nu_train.weight_E250, comb_train.weight))
            data_train = pd.concat([nu_train[features] , comb_train[features]], sort=False)
            target_train = np.repeat ([0, 1], map (len, (nu_train, comb_train)))
            data_test = pd.concat([nu_test[features] , comb_test[features]], sort=False)
            mu_true = np.array(np.ones(len(comb_test)))
            nu_true = np.array(np.zeros(len(nu_test)))
            target_test = np.hstack((nu_true, mu_true))
        else:
            print('Not using Corsika')
            weight_train = np.concatenate ((nu_train.weight_E250, mu_train.weight))
            data_train = pd.concat([nu_train[features] , mu_train[features]], sort=False)
            target_train = np.repeat ([0, 1], map (len, (nu_train, mu_train)))
            data_test = pd.concat([nu_test[features] , mu_test[features]], sort=False)
            mu_true = np.array(np.ones(len(mu_test)))
            nu_true = np.array(np.zeros(len(nu_test)))
            target_test = np.hstack((nu_true, mu_true))
        return data_train, target_train, data_test, target_test, mu_true, nu_true
    def save(data_train, target_train, data_test, target_test, seed, use_corsika, train):
        #save spit, weighted, data
        print('Saving...')
        if feat == True:
            if use_corsika:
                ensure_dir('/data/user/ssclafani/data/cscd/{}_bdt/seed_{}/corsika/split_{}/'.format(feat, seed, train ))
                pd.to_pickle(data_test, '/data/user/ssclafani/data/cscd/{}_bdt/seed_{}/corsika/split_{}/data_test_split_{}.pickle'.format(feat, seed, train, train))
                pd.to_pickle(target_test, '/data/user/ssclafani/data/cscd/{}_bdt/seed_{}/corsika/split_{}/target_test_split_{}.pickle'.format(feat, seed, train, train))
                pd.to_pickle(data_train, '/data/user/ssclafani/data/cscd/{}_bdt/seed_{}/corsika/split_{}/data_train_split_{}.pickle'.format(feat, seed, train, train ))
                pd.to_pickle(target_train, '/data/user/ssclafani/data/cscd/{}_bdt/seed_{}/corsika/split_{}/target_train_split_{}.pickle'.format(feat, seed, train, train ))
            else:
                ensure_dir('/data/user/ssclafani/data/cscd/{}_bdt/seed_{}/no_corsika/split_{}/'.format(seed, train ))
                pd.to_pickle(data_test, '/data/user/ssclafani/data/cscd/{}_bdt/seed_{}/no_corsika/split_{}/data_test_split_{}.pickle'.format(feat, seed, train, train ))
                pd.to_pickle(target_test, '/data/user/ssclafani/data/cscd/{}_bdt/seed_{}/no_corsika/split_{}/target_test_split_{}.pickle'.format(feat, seed, train, train ))
                pd.to_pickle(data_train, '/data/user/ssclafani/data/cscd/{}_bdt/seed_{}/no_corsika/split_{}/data_train_split_{}.pickle'.format(feat, seed, train, train ))
                pd.to_pickle(target_train, '/data/user/ssclafani/data/cscd/{}_bdt/seed_{}/no_corsika/split_{}/target_train_split_{}.pickle'.format(feat, seed, train, train ))
        else:
            if use_corsika:
                print('saving')
                ensure_dir('/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/'.format(seed, train ))
                pd.to_pickle(data_test, '/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/data_test_split_{}_egen.pickle'.format(seed, train, train))
                pd.to_pickle(target_test, '/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/target_test_split_{}_egen.pickle'.format(seed, train, train))
                pd.to_pickle(data_train, '/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/data_train_split_{}_egen.pickle'.format(seed, train, train ))
                pd.to_pickle(target_train, '/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/target_train_split_{}_egen.pickle'.format(seed, train, train ))
                
                #pd.to_pickle(data_test, '/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/data_test_split_{}.pickle'.format(seed, train, train))
                #pd.to_pickle(target_test, '/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/target_test_split_{}.pickle'.format(seed, train, train))
                #pd.to_pickle(data_train, '/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/data_train_split_{}.pickle'.format(seed, train, train ))
                #pd.to_pickle(target_train, '/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/target_train_split_{}.pickle'.format(seed, train, train ))
            else:
                ensure_dir('/data/user/ssclafani/data/cscd/bdt/seed_{}/no_corsika/split_{}/'.format(seed, train ))
                pd.to_pickle(data_test, '/data/user/ssclafani/data/cscd/bdt/seed_{}/no_corsika/split_{}/data_test_split_{}.pickle'.format(seed, train, train ))
                pd.to_pickle(target_test, '/data/user/ssclafani/data/cscd/bdt/seed_{}/no_corsika/split_{}/target_test_split_{}.pickle'.format(seed, train, train ))
                pd.to_pickle(data_train, '/data/user/ssclafani/data/cscd/bdt/seed_{}/no_corsika/split_{}/data_train_split_{}.pickle'.format(seed, train, train ))
                pd.to_pickle(target_train, '/data/user/ssclafani/data/cscd/bdt/seed_{}/no_corsika/split_{}/target_train_split_{}.pickle'.format(seed, train, train ))
                
    nu, mu, corsika = load_and_weight(use_corsika)
    data_train, target_train, data_test, target_test, mu_true, nu_true = prep_data(nu, mu, corsika, seed, train)
    save(data_train, target_train, data_test, target_test, seed, use_corsika, train)

@cli.command()
@click.option('--n_cv', default = 0)
@click.option('--train_size', default = 0.5)
@click.option('--max_depth', default = 8)
@click.option('--lr', default = 0.02)
@click.option('--feat', default = None)
@click.option('--n_cpu', default = 12)
@click.option('--n_estimators', default = 2001)
@click.option('--seed', default = 3)
@click.option('--use_corsika', default=True)
@click.option('--early_stop', default=20)
def step01_train(n_cv, train_size, max_depth, lr, feat, n_estimators, seed , use_corsika, n_cpu, early_stop):
    #load the mc and train a BDT
    def load_train_test(train_size, feat, seed):
        if feat == 'slim':
            if use_corsika:
                print('loading /data/user/ssclafani/data/cscd/slim_bdt/seed_{}/corsika/split_{}/data_test_split_{}.pickle'.format(seed, train_size, train_size ))
                data_test = pd.read_pickle('/data/user/ssclafani/data/cscd/slim_bdt/seed_{}/corsika/split_{}/data_test_split_{}.pickle'.format(seed, train_size, train_size ))
                target_test = pd.read_pickle( '/data/user/ssclafani/data/cscd/slim_bdt/seed_{}/corsika/split_{}/target_test_split_{}.pickle'.format(seed, train_size, train_size ))
                data_train =  pd.read_pickle('/data/user/ssclafani/data/cscd/slim_bdt/seed_{}/corsika/split_{}/data_train_split_{}.pickle'.format(seed, train_size, train_size ))
                target_train = pd.read_pickle('/data/user/ssclafani/data/cscd/slim_bdt/seed_{}/corsika/split_{}/target_train_split_{}.pickle'.format(seed, train_size, train_size ))
            else:
                data_test = pd.read_pickle('/data/user/ssclafani/data/cscd/slim_bdt/seed_{}/no_corsika/split_{}/data_test_split_{}.pickle'.format(seed, train_sizei, train_size))
                target_test = pd.read_pickle( '/data/user/ssclafani/data/cscd/slim_bdt/seed_{}/no_corsika/split_{}/target_test_split_{}.pickle'.format(seed, train_size, train_size ))
                data_train = pd.read_pickle('/data/user/ssclafani/data/cscd/slim_bdt/seed_{}/no_corsika/split_{}/data_train_split_{}.pickle'.format(seed, train_size, train_size ))
                target_train  = pd.read_pickle('/data/user/ssclafani/data/cscd/slimbdt/seed_{}/no_corsika/split_{}/target_train_split_{}.pickle'.format(seed, train_size, train_size ))
        elif feat == 'min':
            if use_corsika:
                print('loading /data/user/ssclafani/data/cscd/min_bdt/seed_{}/corsika/split_{}/data_test_split_{}.pickle'.format(seed, train_size, train_size ))
                data_test = pd.read_pickle('/data/user/ssclafani/data/cscd/min_bdt/seed_{}/corsika/split_{}/data_test_split_{}.pickle'.format(seed, train_size, train_size ))
                target_test = pd.read_pickle( '/data/user/ssclafani/data/cscd/min_bdt/seed_{}/corsika/split_{}/target_test_split_{}.pickle'.format(seed, train_size, train_size ))
                data_train =  pd.read_pickle('/data/user/ssclafani/data/cscd/min_bdt/seed_{}/corsika/split_{}/data_train_split_{}.pickle'.format(seed, train_size, train_size ))
                target_train = pd.read_pickle('/data/user/ssclafani/data/cscd/min_bdt/seed_{}/corsika/split_{}/target_train_split_{}.pickle'.format(seed, train_size, train_size ))
            else:
                data_test = pd.read_pickle('/data/user/ssclafani/data/cscd/min_bdt/seed_{}/no_corsika/split_{}/data_test_split_{}.pickle'.format(seed, train_sizei, train_size))
                target_test = pd.read_pickle( '/data/user/ssclafani/data/cscd/min_bdt/seed_{}/no_corsika/split_{}/target_test_split_{}.pickle'.format(seed, train_size, train_size ))
                data_train = pd.read_pickle('/data/user/ssclafani/data/cscd/min_bdt/seed_{}/no_corsika/split_{}/data_train_split_{}.pickle'.format(seed, train_size, train_size ))
                target_train  = pd.read_pickle('/data/user/ssclafani/data/cscd/min_bdt/seed_{}/no_corsika/split_{}/target_train_split_{}.pickle'.format(seed, train_size, train_size ))
        else:
            if use_corsika:
                print('loading /data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/data_test_split_{}.pickle'.format(seed, train_size, train_size ))
                data_test = pd.read_pickle('/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/data_test_split_{}.pickle'.format(seed, train_size, train_size ))
                target_test = pd.read_pickle( '/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/target_test_split_{}.pickle'.format(seed, train_size, train_size ))
                data_train =  pd.read_pickle('/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/data_train_split_{}.pickle'.format(seed, train_size, train_size ))
                target_train = pd.read_pickle('/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/split_{}/target_train_split_{}.pickle'.format(seed, train_size, train_size ))
            else:
                data_test = pd.read_pickle('/data/user/ssclafani/data/cscd/bdt/seed_{}/no_corsika/split_{}/data_test_split_{}.pickle'.format(seed, train_sizei, train_size))
                target_test = pd.read_pickle( '/data/user/ssclafani/data/cscd/bdt/seed_{}/no_corsika/split_{}/target_test_split_{}.pickle'.format(seed, train_size, train_size ))
                data_train = pd.read_pickle('/data/user/ssclafani/data/cscd/bdt/seed_{}/no_corsika/split_{}/data_train_split_{}.pickle'.format(seed, train_size, train_size ))
                target_train  = pd.read_pickle('/data/user/ssclafani/data/cscd/bdt/seed_{}/no_corsika/split_{}/target_train_split_{}.pickle'.format(seed, train_size, train_size ))
        return data_test, target_test, data_train, target_train    
    def train(data_train, target_train, data_test, target_test, max_depth, lr, n_estimators, seed, n_cpu, early_stop):
        max_depth = max_depth 
        learning_rate = lr
        min_child_weight = 1
        gamma= 0.0
        n_estimators = n_estimators
        seed = seed
        subsample = 0.5
        rf = bdt = xgboost.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                learning_rate=learning_rate, nthread=n_cpu, min_child_weight=min_child_weight, 
                subsample=subsample, shuffle=True, gamma=gamma, seed=seed)
        if n_cv == 0:
            rf.get_params()
            if early_stop > 0 :
                bdt.fit(data_train, target_train, 
                    eval_set= [(data_train, target_train), (data_test, target_test)],
                    eval_metric = ['error', 'logloss', 'rmse', 'auc'],
                    early_stopping_rounds=early_stop, verbose=20)
            else:
                bdt.fit(data_train, target_train, 
                    eval_set= [(data_train, target_train), (data_test, target_test)],
                    eval_metric = ['error', 'logloss', 'rmse', 'auc'])
        return bdt 

    data_test, target_test, data_train, target_train = load_train_test(train_size, feat, seed)
    bdt = train(data_train, target_train, data_test, target_test, max_depth, lr, n_estimators, seed, n_cpu, early_stop)
    
        
    if feat == 'slim':
        ensure_dir('/data/user/ssclafani/data/cscd/slim_bdt/seed_{}/corsika/bdt/n_cv_{}/'.format(seed, n_cv))
        dump(bdt, '/data/user/ssclafani/data/cscd/slim_bdt/seed_{}/corsika/bdt/n_cv_{}/bdt_max_depth_{}_n_est_{}lr_{}_seed_{}_train_size_{}.joblib'.format(
                seed, n_cv, max_depth, n_estimators, lr, seed, int(train_size*100.)))
    elif feat == 'min':
        ensure_dir('/data/user/ssclafani/data/cscd/min_bdt/seed_{}/corsika/bdt/n_cv_{}/'.format(seed, n_cv))
        dump(bdt, '/data/user/ssclafani/data/cscd/min_bdt/seed_{}/corsika/bdt/n_cv_{}/bdt_max_depth_{}_n_est_{}lr_{}_seed_{}_train_size_{}.joblib'.format(
                seed, n_cv, max_depth, n_estimators, lr, seed, int(train_size*100.)))
    else:
        if use_corsika:
            ensure_dir('/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/bdt/n_cv_{}/'.format(seed, n_cv))
            dump(bdt, '/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/bdt/n_cv_{}/bdt_max_depth_{}_n_est_{}lr_{}_seed_{}_train_size_{}.joblib'.format(
                    seed, n_cv, max_depth, n_estimators, lr, seed, int(train_size*100.)))


@cli.command()
@click.option('--n_cv', default=0)
@click.option('--train_size', default = 0.5)
@click.option('--max_depth', default = 8)
@click.option('--lr', default = 0.02)
@click.option('--n_cpu', default = 12)
@click.option('--n_estimators', default = 2001)
@click.option('--seed', default = 3)
@click.option('--use_corsika', default=True)
@click.option('--hist', default=False)
@click.option('--systematics', default=None)
@click.option('--feat', default=None)
@click.option('--save', default=True)
@click.option('--year', default=2013)
@click.option('--save_to_ana', default=False)
@click.option('--mu_cut', default=.5e-4)
def process_through_bdt(n_cv, train_size, max_depth, lr, n_estimators, seed , use_corsika, n_cpu, hist, systematics, feat, save, year, save_to_ana, mu_cut):
    #load data, mc proces through BDT
    #years = [2014]
    add_cut = False 
    livetimes = {2012: 317.00, 2013: 365} #357.61, 2014: 362.47}
    if feat == 'slim':
        features_step01 = [u'DNNMESE_E', u'DNNMESE_azi', u'DNNMESE_azi_unc', 
                        #u'DNNMESE_x',u'DNNMESE_y', u'DNNMESE_z', 
                        u'DNNMESE_zen', u'DNNMESE_zen_unc',
                        u'DNN_es_length_02', u'DNN_es_length_log_unc_02',
                        u'DNN_es_p_starting_300m_mp_big_02', u'DNN_es_p_track',
                        u'cgen_E', u'cgen_azi',
                        u'cgen_time', u'cgen_x', u'cgen_y', u'cgen_z', u'cgen_zen',
                        u'cscdSBU_DepthFirstHit', u'cscdSBU_I3XYScale',
                        u'cscdSBU_MaxQtotRatio_HLC', u'cscdSBU_Qtot_HLC',
                        u'cscdSBU_VetoEarliestLayer', u'cscdSBU_VetoMaxDomChargeString',
                        u'cscdSBU_distance_deepcore', u'cscdSBU_distance_icecube',
                        u'cscdl3_cont_tag',# u'dec', 
                        u'es_03_starting',
                        u'es_cscdl2_big_starting', u'es_cscdl3_02_starting'] #u'monopod_E',
                           #u'monopod_azi', #u'monopod_x', u'monopod_y', u'monopod_z',
                           #u'monopod_zen', #u'ow', 
                          # u'ra', u'sigma']
        '''
        features_step01 = [#u'DNNMESE_E', u'DNNMESE_azi', #u'DNNMESE_azi_unc', 
            #u'DNNMESE_x',u'DNNMESE_y', u'DNNMESE_z', 
            #u'DNNMESE_zen', #u'DNNMESE_zen_unc',
            u'DNN_es_length_02', #u'DNN_es_length_log_unc_02',
            u'DNN_es_p_starting_300m_mp_big_02', u'DNN_es_p_track',
            u'cgen_E', u'cgen_azi',
            u'cgen_time', u'cgen_x', u'cgen_y', u'cgen_z', u'cgen_zen',
            u'cscdSBU_DepthFirstHit', u'cscdSBU_I3XYScale',
            u'cscdSBU_MaxQtotRatio_HLC', u'cscdSBU_Qtot_HLC',
            u'cscdSBU_VetoEarliestLayer', u'cscdSBU_VetoMaxDomChargeString',
            u'cscdSBU_distance_deepcore', u'cscdSBU_distance_icecube',
            u'cscdl3_cont_tag']# u'dec', 
            #u'es_03_starting',
            #u'es_cscdl2_big_starting', u'es_cscdl3_02_starting'] #u'monopod_E',
               #u'monopod_azi', #u'monopod_x', u'monopod_y', u'monopod_z',
               #u'monopod_zen', #u'ow', 
              # u'ra', u'sigma']
        '''

        features_step02 =  [u'DNNMESE_E', u'DNNMESE_azi', u'DNNMESE_azi_unc', 
           u'DNNMESE_x', u'DNNMESE_y', u'DNNMESE_z', 
           u'DNNMESE_zen', u'DNNMESE_zen_unc',
           u'DNN_es_length_02', u'DNN_es_length_log_unc_02',
           u'DNN_es_p_starting_300m_mp_big_02', u'DNN_es_p_track',
           u'cgen_E', u'cgen_azi',
           u'cgen_time', u'cgen_x', u'cgen_y', u'cgen_z', u'cgen_zen',
           u'cscdSBU_DepthFirstHit', u'cscdSBU_I3XYScale',
           u'cscdSBU_MaxQtotRatio_HLC', u'cscdSBU_Qtot_HLC',
           u'cscdSBU_VetoEarliestLayer', u'cscdSBU_VetoMaxDomChargeString',
           u'cscdSBU_distance_deepcore', u'cscdSBU_distance_icecube',
           u'cscdl3_cont_tag', u'dec', u'es_03_starting',
           u'es_cscdl2_big_starting', u'es_cscdl3_02_starting', #u'monopod_E',
           #u'monopod_azi', #u'monopod_x', u'monopod_y', u'monopod_z',
           #u'monopod_zen', #u'ow', 
           u'ra', u'sigma']

    elif feat == 'min':
        features_step01 = [#u'DNNMESE_E', u'DNNMESE_azi', u'DNNMESE_azi_unc', 
                #u'DNNMESE_x',u'DNNMESE_y', u'DNNMESE_z', 
                #u'DNNMESE_zen', u'DNNMESE_zen_unc',
                u'DNN_es_length_02', #u'DNN_es_length_log_unc_02',
                u'DNN_es_p_starting_300m_mp_big_02', u'DNN_es_p_track',
                u'cgen_E', u'cgen_azi',
                u'cgen_time', u'cgen_x', u'cgen_y', u'cgen_z', u'cgen_zen',
                u'cscdSBU_DepthFirstHit', u'cscdSBU_I3XYScale',
                u'cscdSBU_MaxQtotRatio_HLC', u'cscdSBU_Qtot_HLC',
                u'cscdSBU_VetoEarliestLayer', u'cscdSBU_VetoMaxDomChargeString',
                u'cscdSBU_distance_deepcore', u'cscdSBU_distance_icecube',
                u'cscdl3_cont_tag']# u'dec', 
                #u'es_03_starting',
                #u'es_cscdl2_big_starting', u'es_cscdl3_02_starting'] #u'monopod_E',
                   #u'monopod_azi', #u'monopod_x', u'monopod_y', u'monopod_z',
                   #u'monopod_zen', #u'ow', 
                  # u'ra', u'sigma']

        features_step02 =  [u'DNNMESE_E', u'DNNMESE_azi', u'DNNMESE_azi_unc', 
           u'DNNMESE_x', u'DNNMESE_y', u'DNNMESE_z', 
           u'DNNMESE_zen', u'DNNMESE_zen_unc',
           u'DNN_es_length_02', u'DNN_es_length_log_unc_02',
           u'DNN_es_p_starting_300m_mp_big_02', u'DNN_es_p_track',
           u'cgen_E', u'cgen_azi',
           u'cgen_time', u'cgen_x', u'cgen_y', u'cgen_z', u'cgen_zen',
           u'cscdSBU_DepthFirstHit', u'cscdSBU_I3XYScale',
           u'cscdSBU_MaxQtotRatio_HLC', u'cscdSBU_Qtot_HLC',
           u'cscdSBU_VetoEarliestLayer', u'cscdSBU_VetoMaxDomChargeString',
           u'cscdSBU_distance_deepcore', u'cscdSBU_distance_icecube',
           u'cscdl3_cont_tag', u'dec', u'es_03_starting',
           u'es_cscdl2_big_starting', u'es_cscdl3_02_starting', #u'monopod_E',
           #u'monopod_azi', #u'monopod_x', u'monopod_y', u'monopod_z',
           #u'monopod_zen', #u'ow', 
           u'ra', u'sigma']
    else:
        features_step01 = [u'DNNMESE_E', u'DNNMESE_azi', u'DNNMESE_azi_unc', 
                        u'DNNMESE_x',u'DNNMESE_y', u'DNNMESE_z', 
                        u'DNNMESE_zen', u'DNNMESE_zen_unc',
                        u'DNN_es_length_02', u'DNN_es_length_log_unc_02',
                        u'DNN_es_p_starting_300m_mp_big_02', u'DNN_es_p_track',
                        u'cgen_E', u'cgen_azi',
                        u'cgen_time', u'cgen_x', u'cgen_y', u'cgen_z', u'cgen_zen',
                        u'cscdSBU_DepthFirstHit', u'cscdSBU_I3XYScale',
                        u'cscdSBU_MaxQtotRatio_HLC', u'cscdSBU_Qtot_HLC',
                        u'cscdSBU_VetoEarliestLayer', u'cscdSBU_VetoMaxDomChargeString',
                        u'cscdSBU_distance_deepcore', u'cscdSBU_distance_icecube',
                        u'cscdl3_cont_tag', u'dec', u'es_03_starting',
                        u'es_cscdl2_big_starting', u'es_cscdl3_02_starting', #u'monopod_E',
                           #u'monopod_azi', #u'monopod_x', u'monopod_y', u'monopod_z',
                           #u'monopod_zen', #u'ow', 
                           u'ra', u'sigma']


        features_step02 =  [u'DNNMESE_E', u'DNNMESE_azi', u'DNNMESE_azi_unc', 
           u'DNNMESE_x', u'DNNMESE_y', u'DNNMESE_z', 
           u'DNNMESE_zen', u'DNNMESE_zen_unc',
           u'DNN_es_length_02', u'DNN_es_length_log_unc_02',
           u'DNN_es_p_starting_300m_mp_big_02', u'DNN_es_p_track',
           u'cgen_E', u'cgen_azi',
           u'cgen_time', u'cgen_x', u'cgen_y', u'cgen_z', u'cgen_zen',
           u'cscdSBU_DepthFirstHit', u'cscdSBU_I3XYScale',
           u'cscdSBU_MaxQtotRatio_HLC', u'cscdSBU_Qtot_HLC',
           u'cscdSBU_VetoEarliestLayer', u'cscdSBU_VetoMaxDomChargeString',
           u'cscdSBU_distance_deepcore', u'cscdSBU_distance_icecube',
           u'cscdl3_cont_tag', u'dec', u'es_03_starting',
           u'es_cscdl2_big_starting', u'es_cscdl3_02_starting', #u'monopod_E',
           #u'monopod_azi', #u'monopod_x', u'monopod_y', u'monopod_z',
           #u'monopod_zen', #u'ow', 
           u'ra', u'sigma']
    print('Year {}'.format(year))
    data = pd.read_hdf('/data/user/ssclafani/data/cscd/IC86_{}/IC86_{}_data_es_test.hdf5'.format(year, year))
    
    print('Data Events in 1 year pre-cut: {}'.format(len(data)))
    #ensure sigmas make sense!
    max_depth_1 = max_depth
    n_estimators_1 = n_estimators
    data['sigma'] = np.where (np.isfinite (data.sigma), data.sigma, np.radians (180))
    data.loc[data.sigma > np.pi, 'sigma'] = np.pi
    if feat == 'slim':
        rf = bdt = load('/data/user/ssclafani/data/cscd/slim_bdt/seed_{}/corsika/bdt/n_cv_{}/bdt_max_depth_{}_n_est_{}lr_{}_seed_{}_train_size_{}.joblib'.format(
                seed, n_cv, max_depth, n_estimators, lr, seed, int(train_size*100.)))
    elif feat == 'min':
        rf = bdt = load('/data/user/ssclafani/data/cscd/min_bdt/seed_{}/corsika/bdt/n_cv_{}/bdt_max_depth_{}_n_est_{}lr_{}_seed_{}_train_size_{}.joblib'.format(
                seed, n_cv, max_depth, n_estimators, lr, seed, int(train_size*100.)))
    else:
        rf = bdt = load('/data/user/ssclafani/data/cscd/bdt/seed_{}/corsika/bdt/n_cv_{}/bdt_max_depth_{}_n_est_{}lr_{}_seed_{}_train_size_{}.joblib'.format(
                seed, n_cv, max_depth, n_estimators, lr, seed, int(train_size*100.)))


    imps = bdt.feature_importances_
    names = np.array(features_step01)
    order = np.argsort(imps)[::-1]
    #print(len(names) , len(order), len(imps))
    for (name, imp) in zip(names[order], imps[order]):
        print('{:30s}{:.6f}'.format(name, imp))
 
    nu_score = rf.predict_proba(data[features_step01])
    mcut = mu_cut
    data['mu_score'] = nu_score[:,1]
    step01_data = data[data['mu_score'] < mcut]
    print('Data surviving Step01: {}'.format(len(step01_data)))

    n_estimators = n_estimators_2 = 500
    max_depth = 5
    step02_rf = load('/data/user/ssclafani/data/cscd/bdt_es_rf_step02_{}_{}_test3.joblib'.format(n_estimators, max_depth))
    
    c_scores = step02_rf.predict_proba(step01_data[features_step02])
    c_scores = c_scores[:,1]
    step01_data['c_score'] = c_scores
    ccut = 0.8 #0.90
    step02_data = step01_data[step01_data['c_score'] > ccut]
    print('Data surviving Step02: {}'.format(len(step02_data)))
    lt_days = livetimes[year]
    livetime = lt_days * 86400
   
        
    mu = pd.read_pickle('/data/user/ssclafani/data/cscd/mu_es_df_{}_{}_lr_0.02test.pickle'.format(n_estimators_1, max_depth_1))
    mu['weight_total'] = mu.weight
    if systematics:
        print('systematics')

        nu = pd.read_hdf('/data/user/ssclafani/data/cscd/IC86_2013/IC86_2013_mc_{}.hdf5'.format(systematics))
    else:
        nu = pd.read_hdf('/data/user/ssclafani/data/cscd/IC86_2013/IC86_2013_egen.hdf5'.format(systematics))
        #nu = pd.read_pickle('/data/user/ssclafani/data/cscd/nu_es_df_{}_{}_lr_0.02test.pickle'.format(n_estimators_1, max_depth_1))
        #nu = pd.read_pickle('/data/user/ssclafani/data/cscd/nu_es_df_{}_{}_lr_0.02test.pickle'.format(n_estimators_1, max_depth_1))

    
    weight_mc = True 
    if weight_mc:
        nu['true_ra'] = nu.trueRa
        nu['true_dec'] = nu.trueDec
        nu['true_energy'] = nu.trueE
        nu['oneweight'] = nu.ow
        nu['dpsi'] = astro.angular_distance (nu.true_ra, nu.true_dec, nu.ra, nu.dec)
        nu['mese_dpsi'] = astro.angular_distance (nu.true_ra, nu.true_dec, nu.DNNMESE_azi, nu.DNNMESE_zen - np.pi/2)
        honda = NewNuFlux.makeFlux ('honda2006')
        honda.knee_reweighting_model = 'gaisserH3a_elbert'
        honda.relative_kaon_contribution = .91
        af = AtmosphericSelfVeto.AnalyticPassingFraction
        honda_veto_mese = af('conventional', veto_threshold=1e2)
        veto_args = nu.trueptype, nu.true_energy, -np.sin (nu.true_dec), 1950. - nu.DNNlabels_true_VertexZ
        phi_atm = np.vectorize (honda.getFlux) (nu.trueptype, nu.true_energy, -np.sin (nu.true_dec))
        veto_atm = honda_veto_mese (*veto_args)

        nu['weight_atm'] = 2* nu.oneweight * phi_atm * veto_atm
        nu['weight_E250'] = 2.23e-18 * nu.oneweight * (nu.true_energy/1e5)**-2.5
        nu['weight_E200'] = 1e-18 * nu.oneweight * (nu.true_energy/1e5)**-2
        nu['weight_E225'] = 1.5e-18 * nu.oneweight * (nu.true_energy/1e5)**-2.25
        nu['weight_total'] = nu.weight_atm + nu.weight_E200
    nu.weight_total = nu.weight_atm + nu.weight_E200

    print(sum(mu.weight*livetime))
    print(sum(nu.weight_total*livetime))

    nu['energy'] = nu.cgen_E
    mu['energy'] = mu.cgen_E
    nu_score = rf.predict_proba(nu[features_step01])
    nu['mu_score'] = nu_score[:,1]
    cut_mc = nu[nu['mu_score'] < mcut]
    print('Neutrinos surviving Step01: {}'.format(sum(cut_mc.weight_total*livetime)))
    print('Neutrinos Atm surviving Step01: {}'.format(sum(cut_mc.weight_atm * livetime)))
    print('Neutrinos E-2 surviving Step01: {}'.format(sum(cut_mc.weight_E200 * livetime)))
    print('Neutrinos E-2.5 surviving Step01: {}'.format(sum(cut_mc.weight_E250 * livetime)))

    nu_score = rf.predict_proba(mu[features_step01])
    mu['mu_score'] = nu_score[:,1]
    cut_mu = mu[mu['mu_score'] < mcut]
    print('Muons surviving Step01: {}'.format(sum(cut_mu.weight_total*livetime)))

    c_scores = step02_rf.predict_proba(cut_mc[features_step02])
    c_scores = c_scores[:,1]
    cut_mc['c_score'] = c_scores
    step02_mc = cut_mc[cut_mc['c_score'] > ccut]
    print('Neutrinos surviving Step02: {}'.format(sum(step02_mc.weight_total * livetime)))
    print('Neutrinos Atm surviving Step01: {}'.format(sum(step02_mc.weight_atm * livetime)))
    print('Neutrinos E-2 surviving Step01: {}'.format(sum(step02_mc.weight_E200 * livetime)))
    print('Neutrinos E-2.5 surviving Step01: {}'.format(sum(step02_mc.weight_E250 * livetime)))
    
    c_scores = step02_rf.predict_proba(cut_mu[features_step02])
    c_scores = c_scores[:,1]
    cut_mu['c_score'] = c_scores
    step02_mu = cut_mu[cut_mu['c_score'] > ccut]
    print('Muons surviving Step02: {}'.format(sum(step02_mu.weight_total * livetime)))
    
    if add_cut:
        #define additional cut here
        
        data_mask = (step02_data.es_03_starting  > .2)
        nu_mask = (step02_mc.es_03_starting  > .2)
        mu_mask = (step02_mu.es_03_starting  > .2)
    
        step02_data = step02_data[data_mask]
        step02_mc = step02_mc[nu_mask]
        step02_mu = step02_mu[mu_mask]
        print('Muons surviving Additional Cut: {}'.format(sum(step02_mu.weight_total * livetime)))
        print('Neutrinos surviving Additional Cut: {}'.format(sum(step02_mu.weight_total * livetime)))
        print('Data surviving Additional Cut: {}'.format(len(step02_data)))

    lvl3 = pd.concat([step02_mu, step02_mc], sort=False)

    ra, dec = astro.dir_to_equa(step02_data.cgen_zen, step02_data.cgen_azi, step02_data.MJD)
    step02_data['cgen_zen'], step02_data['cgen_azi'] = astro.equa_to_dir(ra, dec, step02_data.MJD)
    
    step02_data['logE'] = np.log10(step02_data.cgen_E)
    lvl3['logE'] = np.log10(lvl3.cgen_E)

    mask = np.isnan(step02_data.logE)
    step02_data.loc[mask, 'cgen_E'] = step02_data.loc[mask, 'DNNMESE_E']
    step02_data['logE'] = np.log10(step02_data.cgen_E)

    step02_mc['logE'] = np.log10(step02_mc.cgen_E)
    mask = np.isnan(step02_mc.logE)
    step02_mc.loc[mask, 'cgen_E'] = step02_mc.loc[mask, 'DNNMESE_E']
    step02_mc['logE'] = np.log10(step02_mc.cgen_E)

    lvl3['logE'] = np.log10(lvl3.cgen_E)
    mask = np.isnan(lvl3.logE)
    lvl3.loc[mask, 'cgen_E'] = lvl3.loc[mask, 'DNNMESE_E']
    lvl3['logE'] = np.log10(lvl3.cgen_E)

    step02_mc['time'] = np.random.uniform(step02_data.MJD.min(), step02_data.MJD.max(), len(step02_mc))
    lvl3['time'] = np.random.uniform(step02_data.MJD.min(), step02_data.MJD.max(), len(lvl3))

    step02_data['time'] = step02_data.MJD
    step02_data['run'] = step02_data.Run
    step02_data['event'] = step02_data.Event
    step02_data['ra'] = ra
    step02_data['dec'] = dec
    step02_data['azi'] = step02_data.cgen_azi
    step02_data['zen'] = step02_data.cgen_zen
    step02_data['angErr'] = step02_data.sigma
    step02_data['logE'] = np.log10(step02_data.cgen_E)

    #step02_mc['time'] = step02_mc.MJD
    step02_mc['run'] = step02_mc.Run
    step02_mc['event'] = step02_data.Event
    step02_mc['ra'] = step02_mc.ra
    step02_mc['dec'] = step02_mc.dec
    step02_mc['azi'] = step02_mc.cgen_azi
    step02_mc['zen'] = step02_mc.cgen_zen
    step02_mc['angErr'] = step02_mc.sigma
    step02_mc['logE'] = np.log10(step02_mc.cgen_E)

    lvl3['run'] = lvl3.Run
    lvl3['event'] = step02_data.Event
    lvl3['ra'] = lvl3.ra
    lvl3['dec'] = lvl3.dec
    lvl3['azi'] = lvl3.cgen_azi
    lvl3['zen'] = lvl3.cgen_zen
    lvl3['angErr'] = lvl3.sigma
    lvl3['logE'] = np.log10(lvl3.cgen_E)

    version = '3'
    cutname = mcut 

    #fix_GRL event number
    if os.path.isfile('/data/user/ssclafani/data/analyses/ECAS/version_000_p01/GRL/IC86_{}_exp.npy'.format(year)):
        print('Fixing GRL')
        grl = np.load('/data/user/ssclafani/data/analyses/ECAS/version_000_p01/GRL/IC86_{}_exp.npy'.format(year))
        ns = []
        for run in np.unique(data.Run):
            #print(run)
            mask = (step02_data.Run == run)
            n = len(step02_data[mask])
            #print(n)
            ns = np.append(ns, n)
        print(len(ns))
        print(len(grl['start']))
        grl['events'] = ns
        np.save('/data/user/ssclafani/data/analyses/ECAS/version_000_p01/GRL/IC86_{}_exp.npy'.format(year), grl)
    if systematics:
        step02_mc.to_pickle('/data/user/ssclafani/data/cscd/ECAS/mc_{}_full_{}_v{}_{}_{}.pickle'.format(year, cutname, version, systematics, n_estimators_1))
        np.save('/data/user/ssclafani/data/cscd/ECAS/mc_7yr_{}_v{}_{}_{}_{}.npy'.format(cutname, version, systematics, n_estimators_1, max_depth_1),  
                np.array(zip(step02_mc ['ra'], step02_mc['dec'],  step02_mc['sigma'], step02_mc.logE, step02_mc.time,
                            step02_mc.oneweight, step02_mc.true_energy, step02_mc.true_ra, step02_mc.true_dec), 
                         dtype=[('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8'), ('time', '<f8'), ('oneweight', '<f8'), 
                                ('trueE', '<f8'), ('trueRa', '<f8'), ('trueDec', '<f8')]))
    if feat == 'slim':
        fake_7yrs = step02_data.append(step02_data).append(step02_data).append(step02_data).append(step02_data).append(step02_data).append(step02_data)
        len(fake_7yrs)
        if save == True:
            print('saving Data {}'.format(year))
            ensure_dir('/data/user/ssclafani/data/cscd/ECAS/')
            np.save('/data/user/ssclafani/data/cscd/ECAS/data_1yr_{}_{}_v{}_slim.npy'.format(year, cutname, version),  
                np.array(zip(step02_data.time, step02_data.run, step02_data.event, step02_data ['ra'], step02_data['dec'],  step02_data['sigma'] , step02_data['logE']), 
                dtype=[('time', '<f8'), ('run', '<f8'), ('event', '<f8'), ('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8')]))
            np.save('/data/user/ssclafani/data/cscd/ECAS/data_1yr_{}_{}_v{}_slim.npy'.format(year, cutname, version),  
                np.array(zip(step02_data.time, step02_data.run, step02_data.event, step02_data ['ra'], step02_data['dec'],  step02_data['sigma'] , step02_data['logE']), 
                dtype=[('time', '<f8'), ('run', '<f8'), ('event', '<f8'), ('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8')]))

            sevenyears = False 
            if sevenyears:
                data = {}
                print('saving 7yrs')
                ys = ['2012', '2013', '2014']
                for y in ys:
                    print(y)
                    print('/data/user/ssclafani/data/cscd/ECAS/data_1yr_{}_{}_v{}_slim.npy'.format(y, cutname, version))
                    data[y] = np.load('/data/user/ssclafani/data/cscd/ECAS/data_1yr_{}_{}_v{}_slim.npy'.format(y, cutname, version))
                    print(y, len(data[y]))
                lt_years = 7
                data_events_per_year = int((len(data['2013']) + len(data['2014'])) / 2.)
                print(data_events_per_year) 
                type = 'usealldata'
                full_data = np.concatenate((data['2012'], data['2013'], data['2014']), axis=0)
                def create_n_years(n=7, type='usealldata'):
                    if type == 'oneyear':
                        seven_yr_data = np.concatenate((data2014, data2014, data2014, data2014, data2014, data2014, data2014))
                    elif type == 'usealldata':
                        six_year_data = np.append(full_data, full_data)
                        print('Double data = {}'.format(len(six_year_data)))
                        n_new_events =  (lt_years*data_events_per_year) - len(six_year_data) 
                        print('New events to sample {}'.format(n_new_events))
                        new_events = np.random.choice(full_data, n_new_events)
                        seven_yr_data = np.append(six_year_data, new_events)
                        print('Total Events: {}'.format(len(seven_yr_data)))
                    elif type == 'fullsample':
                        seven_yr_data = np.random.choice(full_data, lt_years*data_events_per_year, replace=True)
                    return seven_yr_data

                seven_yr_data = create_n_years(type='usealldata')
                np.save('/data/user/ssclafani/data/cscd/ECAS/data_7yr_5e-05_v2_slim.npy', seven_yr_data)       
            print(sum(step02_mc.weight_E200))
            print('saving mc')
            np.save('/data/user/ssclafani/data/cscd/mc_7yr_{}_v{}_slim_{}_{}.npy'.format(cutname, version, n_estimators_1, max_depth_1),  
                    np.array(zip(step02_mc ['ra'], step02_mc['dec'],  step02_mc['sigma'], step02_mc.logE, step02_mc.time,
                                step02_mc.oneweight, step02_mc.true_energy, step02_mc.true_ra, step02_mc.true_dec), 
                             dtype=[('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8'), ('time', '<f8'), ('oneweight', '<f8'), 
                                    ('trueE', '<f8'), ('trueRa', '<f8'), ('trueDec', '<f8')]))     
#if save_to_ana:
            #    print('Warning: Overwriting data in analyses directory')
            #    np.save('/data/user/ssclafani/data/analyses/ECAS/version_000_p01/IC86_{}_exp.npy'.format(year, cutname, version),  
            #        np.array(zip(step02_data.time, step02_data.run, step02_data.event, step02_data ['ra'], step02_data['dec'],  step02_data['sigma'] , step02_data['logE']), 
            #        dtype=[('time', '<f8'), ('run', '<f8'), ('event', '<f8'), ('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8')]))
                
            step02_data.to_pickle('/data/user/ssclafani/data/cscd/ECAS/data_{}_full_{}_v{}_slim_{}_{}.pickle'.format(year, cutname, version, n_estimators_1, max_depth_1))
            step02_mc.to_pickle('/data/user/ssclafani/data/cscd/ECAS/mc_{}_full_{}_v{}_slim_{}.pickle'.format(year, cutname, version, n_estimators_1))
            step02_mu.to_pickle('/data/user/ssclafani/data/cscd/ECAS/mu_{}_full_{}_v{}_slim_{}.pickle'.format(year, cutname, version, n_estimators_1))
    elif feat == 'min':
        fake_7yrs = step02_data.append(step02_data).append(step02_data).append(step02_data).append(step02_data).append(step02_data).append(step02_data)
        len(fake_7yrs)
        if save == True:
            print('saving Data {}'.format(year))
            ensure_dir('/data/user/ssclafani/data/cscd/ECAS/')
            np.save('/data/user/ssclafani/data/cscd/ECAS/data_1yr_{}_{}_v{}_min.npy'.format(year, cutname, version),  
                np.array(zip(step02_data.time, step02_data.run, step02_data.event, step02_data ['ra'], step02_data['dec'],  step02_data['sigma'] , step02_data['logE']), 
                dtype=[('time', '<f8'), ('run', '<f8'), ('event', '<f8'), ('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8')]))

            sevenyears = True 
            if sevenyears:
                data = {}
                print('saving 7yrs')
                ys = ['2012', '2013', '2014']
                for y in ys:
                    print(y)
                    print('/data/user/ssclafani/data/cscd/ECAS/data_1yr_{}_{}_v{}_min.npy'.format(y, cutname, version))
                    data[y] = np.load('/data/user/ssclafani/data/cscd/ECAS/data_1yr_{}_{}_v{}_min.npy'.format(y, cutname, version))
                    print(y, len(data[y]))
                lt_years = 7
                data_events_per_year = int((len(data['2013']) + len(data['2014'])) / 2.)
                print(data_events_per_year) 
                type = 'usealldata'
                full_data = np.concatenate((data['2012'], data['2013'], data['2014']), axis=0)
                def create_n_years(n=7, type='usealldata'):
                    if type == 'oneyear':
                        seven_yr_data = np.concatenate((data2014, data2014, data2014, data2014, data2014, data2014, data2014))
                    elif type == 'usealldata':
                        six_year_data = np.append(full_data, full_data)
                        print('Double data = {}'.format(len(six_year_data)))
                        n_new_events =  (lt_years*data_events_per_year) - len(six_year_data) 
                        print('New events to sample {}'.format(n_new_events))
                        new_events = np.random.choice(full_data, n_new_events)
                        seven_yr_data = np.append(six_year_data, new_events)
                        print('Total Events: {}'.format(len(seven_yr_data)))
                    elif type == 'fullsample':
                        seven_yr_data = np.random.choice(full_data, lt_years*data_events_per_year, replace=True)
                    return seven_yr_data

                seven_yr_data = create_n_years(type='usealldata')
                np.save('/data/user/ssclafani/data/cscd/ECAS/data_7yr_5e_05_v2_min.npy', seven_yr_data)       
            print(sum(step02_mc.weight_E200))
            print('saving mc')
            np.save('/data/user/ssclafani/data/cscd/mc_7yr_{}_v{}_min_{}_{}.npy'.format(cutname, version, n_estimators_1, max_depth_1),  
                    np.array(zip(step02_mc ['ra'], step02_mc['dec'],  step02_mc['sigma'], step02_mc.logE, step02_mc.time,
                                step02_mc.oneweight, step02_mc.true_energy, step02_mc.true_ra, step02_mc.true_dec), 
                             dtype=[('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8'), ('time', '<f8'), ('oneweight', '<f8'), 
                                    ('trueE', '<f8'), ('trueRa', '<f8'), ('trueDec', '<f8')]))     
#if save_to_ana:
            #    print('Warning: Overwriting data in analyses directory')
            #    np.save('/data/user/ssclafani/data/analyses/ECAS/version_000_p01/IC86_{}_exp.npy'.format(year, cutname, version),  
            #        np.array(zip(step02_data.time, step02_data.run, step02_data.event, step02_data ['ra'], step02_data['dec'],  step02_data['sigma'] , step02_data['logE']), 
            #        dtype=[('time', '<f8'), ('run', '<f8'), ('event', '<f8'), ('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8')]))
                
            step02_data.to_pickle('/data/user/ssclafani/data/cscd/ECAS/data_{}_full_{}_v{}_min_{}_{}.pickle'.format(year, cutname, version, n_estimators_1, max_depth_1))
            step02_mc.to_pickle('/data/user/ssclafani/data/cscd/ECAS/mc_{}_full_{}_v{}_min_{}.pickle'.format(year, cutname, version, n_estimators_1))
            step02_mu.to_pickle('/data/user/ssclafani/data/cscd/ECAS/mu_{}_full_{}_v{}_min_{}.pickle'.format(year, cutname, version, n_estimators_1))
    else:
        if save == True:
            if add_cut:
                ensure_dir('/data/user/ssclafani/data/cscd/ECAS/')
                np.save('/data/user/ssclafani/data/cscd/ECAS/data_1yr_{}_{}_addcut.npy'.format(year, cutname),  
                    np.array(zip(step02_data.time, step02_data.run, step02_data.event, step02_data ['ra'], step02_data['dec'],  step02_data['sigma'] , step02_data['logE']), 
                    dtype=[('time', '<f8'), ('run', '<f8'), ('event', '<f8'), ('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8')]))
                
                print('saving mc with additional cuts!')
                np.save('/data/user/ssclafani/data/cscd/mc_7yr_{}_add_cut.npy'.format(cutname),  
                        np.array(zip(step02_mc ['ra'], step02_mc['dec'],  step02_mc['sigma'], step02_mc.logE, step02_mc.time,
                                    step02_mc.oneweight, step02_mc.true_energy, step02_mc.true_ra, step02_mc.true_dec), 
                                 dtype=[('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8'), ('time', '<f8'), ('oneweight', '<f8'), 
                                        ('trueE', '<f8'), ('trueRa', '<f8'), ('trueDec', '<f8')]))     
            else: 
                ensure_dir('/data/user/ssclafani/data/cscd/ECAS/')
                np.save('/data/user/ssclafani/data/cscd/ECAS/data_1yr_{}_{}_v2_test.npy'.format(year, cutname),  
                    np.array(zip(step02_data.time, step02_data.run, step02_data.event, step02_data ['ra'], step02_data['dec'],  step02_data['sigma'] , step02_data['logE']), 
                    dtype=[('time', '<f8'), ('run', '<f8'), ('event', '<f8'), ('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8')]))
                
                print('saving mc')
                np.save('/data/user/ssclafani/data/cscd/mc_7yr_{}_v2_test_2001_8.npy'.format(cutname),  
                        np.array(zip(step02_mc ['ra'], step02_mc['dec'],  step02_mc['sigma'], step02_mc.logE, step02_mc.time,
                                    step02_mc.oneweight, step02_mc.true_energy, step02_mc.true_ra, step02_mc.true_dec), 
                                 dtype=[('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8'), ('time', '<f8'), ('oneweight', '<f8'), 
                                        ('trueE', '<f8'), ('trueRa', '<f8'), ('trueDec', '<f8')]))     
                step02_data.to_pickle('/data/user/ssclafani/data/cscd/ECAS/data_{}_full_{}_v{}_test_{}_{}.pickle'.format(year, cutname, version, n_estimators_1, max_depth_1))
                step02_mc.to_pickle('/data/user/ssclafani/data/cscd/ECAS/mc_{}_full_{}_v{}_test_{}.pickle'.format(year, cutname, version, n_estimators_1))
                step02_mu.to_pickle('/data/user/ssclafani/data/cscd/ECAS/mu_{}_full_{}_v{}_test_{}.pickle'.format(year, cutname, version, n_estimators_1))

            if save_to_ana:
                print('Warning: Overwriting data in analyses directory')
                np.save('/data/user/ssclafani/data/analyses/ECAS/version_000_p01/IC86_{}_exp.npy'.format(year, cutname, version),  
                    np.array(zip(step02_data.time, step02_data.run, step02_data.event, step02_data ['ra'], step02_data['dec'],  step02_data['sigma'] , step02_data['logE']), 
                    dtype=[('time', '<f8'), ('run', '<f8'), ('event', '<f8'), ('ra', '<f8') , ('dec', '<f8') , ('angErr', '<f8') , ('logE', '<f8')]))
                

if __name__ == '__main__':
    exe_t0 = now ()
    print (' start at {} .'.format (exe_t0))
    cli ()

