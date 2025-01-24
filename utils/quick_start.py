# coding: utf-8

from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    
    
    #print(type(config))
    #aaa
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))
    
    #train_dataset, valid_dataset, test_dataset: [userID, itemID]

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)    #[3, 2048]
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()  #vaild_metric: recall@20
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        #print('i is {}'.format(i))
        hyper_ls.append(config[i] or [None])
    #print('the config[hyperparameters] is {}'.format(config['hyper_parameters']))
    
    #aaa
  

    # combinations
    combinators = list(product(*hyper_ls))
    #print('############')
    #print('combinators is {}'.format(combinators))
    #aaa
    
    total_loops = len(combinators)
    #print('total_loops is {}'.format(total_loops))  #1
    #aaa
    
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()  #对item进行重排序
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)  #models.lgmrec

        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid, best_epoch_idx = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid, best_epoch_idx))

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        
        logger.info('████Current BEST████:\nParameters: {}={},\tbest epoch: {},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], hyper_ret[best_test_idx][3],
            dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v, best_epoch_idx) in hyper_ret:
        logger.info('\tParameters: {}={}, \tbest epoch: {}, \n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, best_epoch_idx, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={}, \tbest epoch: {}, \nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   hyper_ret[best_test_idx][3],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))
    
    # import numpy as np
    # _, i_v_feats = model.agg_mm_neighbors('v')
    # np.save('log/'+config['model'] + '-' + config['dataset']+'-i_v_feats', i_v_feats.detach().cpu().numpy())
    # _, i_t_feats = model.agg_mm_neighbors('t')
    # np.save('log/'+config['model'] + '-' + config['dataset']+'-i_t_feats', i_t_feats.detach().cpu().numpy())

