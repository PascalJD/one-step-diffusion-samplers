def get_train_fn(alg_name):
    if alg_name == 'dis':
        from algorithms.dis.dis_trainer import dis_trainer
        return dis_trainer
    
    elif alg_name == 'disk':
        from algorithms.disk.disk_trainer import disk_trainer
        return disk_trainer
    
    elif alg_name == 'pis':
        from algorithms.pis.pis_trainer import pis_trainer
        return pis_trainer
    
    elif alg_name == 'dds':
        from algorithms.dds.dds_trainer import dds_trainer
        return dds_trainer
        
    elif alg_name == "sdss_vp":
        from algorithms.sdss_vp.trainer import trainer
        return trainer
    else:
        raise ValueError(f'No algorithm named {alg_name}.')
