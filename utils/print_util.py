
def print_results(step, logger, config):
    if config.verbose:
        try:
            if config.algorithm.name == "sdss_vp":
                print(f'\nStep {int(step)}')
                print(f'ELBO {float(logger["KL/elbo"][-1]):.05}; SD loss {float(logger["train/sd"][-1]):.05}') 
                print(f'∆ lnZ {float(logger["logZ/delta_reverse"][-1]):.05}; Reverse ESS {float(logger["ESS/reverse"][-1]):.05}')
            else:
                print(f'\nStep {int(step)}: ELBO {float(logger["KL/elbo"][-1]):.05}; lnZ {float(logger["logZ/reverse"][-1]):.05}; reverse_ESS {float(logger["ESS/reverse"][-1]):.05}')
                try:
                    print(f'∆ lnZ {float(logger["logZ/delta_reverse"][-1]):.05}')
                except:
                    pass
        except:
            print(f'Step {int(step)}: ELBO {float(logger["KL/elbo"][-1])}; lnZ {float(logger["logZ/reverse"][-1])}')
