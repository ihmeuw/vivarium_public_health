def parameter_space():
    for granularity in [0.5]:
        for summer in np.linspace(0.4, 1, 10):
            for winter in np.linspace(0.2, 1, 20):
                print('qsub /homes/alecwd/ceam_development/mumps_runner.sh {} {} {}'.format(summer, winter, granularity))
