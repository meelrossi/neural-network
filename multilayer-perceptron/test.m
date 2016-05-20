
% example of use: test(@xor, @exp_ft, @exp_ft_der, 0.5, 1, 1, 1, false)
% example of use: test(@xor, @tanh_ft, @tanh_ft_der, 0.5, 1, 1, 2, false, 0.9)
% example of use: test(@xor, @tanh_ft, @tanh_ft_der, 0.2, 0.5, 1, 3, false, 0.9, 0.05, 0.1, 20)
% learningType: 1 -> Batch, 2 -> Incremental
% algorithm: 1 -> Original, 2 -> Momentum, 3 -> Adaptative etha
% K: number of positive steps befour changing etha on adaptative etha algorithm
function ret = test(ft, g, g_der, n, betha, learningType, algorithm, graphics, alpha = 0, a = 0, b = 0, K = 0)
    inputs = [1 1 1; 1 0 1; 0 1 1; 0 0 1; 0 1 0; 1 0 0; 1 1 0; 0 0 0];
    % inputs = [1 1; 1 0; 0 1; 0 0];
    algorithms = {
                    {
                     @multilayer_perceptron_batch,
                     @multilayer_perceptron_batch_momentum,
                     @multilayer_perceptron_batch_adaptative_etha
                    }
                    {
                     @multilayer_perceptron_incremental,
                     @multilayer_perceptron_incremental_momentum,
                     @multilayer_perceptron_incremental_adaptative_etha
                    }
                };

    s = ft(inputs);

    t{1} = inputs;
    t{2} = s;

    err = 0.001;

    nets = generate_nets([3 5 2 1]);

    fun = algorithms{learningType}{algorithm};

    resolved_nets = fun(nets, t, err, g, g_der, n, betha, graphics, alpha, a, b, K);

    layer_outputs = forward_step(inputs, resolved_nets, g, betha);

    ret = layer_outputs(size(resolved_nets)(2));
end