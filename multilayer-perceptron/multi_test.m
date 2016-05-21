
% rand("seed", 1) set good seed

% nets = generate_nets([2 15 1]);
% example of use:
% example of use:
% example of use: terrain_training_test([2 15 1], 0.001, @tanh_ft, @tanh_ft_der, 0.5, 0.5, 1, 3, false, 0.9, 0.2, 0.05, 5)

% example of use: terrain_training_test([2 5 2 1], 0.001, @tanh_ft, @tanh_ft_der, 0.5, 1, 2, 1, false)
% example of use: terrain_training_test([2 7 1], 0.001, @tanh_ft, @tanh_ft_der, 0.2, 0.2, 2, 2, false, 0.9)
% example of use: terrain_training_test([2 5 2 1], 0.001, @tanh_ft, @tanh_ft_der, 0.2, 0.5, 2, 3, false, 0.9, 0.05, 0.1, 10)

% learningType: 1 -> Batch, 2 -> Incremental
% algorithm: 1 -> Original, 2 -> Momentum, 3 -> Adaptative etha
% K: number of positive steps befour changing etha on adaptative etha algorithm

% rand("seed", 1) set good seed

% nets = generate_nets([2 15 1]);
% example of use: terrain_training_test([2 20 1], 0.001, @tanh_ft, @tanh_ft_der, 0.2, 0.2, 1, 1, false)
% example of use: terrain_training_test([2 4 1], 0.001, @tanh_ft, @tanh_ft_der, 0.3, 0.2, 1, 2, false, 0.9)
% example of use: terrain_training_test([2 15 1], 0.001, @tanh_ft, @tanh_ft_der, 0.5, 0.4, 1, 3, false, 0.9, 0.2, 0.05, 11)

% example of use: terrain_training_test([2 18 1], 0.0005, @tanh_ft, @tanh_ft_der, 0.2, 0.5, 2, 1, false)
% example of use: terrain_training_test([2 9 1], 0.001, @tanh_ft, @tanh_ft_der, 0.2, 0.2, 2, 2, false, 0.9)
% example of use: terrain_training_test([2 11 1], 0.001, @tanh_ft, @tanh_ft_der, 0.2, 0.2, 2, 3, false, 0.9, 0.2, 0.1, 5)

% learningType: 1 -> Batch, 2 -> Incremental
% algorithm: 1 -> Original, 2 -> Momentum, 3 -> Adaptative etha
% K: number of positive steps befour changing etha on adaptative etha algorithm

function ret = multi_test(g, g_der, n, betha, alpha = 0, a = 0, b = 0, K = 0)
    algorithms = {
                     @multilayer_perceptron_batch,
                     @multilayer_perceptron_batch_momentum,
                     @multilayer_perceptron_batch_adaptative_etha,
                     @multilayer_perceptron_incremental,
                     @multilayer_perceptron_incremental_momentum,
                     @multilayer_perceptron_incremental_adaptative_etha
                };

    data_filename = 'terrain8modif.txt';

    training_set = get_training_set(data_filename);
    maximum = max(max(training_set));

    inputs = [training_set(:, 1) training_set(:, 2)];
    s = [training_set(:, 3)];

    t{1} = inputs./maximum;
    t{2} = s./maximum;


    % now that the net is trained with the training_set lets
    % see which output generates for the complete data set.
    complete_data_set = load('-ascii', data_filename);
    complete_data_set_inputs = [complete_data_set(:, 1) complete_data_set(:, 2)];
    complete_data_set_maximum = max(max(complete_data_set));
    for j = 1 : 20
        net_structure = [2 j 1]
        for i = 1 : 6
            rand('seed', 1);
            nets = generate_nets(net_structure);

            % training net with selected training_set
            fun = algorithms{i}
            resolved_nets = fun(nets, t, 0.001, g, g_der, n, betha, false, alpha, a, b, K);

            layer_outputs = forward_step(complete_data_set_inputs./complete_data_set_maximum, resolved_nets, g, betha);

            ret = (layer_outputs(size(resolved_nets)(2)){1}).*complete_data_set_maximum;

            complete_data_set_error = get_error(size(nets)(2), complete_data_set(:, 3)./complete_data_set_maximum, layer_outputs)
        end
    end
end
